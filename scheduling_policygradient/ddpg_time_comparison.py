import tensorflow as tf
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import gzip, pickle

from scheduling_policygradient.imports.simulation import Simulation
from scheduling_policygradient.policygradient_config import Config
from scheduling_policygradient.imports.scheduling import (maximum_throughput_scheduler,
                                                          max_min_fair_scheduler, delay_sensitive_scheduler)

config = Config()
config.num_episodes = 100
config.steps_per_episode = 5


def gather_state(sim):
    # Gathers a state vector for algorithm input
    state = np.array([], dtype='float32')

    # queue length per user-------------
    requests = np.array([])
    for user in sim.users.values():
        requests = np.append(requests, user.units_requested)
    # if np.sum(requests) > 0:
    #     requests = requests / np.sum(requests)
    state = np.append(state, requests)

    # min time to timeout per user----------
    for user in sim.users.values():
        timeouts = np.array([])
        weighted_size = np.array([])
        timeout_size = np.array([])
        for job in user.jobs:
            timeouts = np.append(timeouts, 1 / (user.latency_max - job.delay))
            # weighted_size = np.append(weighted_size, timeouts[-1] * job.size)
            timeout_size = np.append(timeout_size, job.size)
        if len(timeouts) > 0:
            state = np.append(state, np.max(timeouts))
            # state = np.append(state, np.mean(timeouts))
            # state = np.append(state, weighted_size[np.argmax(timeouts)])
            state = np.append(state, timeout_size[np.argmax(timeouts)])
        else:
            state = np.append(state, 0)
            # state = np.append(state, 10)
            state = np.append(state, 0)

    # channel state per user-----------------
    for user in sim.users.values():
        state = np.append(state, user.channel_quality)

    # Queue length weighted by channel state per user--------------
    # for user in sim.users.values():
    #     quality = user.channel_quality
    #     length = user.units_requested
    #     state = np.append(state, quality * length)

    # data rate satisfaction
    # for user in sim.users.values():  # Gather datarate satisfaction
    #     state = np.append(state, user.datarate_satisfaction)

    return state

def allocation_post_processing(unfixed_allocation_vector, sim, config):
    # Takes a policy allocation vector and performs some sanity checks on it
    # Outputs a ,,valid'' policy vector
    # similar to [1]
    # [1] A Deep Reinforcement Learning-based Approach toDynamic eMBB/URLLC Multiplexing in 5G NR, huang
    fixed_allocation_vector = unfixed_allocation_vector.copy()

    # zero out allocation with no jobs queued
    for user in sim.users.values():
        if user.units_requested == 0:
            fixed_allocation_vector[user.user_id] = 0

    # special case: no resources requested at all by people who were allocated any
    if sum(fixed_allocation_vector) == 0:
        return np.zeros(config.num_users)

    # normalize sum to 1
    if np.sum(fixed_allocation_vector) > 0:
        fixed_allocation_vector = fixed_allocation_vector / np.sum(fixed_allocation_vector)

    # work with resource blocks instead of proportional allocation from here on
    fixed_allocation_vector = fixed_allocation_vector * config.num_channels

    # round allocations
    fixed_allocation_vector = np.round(fixed_allocation_vector, 0)

    # limit allocation to min(allocated, requested)
    for user in sim.users.values():
        fixed_allocation_vector[user.user_id] = min(
            fixed_allocation_vector[user.user_id], user.units_requested)

    # allocate remaining resources to remaining requests, by largest allocation
    remaining_resources = config.num_channels - np.sum(fixed_allocation_vector)
    remaining_requests = np.zeros(config.num_users)
    for user in sim.users.values():
        remaining_requests[user.user_id] = user.units_requested - fixed_allocation_vector[user.user_id]

    allocation_vector = unfixed_allocation_vector.copy()
    while (sum(allocation_vector) > 0) and (remaining_resources > 0):
        largest_allocation_id = np.argmax(allocation_vector)
        if remaining_requests[largest_allocation_id] > 0:
            allocation = min(remaining_requests[largest_allocation_id], remaining_resources)
            fixed_allocation_vector[largest_allocation_id] += allocation
            remaining_resources -= allocation
        allocation_vector[largest_allocation_id] = 0

    return fixed_allocation_vector

actor = tf.keras.models.load_model(
    'C:\\Py\\MasterThesis\\resourceallocation\\scheduling_policygradient\\SavedModels\\ddpg\\actor')

actor.summary()

sim_ddpg = Simulation(config=config)
sim_mt = Simulation(config=config)
sim_mmf = Simulation(config=config)
sim_ds = Simulation(config=config)

print('Testing run times.. 0%', end='')

mean_run_times_per_episode = np.zeros((4, config.num_episodes))
for episode_id in range(config.num_episodes):
    run_times_per_step = np.zeros((4, config.steps_per_episode))
    # ddpg---------------------------------
    for step_id in range(config.steps_per_episode):
        state = gather_state(sim_ddpg)
        start_time = default_timer()
        actor_allocation = actor(state[np.newaxis]).numpy().squeeze()
        end_time = default_timer()
        allocation = allocation_post_processing(actor_allocation, sim=sim_ddpg, config=config)
        run_times_per_step[0, step_id] = end_time - start_time

        sim_ddpg.step(allocation)
        sim_ddpg.generate_jobs(chance=config.new_job_chance)

    mean_run_times_per_episode[0, episode_id] = np.mean(run_times_per_step[0, :])
    # mt-----------------------------------
    for step_id in range(config.steps_per_episode):
        start_time = default_timer()
        allocation = maximum_throughput_scheduler(
            users=list(sim_mt.users.values()),
            resources_available=config.num_channels)
        end_time = default_timer()
        run_times_per_step[1, step_id] = end_time - start_time

        sim_mt.step(allocation)
        sim_mt.generate_jobs(chance=config.new_job_chance)

    mean_run_times_per_episode[1, episode_id] = np.mean(run_times_per_step[1, :])
    # mmf----------------------------------
    for step_id in range(config.steps_per_episode):
        start_time = default_timer()
        allocation = max_min_fair_scheduler(
            users=list(sim_mmf.users.values()),
            resources_available=config.num_channels)
        end_time = default_timer()
        run_times_per_step[2, step_id] = end_time - start_time

        sim_mmf.step(allocation)
        sim_mmf.generate_jobs(chance=config.new_job_chance)

    mean_run_times_per_episode[2, episode_id] = np.mean(run_times_per_step[2, :])
    # ds-----------------------------------
    for step_id in range(config.steps_per_episode):
        start_time = default_timer()
        allocation = delay_sensitive_scheduler(
            users=list(sim_ds.users.values()),
            resources_available=config.num_channels)
        end_time = default_timer()
        run_times_per_step[3, step_id] = end_time - start_time

        sim_ds.step(allocation)
        sim_ds.generate_jobs(chance=config.new_job_chance)

    mean_run_times_per_episode[3, episode_id] = np.mean(run_times_per_step[3, :])
    # reset---------------------------------
    sim_ddpg.reset()
    sim_mt.reset()
    sim_mmf.reset()
    sim_ds.reset()

    # progress update
    if np.mod(episode_id, 100) == 0:
        progress = np.round(episode_id / config.num_episodes * 100, 1)
        print('\rTesting run times..', progress, '%', end='', flush=True)

print('\rTesting run times.. done.', flush=True)

with gzip.open(
        'C:\\Py\\MasterThesis\\resourceallocation\\scheduling_policygradient\\logs\\testing_run_times.gstor',
        'wb') as file:
    pickle.dump(mean_run_times_per_episode, file)

# Plotting
plt.figure()
plt.plot(mean_run_times_per_episode[0, :])
plt.plot(mean_run_times_per_episode[1, :])
plt.plot(mean_run_times_per_episode[2, :])
plt.plot(mean_run_times_per_episode[3, :])

plt.legend(['DDPG', 'MT', 'MMF', 'DS'])

plt.figure()
plt.barh(0, np.mean(mean_run_times_per_episode[0, :]))
plt.barh(1, np.mean(mean_run_times_per_episode[1, :]))
plt.barh(2, np.mean(mean_run_times_per_episode[2, :]))
plt.barh(3, np.mean(mean_run_times_per_episode[3, :]))

plt.yticks([0, 1, 2, 3], ['DDPG', 'MT', 'MMF', 'DS'])

plt.show()
