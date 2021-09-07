import gzip
import pickle
from os.path import join
from shutil import copy2

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scheduling_policygradient.imports.actor_critic import ActorCritic
from scheduling_policygradient.imports.scheduling import delay_sensitive_scheduler
from scheduling_policygradient.imports.scheduling import max_min_fair_scheduler
from scheduling_policygradient.imports.scheduling import maximum_throughput_scheduler
from scheduling_policygradient.imports.scheduling import random_scheduler
from scheduling_policygradient.imports.simulation import Simulation
from scheduling_policygradient.policygradient_config import Config


class Runner:
    def __init__(self):
        self.config = Config()

        self.ddpg_simulation = Simulation(config=self.config)

        self.actor_critic = ActorCritic(size_state=3 * self.config.num_users, size_action=self.config.num_users,
                                        num_hidden=self.config.num_hidden,
                                        learning_rate_q=self.config.learning_rate_q,
                                        learning_rate_a=self.config.learning_rate_a,
                                        batch_size=self.config.batch_size,
                                        min_experiences=self.config.min_experiences,
                                        max_experiences=self.config.max_experiences,
                                        min_priority=self.config.min_priority,
                                        prioritization_factors=self.config.prioritization_factors,
                                        gamma=self.config.gamma,
                                        loss_function=self.config.loss_function)

        self.last_checkpoint_episode = 0
        self.last_checkpoint_reward = 0

    def gather_state(self):
        # Gathers a state vector for algorithm input
        state = np.array([], dtype='float32')

        # queue length per user-------------
        requests = np.array([])
        for user in self.ddpg_simulation.users.values():
            requests = np.append(requests, user.units_requested)
        # if np.sum(requests) > 0:  # normalize requests vector
        #     requests = requests / sum(requests)
        # state = np.append(state, requests)  # append requests as is
        state = np.append(state, requests / self.config.num_channels)  # or relative to available res

        # channel state per user-----------------
        for user in self.ddpg_simulation.users.values():
            state = np.append(state, user.channel_quality)

        # min time to timeout per user----------
        for user in self.ddpg_simulation.users.values():
            timeouts = np.array([])
            # weighted_size = np.array([])
            # timeout_size = np.array([])
            for job in user.jobs:
                timeouts = np.append(timeouts, 1/(user.latency_max - job.delay))
                # weighted_size = np.append(weighted_size, timeouts[-1] * job.size)
                # timeout_size = np.append(timeout_size, job.size)
            if len(timeouts) > 0:
                state = np.append(state, np.max(timeouts))
                # state = np.append(state, np.mean(timeouts))
                # state = np.append(state, weighted_size[np.argmax(timeouts)])
                # state = np.append(state, timeout_size[np.argmax(timeouts)])
            else:
                state = np.append(state, 0)
                # state = np.append(state, 10)
                # state = np.append(state, 0)
                # state = np.append(state, 0)

        # Queue length weighted by channel state per user--------------
        # for user in self.ddpg_simulation.users.values():
        #     quality = user.channel_quality
        #     length = user.units_requested
        #     state = np.append(state, quality * length)

        return state

    @staticmethod
    def add_noise(allocation_vector, noise_multiplier):
        # Adds noise to a policy vector, for exploration
        noise = noise_multiplier * (np.random.rand(len(allocation_vector)) - 0.5)

        return allocation_vector + noise

    def allocation_post_processing(self, unfixed_allocation_vector):
        # Takes a policy allocation vector and performs some sanity checks on it
        # Outputs a ,,valid'' policy vector
        # similar to [1]
        # [1] A Deep Reinforcement Learning-based Approach toDynamic eMBB/URLLC Multiplexing in 5G NR, huang
        fixed_allocation_vector = unfixed_allocation_vector.copy()

        # zero out allocation with no jobs queued
        for user in self.ddpg_simulation.users.values():
            if user.units_requested == 0:
                fixed_allocation_vector[user.user_id] = 0
            else:
                if fixed_allocation_vector[user.user_id] == 0:
                    fixed_allocation_vector[user.user_id] += 0.001

        # special case: no resources requested at all
        if sum(fixed_allocation_vector) == 0:
            return np.zeros(self.config.num_users)

        # normalize sum to 1
        fixed_allocation_vector = fixed_allocation_vector / np.sum(fixed_allocation_vector)

        # work with resource blocks instead of proportional allocation from here on
        fixed_allocation_vector = fixed_allocation_vector * self.config.num_channels

        # round allocations
        fixed_allocation_vector = np.floor(fixed_allocation_vector)

        # limit allocation to min(allocated, requested)
        for user in self.ddpg_simulation.users.values():
            fixed_allocation_vector[user.user_id] = min(
                fixed_allocation_vector[user.user_id], user.units_requested)

        # allocate remaining resources to remaining requests, by largest allocation
        remaining_resources = self.config.num_channels - np.sum(fixed_allocation_vector)
        remaining_requests = np.zeros(self.config.num_users)
        for user in self.ddpg_simulation.users.values():
            remaining_requests[user.user_id] = user.units_requested - fixed_allocation_vector[user.user_id]

        allocation_vector = unfixed_allocation_vector.copy() + 0.001
        while (sum(allocation_vector) > 0) and (remaining_resources > 0):
            largest_allocation_id = np.argmax(allocation_vector)
            if remaining_requests[largest_allocation_id] > 0:
                allocation = min(remaining_requests[largest_allocation_id], remaining_resources)
                fixed_allocation_vector[largest_allocation_id] += allocation
                remaining_resources -= allocation
            allocation_vector[largest_allocation_id] = 0

        return fixed_allocation_vector

    def allocation_post_processing_alt(self, unfixed_allocation_vector):
        fixed_allocation_vector = unfixed_allocation_vector.copy()
        # zero out allocation with no jobs queued
        for user in self.ddpg_simulation.users.values():
            if user.units_requested == 0:
                fixed_allocation_vector[user.user_id] = 0
            else:
                if fixed_allocation_vector[user.user_id] == 0:
                    fixed_allocation_vector[user.user_id] += 0.001

        # special case: no resources requested at all
        if sum(fixed_allocation_vector) == 0:
            return np.zeros(self.config.num_users)

        # normalize sum to 1
        fixed_allocation_vector = fixed_allocation_vector / np.sum(fixed_allocation_vector)

        # work with resource blocks instead of proportional allocation from here on
        fixed_allocation_vector = fixed_allocation_vector * self.config.num_channels

        # limit allocation to min(allocated, requested)
        for user in self.ddpg_simulation.users.values():
            fixed_allocation_vector[user.user_id] = min(
                fixed_allocation_vector[user.user_id], user.units_requested)

        floored_allocation = np.floor(fixed_allocation_vector)
        leftovers = fixed_allocation_vector - floored_allocation
        fixed_allocation_vector = floored_allocation

        # from remaining resources, distribute at most one resource to every user that was robbed by flooring
        remaining_resources = self.config.num_channels - np.sum(fixed_allocation_vector)
        for loop_id in range(len(leftovers[leftovers > 0])):
            if remaining_resources == 0:
                return fixed_allocation_vector
            max_id = np.argmax(leftovers)
            fixed_allocation_vector[max_id] += 1
            remaining_resources -= 1
            leftovers[max_id] = 0

        return fixed_allocation_vector

    def allocation_post_processing_alt_testing(self, unfixed_allocation_vector):
        fixed_allocation_vector = unfixed_allocation_vector.copy()
        # zero out allocation with no jobs queued
        for user in self.ddpg_simulation.users.values():
            if user.units_requested == 0:
                fixed_allocation_vector[user.user_id] = 0
            else:
                if fixed_allocation_vector[user.user_id] == 0:
                    fixed_allocation_vector[user.user_id] += 0.001

        # special case: no resources requested at all
        if sum(fixed_allocation_vector) == 0:
            return np.zeros(self.config.num_users)

        # normalize sum to 1
        fixed_allocation_vector = fixed_allocation_vector / np.sum(fixed_allocation_vector)

        # work with resource blocks instead of proportional allocation from here on
        fixed_allocation_vector = fixed_allocation_vector * self.config.num_channels

        # limit allocation to min(allocated, requested)
        for user in self.ddpg_simulation.users.values():
            fixed_allocation_vector[user.user_id] = min(
                fixed_allocation_vector[user.user_id], user.units_requested)

        floored_allocation = np.floor(fixed_allocation_vector)
        leftovers = fixed_allocation_vector - floored_allocation
        fixed_allocation_vector = floored_allocation

        # from remaining resources, distribute at most one resource to every user that was robbed by flooring
        remaining_resources = self.config.num_channels - np.sum(fixed_allocation_vector)
        for loop_id in range(len(leftovers[leftovers > 0])):
            if remaining_resources == 0:
                return fixed_allocation_vector
            max_id = np.argmax(leftovers)
            fixed_allocation_vector[max_id] += 1
            remaining_resources -= 1
            leftovers[max_id] = 0

        # Allocate remaining resources, for use in testing
        if remaining_resources > 0:
            remaining_requests = np.zeros(self.config.num_users)
            for user in self.ddpg_simulation.users.values():
                remaining_requests[user.user_id] = user.units_requested - fixed_allocation_vector[user.user_id]
            while (sum(remaining_requests) > 0) and (remaining_resources > 0):
                max_request_id = np.argmax(remaining_requests)
                allocation = min(remaining_requests[max_request_id], remaining_resources)
                fixed_allocation_vector[max_request_id] += allocation
                remaining_requests[max_request_id] -= allocation
                remaining_resources -= allocation

        # distribute remaining resources by channel quality
        # remaining_resources = self.config.num_channels - np.sum(fixed_allocation_vector)
        # remaining_requests = np.zeros(self.config.num_users)
        # for user in self.ddpg_simulation.users.values():
        #     remaining_requests[user.user_id] = user.units_requested - fixed_allocation_vector[user.user_id]
        # channel_qualities = np.zeros(self.config.num_users)
        # for user in self.ddpg_simulation.users.values():
        #     channel_qualities[user.user_id] = user.channel_quality
        # while (remaining_resources > 0) and (np.sum(remaining_requests) > 0):
        #     max_quality_id = np.argmax(channel_qualities)
        #     if remaining_requests[max_quality_id] > 0:
        #         allocation = min(remaining_requests[max_quality_id], remaining_resources)
        #         fixed_allocation_vector[max_quality_id] += allocation
        #         remaining_requests[max_quality_id] = 0
        #         remaining_resources -= allocation
        #     channel_qualities[max_quality_id] = 0

        return fixed_allocation_vector

    def checkpoint_criterion(self, episode_id, episode_rewards):
        check_length = episode_id > self.config.noise_decay_to_zero_threshold * self.config.num_episodes
        if not check_length:
            return False
        check_interval = episode_id > self.last_checkpoint_episode + 10
        if not check_interval:
            return False
        check_mean = np.mean(episode_rewards[max(0, episode_id-400):episode_id]) > 1.01 * self.last_checkpoint_reward
        if check_mean:
            self.last_checkpoint_episode = episode_id
            self.last_checkpoint_reward = np.mean(episode_rewards[(episode_id-10):episode_id])
            return True
        return False

    def train(self):
        # Trains the ddpg neural networks according to config

        # Work load self-check------------------------------------------------------------------------------------------
        mean_arrival_load = self.config.new_job_chance * (
                self.config.num_users_per_job['Normal'] * self.config.normal_job_size_max / 2 +
                self.config.num_users_per_job['High Datarate'] * self.config.high_datarate_job_size_max / 2 +
                self.config.num_users_per_job['Low Latency'] * self.config.low_latency_job_size_max / 2 +
                self.config.num_users_per_job['Emergency Vehicle'] * self.config.EV_job_size_max / 2)

        print('Mean arrival load per resource:', mean_arrival_load / self.config.num_channels)

        # Parameter Setup-----------------------------------------------------------------------------------------------
        noise_multiplier = self.config.initial_noise_multiplier

        # logging
        mean_episode_errors = np.zeros(self.config.num_episodes)
        ddpg_episode_rewards = np.zeros(self.config.num_episodes)

        for episode_id in range(self.config.num_episodes):
            step_errors = np.zeros(self.config.steps_per_episode)
            relative_step_errors = np.zeros(self.config.steps_per_episode)
            for step_id in range(self.config.steps_per_episode):
                # Allocate resources and step simulation----------------------------------------------------------------
                state_old = self.gather_state()
                actor_allocation = self.actor_critic.get_action(state_old).numpy().squeeze()
                noisy_actor_allocation = self.add_noise(actor_allocation, noise_multiplier)
                noisy_actor_allocation = np.array([min(max(val, 0), 1) for val in noisy_actor_allocation])  # clip
                if sum(noisy_actor_allocation) > 0:
                    noisy_actor_allocation = noisy_actor_allocation / sum(noisy_actor_allocation)  # re-normalize
                processed_allocation = self.allocation_post_processing_alt(noisy_actor_allocation).astype(int)

                self.ddpg_simulation.step(processed_allocation)
                self.ddpg_simulation.generate_jobs(chance=self.config.new_job_chance)  # queue new jobs

                # Add experience into buffer, log and train-------------------------------------------------------------
                # if sum(processed_allocation) == self.config.num_channels:
                following_state = self.gather_state()
                self.actor_critic.add_experience(state=state_old,
                                                 action=noisy_actor_allocation,
                                                 reward=self.ddpg_simulation.rewards[-1],
                                                 following_state=following_state)

                # following_action = tf.squeeze(self.actor_critic.actor_target(following_state[np.newaxis]))
                # following_returns_estimate = self.actor_critic.get_q(following_state, following_action).numpy().squeeze().squeeze()
                following_returns_estimate = 0
                returns_estimate = self.actor_critic.get_q(state_old, noisy_actor_allocation).numpy().squeeze().squeeze()

                estimation_error = returns_estimate - \
                                   (self.ddpg_simulation.rewards[-1] + self.config.gamma * following_returns_estimate)
                step_errors[step_id] = np.power(estimation_error, 2)

                self.actor_critic.train()

            # logging
            mean_episode_errors[episode_id] = np.mean(step_errors[step_errors != 0])
            ddpg_episode_rewards[episode_id] = np.sum(self.ddpg_simulation.rewards)

            # Checkpointing---------------------------------------------------------------------------------------------
            if self.checkpoint_criterion(episode_id=episode_id, episode_rewards=ddpg_episode_rewards):
                self.actor_critic.actor_primary.predict(
                    np.random.rand(len(state_old))[np.newaxis])
                self.actor_critic.actor_primary.save(
                    join(self.config.model_path, 'actor', 'cp'), save_format='tf')

            # Reset simulation for the next episode---------------------------------------------------------------------
            self.ddpg_simulation.reset()

            # Anneal Parameters-----------------------------------------------------------------------------------------
            # Increase experience weight effect on sampling
            self.actor_critic.prioritization_factors[1] = min(
                1, self.actor_critic.prioritization_factors[1] + self.config.prioritization_factor_gain)
            noise_multiplier = max(0, noise_multiplier - self.config.noise_decay)

            # Print gradient magnitude, analytics-----------------------------------------------------------------------
            if (episode_id + 1) % 5 == 0:
                # if len(self.dqn.gradient_magnitude) > 0:  # Account for episodes without training
                #     print('Recent Max gradient: ' + str(np.max(self.dqn.gradient_magnitude)) +
                #           ', Recent mean gradient: ' + str(np.round(np.mean(self.dqn.gradient_magnitude), 1))
                #           )
                self.actor_critic.gradient_magnitude = np.array([], dtype=int)  # Reset log

            # Progress print--------------------------------------------------------------------------------------------
            if episode_id % max(int(self.config.num_episodes / 100), 1) == 0:
                completion = np.round((episode_id + 1) / self.config.num_episodes * 100, 1)
                print('\rProgress:', completion, '%', end='')

        print('\r ..Done', flush=True)

        # Logging-------------------------------------------------------------------------------------------------------
        with gzip.open(join(self.config.log_path, 'training_rewardestimationcritic.gstor'), 'wb') as file:
            pickle.dump([mean_episode_errors], file)
        with gzip.open(join(self.config.log_path, 'training_rewardsachieved.gstor'), 'wb') as file:
            pickle.dump(ddpg_episode_rewards, file)

        # Plotting------------------------------------------------------------------------------------------------------
        plt.figure()
        plt.title('Critic mean square cost estimation error')
        plt.xlabel('Episode')
        plt.ylabel('Squared mean estimation error')
        mean_episode_errors_smooth = np.zeros(self.config.num_episodes)
        mean_episode_errors_smooth[0] = mean_episode_errors[0]
        for episode_id in range(1, self.config.num_episodes):
            mean_episode_errors_smooth[episode_id] = np.mean(mean_episode_errors[max(0, episode_id - 500):episode_id])
        plt.plot(mean_episode_errors_smooth)
        plt.grid(alpha=0.25)
        plt.tight_layout()

        plt.figure()
        plt.title('DDPG Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        mean_episode_rewards_smooth = np.zeros(self.config.num_episodes)
        mean_episode_rewards_smooth[0] = ddpg_episode_rewards[0]
        for episode_id in range(1, self.config.num_episodes):
            mean_episode_rewards_smooth[episode_id] = np.mean(
                ddpg_episode_rewards[max(0, episode_id - 100):episode_id])
        plt.plot(mean_episode_rewards_smooth)
        plt.grid(alpha=.25)
        plt.tight_layout()

        # Save actor_critic---------------------------------------------------------------------------------------------
        # Load with tf.keras.models.load_model('path')
        # required bc custom model save bug:
        self.actor_critic.q_primary.predict(
            np.random.rand(len(processed_allocation) + len(state_old))[np.newaxis])
        self.actor_critic.q_primary.save(join(self.config.model_path, 'q_primary'), save_format='tf')

        self.actor_critic.actor_primary.predict(
            np.random.rand(len(state_old))[np.newaxis])
        self.actor_critic.actor_primary.save(join(self.config.model_path, 'actor'), save_format='tf')

        # Save associated config----------------------------------------------------------------------------------------
        copy2('policygradient_config.py', self.config.model_path)

    def test(self):
        # Tests a trained ddpg against basic algorithms

        # Work load self-check------------------------------------------------------------------------------------------
        mean_arrival_load = self.config.new_job_chance * (
                self.config.num_users_per_job['Normal'] * self.config.normal_job_size_max / 2 +
                self.config.num_users_per_job['High Datarate'] * self.config.high_datarate_job_size_max / 2 +
                self.config.num_users_per_job['Low Latency'] * self.config.low_latency_job_size_max / 2 +
                self.config.num_users_per_job['Emergency Vehicle'] * self.config.EV_job_size_max / 2)
        print('Mean arrival load per resource:', mean_arrival_load / self.config.num_channels)

        # Load trained models-------------------------------------------------------------------------------------------
        actor = tf.keras.models.load_model(join(self.config.model_path, 'actor'))

        # Set up comparison setups--------------------------------------------------------------------------------------
        maximum_throughput_simulation = Simulation(config=self.config)
        max_min_fair_simulation = Simulation(config=self.config)
        delay_sensitive_simulation = Simulation(config=self.config)
        random_simulation = Simulation(config=self.config)

        # logging
        actor_resources_used = np.zeros((self.config.num_episodes, self.config.steps_per_episode))
        actor_critic_mean_throughput_per_episode = np.zeros(self.config.num_episodes)
        actor_critic_datarate_satisfaction_per_episode = np.zeros(self.config.num_episodes)
        actor_critic_latency_violations_per_episode = np.zeros(self.config.num_episodes)
        actor_critic_rewards_per_episode = np.zeros(self.config.num_episodes)
        actor_critic_ev_latency_violations_per_episode = np.zeros(self.config.num_episodes)

        max_throughput_mean_throughput_per_episode = np.zeros(self.config.num_episodes)
        max_throughput_datarate_satisfaction_per_episode = np.zeros(self.config.num_episodes)
        max_throughput_latency_violations_per_episode = np.zeros(self.config.num_episodes)
        max_throughput_rewards_per_episode = np.zeros(self.config.num_episodes)
        max_throughput_ev_latency_violations_per_episode = np.zeros(self.config.num_episodes)

        max_min_fair_mean_throughput_per_episode = np.zeros(self.config.num_episodes)
        max_min_fair_datarate_satisfaction_per_episode = np.zeros(self.config.num_episodes)
        max_min_fair_latency_violations_per_episode = np.zeros(self.config.num_episodes)
        max_min_fair_rewards_per_episode = np.zeros(self.config.num_episodes)
        max_min_fair_ev_latency_violations_per_episode = np.zeros(self.config.num_episodes)

        delay_sensitive_mean_throughput_per_episode = np.zeros(self.config.num_episodes)
        delay_sensitive_datarate_satisfaction_per_episode = np.zeros(self.config.num_episodes)
        delay_sensitive_latency_violations_per_episode = np.zeros(self.config.num_episodes)
        delay_sensitive_rewards_per_episode = np.zeros(self.config.num_episodes)
        delay_sensitive_ev_latency_violations_per_episode = np.zeros(self.config.num_episodes)

        random_sim_mean_throughput_per_episode = np.zeros(self.config.num_episodes)
        random_sim_datarate_sastisfaction_per_episode = np.zeros(self.config.num_episodes)
        random_sim_latency_violations_per_episode = np.zeros(self.config.num_episodes)
        random_sim_rewards_per_episode = np.zeros(self.config.num_episodes)
        random_sim_ev_latency_violations_per_episode = np.zeros(self.config.num_episodes)

        for episode_id in range(self.config.num_episodes):
            # Allocate resources and step simulation--------------------------------------------------------------------
            # ddpg
            for step_id in range(self.config.steps_per_episode):
                state_old = self.gather_state()

                # actor_allocation = self.actor_critic.get_action(state_old).numpy().squeeze()
                actor_allocation = actor(state_old[np.newaxis]).numpy().squeeze()
                processed_allocation = self.allocation_post_processing_alt_testing(actor_allocation).astype(int)
                # logging
                resources_requested = 0
                for user in self.ddpg_simulation.users.values():
                    resources_requested += user.units_requested
                if resources_requested >= self.config.num_channels:
                    actor_resources_used[episode_id, step_id] = np.sum(processed_allocation)
                else:
                    actor_resources_used[episode_id, step_id] = -1

                self.ddpg_simulation.step(processed_allocation)
                self.ddpg_simulation.generate_jobs(chance=self.config.new_job_chance)

            # max_throughput
            for step_id in range(self.config.steps_per_episode):
                max_throughput_allocation = maximum_throughput_scheduler(
                    users=list(maximum_throughput_simulation.users.values()),
                    resources_available=self.config.num_channels)
                maximum_throughput_simulation.step(max_throughput_allocation)
                maximum_throughput_simulation.generate_jobs(chance=self.config.new_job_chance)

            # max_min_fair
            for step_id in range(self.config.steps_per_episode):
                max_min_fair_allocation = max_min_fair_scheduler(
                    users=list(max_min_fair_simulation.users.values()),
                    resources_available=self.config.num_channels)
                max_min_fair_simulation.step(max_min_fair_allocation)
                max_min_fair_simulation.generate_jobs(chance=self.config.new_job_chance)

            # delay sensitive
            for step_id in range(self.config.steps_per_episode):
                delay_sensitive_allocation = delay_sensitive_scheduler(
                    users=list(delay_sensitive_simulation.users.values()),
                    resources_available=self.config.num_channels)
                delay_sensitive_simulation.step(delay_sensitive_allocation)
                delay_sensitive_simulation.generate_jobs(chance=self.config.new_job_chance)

            # random
            for step_id in range(self.config.steps_per_episode):
                random_allocation = random_scheduler(
                    users=list(random_simulation.users.values()),
                    resources_available=self.config.num_channels)
                random_simulation.step(random_allocation)
                random_simulation.generate_jobs(chance=self.config.new_job_chance)

            # Collect statistics----------------------------------------------------------------------------------------
            actor_critic_mean_throughput_per_episode[episode_id] = np.mean(
                self.ddpg_simulation.sum_capacity)
            actor_critic_datarate_satisfaction_per_episode[episode_id] = np.sum(
                self.ddpg_simulation.datarate_satisfaction)
            actor_critic_latency_violations_per_episode[episode_id] = np.sum(
                self.ddpg_simulation.jobs_lost)
            actor_critic_rewards_per_episode[episode_id] = np.sum(
                self.ddpg_simulation.rewards)
            actor_critic_ev_latency_violations_per_episode[episode_id] = np.sum(
                self.ddpg_simulation.jobs_lost_EV_only)

            max_throughput_mean_throughput_per_episode[episode_id] = np.mean(
                maximum_throughput_simulation.sum_capacity)
            max_throughput_datarate_satisfaction_per_episode[episode_id] = np.sum(
                maximum_throughput_simulation.datarate_satisfaction)
            max_throughput_latency_violations_per_episode[episode_id] = np.sum(
                maximum_throughput_simulation.jobs_lost)
            max_throughput_rewards_per_episode[episode_id] = np.sum(
                maximum_throughput_simulation.rewards)
            max_throughput_ev_latency_violations_per_episode[episode_id] = np.sum(
                maximum_throughput_simulation.jobs_lost_EV_only)

            max_min_fair_mean_throughput_per_episode[episode_id] = np.mean(
                max_min_fair_simulation.sum_capacity)
            max_min_fair_datarate_satisfaction_per_episode[episode_id] = np.sum(
                max_min_fair_simulation.datarate_satisfaction)
            max_min_fair_latency_violations_per_episode[episode_id] = np.sum(
                max_min_fair_simulation.jobs_lost)
            max_min_fair_rewards_per_episode[episode_id] = np.sum(
                max_min_fair_simulation.rewards)
            max_min_fair_ev_latency_violations_per_episode[episode_id] = np.sum(
                max_min_fair_simulation.jobs_lost_EV_only)

            delay_sensitive_mean_throughput_per_episode[episode_id] = np.mean(
                delay_sensitive_simulation.sum_capacity)
            delay_sensitive_datarate_satisfaction_per_episode[episode_id] = np.sum(
                delay_sensitive_simulation.datarate_satisfaction)
            delay_sensitive_latency_violations_per_episode[episode_id] = np.sum(
                delay_sensitive_simulation.jobs_lost)
            delay_sensitive_rewards_per_episode[episode_id] = np.sum(
                delay_sensitive_simulation.rewards)
            delay_sensitive_ev_latency_violations_per_episode[episode_id] = np.sum(
                delay_sensitive_simulation.jobs_lost_EV_only)

            random_sim_mean_throughput_per_episode[episode_id] = np.mean(
                random_simulation.sum_capacity)
            random_sim_datarate_sastisfaction_per_episode[episode_id] = np.sum(
                random_simulation.datarate_satisfaction)
            random_sim_latency_violations_per_episode[episode_id] = np.sum(
                random_simulation.jobs_lost)
            random_sim_rewards_per_episode[episode_id] = np.sum(
                random_simulation.rewards)
            random_sim_ev_latency_violations_per_episode[episode_id] = np.sum(
                random_simulation.jobs_lost_EV_only)

            # Reset simulation for the next episode---------------------------------------------------------------------
            self.ddpg_simulation.reset()
            maximum_throughput_simulation.reset()
            max_min_fair_simulation.reset()
            delay_sensitive_simulation.reset()
            random_simulation.reset()

            # Progress print--------------------------------------------------------------------------------------------
            if episode_id % max(int(self.config.num_episodes / 100), 1) == 0:
                completion = np.round((episode_id + 1) / self.config.num_episodes * 100, 1)
                print('\rProgress:', completion, '%', end='')

        print('\r ..Done', flush=True)
        print('AC', np.sum(actor_critic_ev_latency_violations_per_episode) / self.config.num_episodes, '\n' +
              'MT', np.sum(max_throughput_ev_latency_violations_per_episode) / self.config.num_episodes, '\n' +
              'MM', np.sum(max_min_fair_ev_latency_violations_per_episode) / self.config.num_episodes, '\n' +
              'DS', np.sum(delay_sensitive_ev_latency_violations_per_episode) / self.config.num_episodes, '\n' +
              'RD', np.sum(random_sim_ev_latency_violations_per_episode) / self.config.num_episodes, '\n')

        # Logging-------------------------------------------------------------------------------------------------------
        with gzip.open(join(self.config.log_path, 'testing_throughputs.gstor'), 'wb') as file:
            pickle.dump([max_throughput_mean_throughput_per_episode, max_min_fair_mean_throughput_per_episode,
                         delay_sensitive_mean_throughput_per_episode, actor_critic_mean_throughput_per_episode,
                         random_sim_mean_throughput_per_episode], file)
        with gzip.open(join(self.config.log_path, 'testing_latencyviolations.gstor'), 'wb') as file:
            pickle.dump([max_throughput_latency_violations_per_episode, max_min_fair_latency_violations_per_episode,
                         delay_sensitive_latency_violations_per_episode, actor_critic_latency_violations_per_episode,
                         random_sim_latency_violations_per_episode], file)
        with gzip.open(join(self.config.log_path, 'testing_dataratesat.gstor'), 'wb') as file:
            pickle.dump([max_throughput_datarate_satisfaction_per_episode, max_min_fair_datarate_satisfaction_per_episode,
                         delay_sensitive_datarate_satisfaction_per_episode, actor_critic_datarate_satisfaction_per_episode,
                         random_sim_datarate_sastisfaction_per_episode], file)
        with gzip.open(join(self.config.log_path, 'testing_rewards.gstor'), 'wb') as file:
            pickle.dump([max_throughput_rewards_per_episode, max_min_fair_rewards_per_episode,
                         delay_sensitive_rewards_per_episode, actor_critic_rewards_per_episode,
                         random_sim_rewards_per_episode], file)
        with gzip.open(join(self.config.log_path, 'testing_ev_latencyviolations.gstor'), 'wb') as file:
            pickle.dump([max_throughput_ev_latency_violations_per_episode, max_min_fair_ev_latency_violations_per_episode,
                         delay_sensitive_ev_latency_violations_per_episode, actor_critic_ev_latency_violations_per_episode,
                         random_sim_ev_latency_violations_per_episode], file)
        with gzip.open(join(self.config.log_path, 'testing_resources_used.gstor'), 'wb') as file:
            pickle.dump(actor_resources_used, file)

        # Plotting------------------------------------------------------------------------------------------------------
        fig_throughput_comparison = plt.figure()
        plt.title('Algorithm Throughput comparison')
        plt.bar(0, np.mean(max_throughput_mean_throughput_per_episode), color=self.config.ccolor0, )
        plt.bar(1, np.mean(max_min_fair_mean_throughput_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_mean_throughput_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(actor_critic_mean_throughput_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(random_sim_mean_throughput_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DDPG', 'Random'])
        plt.ylabel('Normalized Throughput')
        plt.tight_layout()

        fig_timeouts_comparison = plt.figure()
        plt.title('Latency constraint violations comparison')
        plt.bar(0, np.mean(max_throughput_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(1, np.mean(max_min_fair_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(actor_critic_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(random_sim_latency_violations_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DDPG', 'Random'])
        plt.ylabel('Mean violations per episode')
        plt.tight_layout()

        fig_datarate_satisfaction = plt.figure()
        plt.title('Sum data rate satisfaction comparison')
        plt.bar(0, np.mean(max_throughput_datarate_satisfaction_per_episode), color=self.config.ccolor0)
        plt.bar(1, np.mean(max_min_fair_datarate_satisfaction_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_datarate_satisfaction_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(actor_critic_datarate_satisfaction_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(random_sim_datarate_sastisfaction_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DDPG', 'Random'])
        plt.ylabel('Mean sum data rate satisfaction per episode')
        plt.tight_layout()

        fig_reward_comparison = plt.figure()
        plt.title('Combined metric comparison')
        plt.bar(0, np.mean(max_throughput_rewards_per_episode), color=self.config.ccolor0)
        plt.bar(1, np.mean(max_min_fair_rewards_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_rewards_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(actor_critic_rewards_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(random_sim_rewards_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DDPG', 'Random'])
        plt.ylabel('Mean combined metric per episode')
        plt.tight_layout()

        fig_timeouts_ev_comparison = plt.figure()
        plt.title('Timeouts for EV only')
        plt.bar(0, np.mean(max_throughput_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(1, np.mean(max_min_fair_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(actor_critic_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(random_sim_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DDPG', 'Random'])
        plt.ylabel('Mean EV timeouts per episode')
        plt.tight_layout()

        fig_resources_used_count = plt.figure()
        res_used, count = np.unique(actor_resources_used, return_counts=True)
        plt.bar(res_used, count)
        plt.grid(alpha=.25)
        plt.xticks(range(int(res_used[0]), int(res_used[-1])+1))
        print(dict(zip(res_used, count)))
