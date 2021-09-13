import numpy as np
import matplotlib.pyplot as plt

from scheduling.imports.simulation import Simulation
from scheduling.scheduling_config import Config
from scheduling.imports.scheduling import maximum_throughput_scheduler
from scheduling.imports.scheduling import max_min_fair_scheduler
from scheduling.imports.scheduling import delay_sensitive_scheduler
from scheduling.imports.scheduling import ev_only_scheduler

config = Config()

# Work load self-check
mean_arrival_load = config.new_job_chance * (config.num_users_per_job['Normal'] * config.normal_job_size_max / 2 +
                                             config.num_users_per_job['High Datarate'] * config.high_datarate_job_size_max / 2 +
                                             config.num_users_per_job['Low Latency'] * config.low_latency_job_size_max)
print('Mean arrival load per resource:', mean_arrival_load / config.num_channels)

# Simulations-----------------------------------------------------------------------------------------------------------
max_throughput_simulation = Simulation(config=config)
max_min_fair_simulation = Simulation(config=config)
delay_sensitive_simulation = Simulation(config=config)

# Statistics------------------------------------------------------------------------------------------------------------
max_throughput_mean_throughput_per_episode = np.zeros(config.num_episodes)
max_throughput_datarate_satisfaction_per_episode = np.zeros(config.num_episodes)
max_throughput_latency_violations_per_episode = np.zeros(config.num_episodes)
max_throughput_latency_violations_ev_only_per_episode = np.zeros(config.num_episodes)
max_throughput_rewards_per_episode = np.zeros(config.num_episodes)

max_min_fair_mean_throughput_per_episode = np.zeros(config.num_episodes)
max_min_fair_datarate_satisfaction_per_episode = np.zeros(config.num_episodes)
max_min_fair_latency_violations_per_episode = np.zeros(config.num_episodes)
max_min_fair_latency_violations_ev_only_per_episode = np.zeros(config.num_episodes)
max_min_fair_rewards_per_episode = np.zeros(config.num_episodes)

delay_sensitive_mean_throughput_per_episode = np.zeros(config.num_episodes)
delay_sensitive_datarate_satisfaction_per_episode = np.zeros(config.num_episodes)
delay_sensitive_latency_violations_per_episode = np.zeros(config.num_episodes)
delay_sensitive_latency_violations_ev_only_per_episode = np.zeros(config.num_episodes)
delay_sensitive_rewards_per_episode = np.zeros(config.num_episodes)

for episode_id in range(config.num_episodes):
    # Simulations-------------------------------------------------------------------------------------------------------
    # MaxThroughput Sim
    for _ in range(config.steps_per_episode):
        # max_throughput_simulation.get_len_job_list()
        allocation_vector = maximum_throughput_scheduler(users=list(max_throughput_simulation.users.values()),
                                                         resources_available=config.num_channels)
        max_throughput_simulation.step(allocation_vector)
        max_throughput_simulation.generate_jobs(chance=config.new_job_chance)

    # MaxMinFair Sim
    for _ in range(config.steps_per_episode):
        allocation_vector = max_min_fair_scheduler(users=list(max_min_fair_simulation.users.values()),
                                                   resources_available=config.num_channels)
        max_min_fair_simulation.step(allocation_vector)
        max_min_fair_simulation.generate_jobs(chance=config.new_job_chance)

    # DelaySensitive Sim
    for _ in range(config.steps_per_episode):
        allocation_vector = delay_sensitive_scheduler(users=list(delay_sensitive_simulation.users.values()),
                                                      resources_available=config.num_channels)

        delay_sensitive_simulation.step(allocation_vector)
        delay_sensitive_simulation.generate_jobs(chance=config.new_job_chance)

    # Datarate satisfaction per user plot for one episode
    # if episode_id == 0:
    #     datarate_satisfaction_figure = plt.figure()
    #     plt.ylim([0, 1])
    #     for user in max_throughput_simulation.users.values():
    #         plt.bar(user.user_id, user.datarate_satisfaction)
    #     datarate_satisfaction_figure2 = plt.figure()
    #     plt.ylim([0, 1])
    #     for user in max_min_fair_simulation.users.values():
    #         plt.bar(user.user_id, user.datarate_satisfaction)
    #     datarate_satisfaction_figure3 = plt.figure()
    #     plt.ylim([0, 1])
    #     for user in delay_sensitive_simulation.users.values():
    #         plt.bar(user.user_id, user.datarate_satisfaction)

    # Collect statistics------------------------------------------------------------------------------------------------
    max_throughput_mean_throughput_per_episode[episode_id] = np.mean(max_throughput_simulation.sum_capacity)
    max_throughput_datarate_satisfaction_per_episode[episode_id] = np.sum(max_throughput_simulation.datarate_satisfaction)
    max_throughput_latency_violations_per_episode[episode_id] = np.sum(max_throughput_simulation.jobs_lost)
    max_throughput_latency_violations_ev_only_per_episode[episode_id] = np.sum(max_throughput_simulation.jobs_lost_EV_only)
    max_throughput_rewards_per_episode[episode_id] = np.mean(max_throughput_simulation.rewards)

    max_min_fair_mean_throughput_per_episode[episode_id] = np.mean(max_min_fair_simulation.sum_capacity)
    max_min_fair_datarate_satisfaction_per_episode[episode_id] = np.sum(max_min_fair_simulation.datarate_satisfaction)
    max_min_fair_latency_violations_per_episode[episode_id] = np.sum(max_min_fair_simulation.jobs_lost)
    max_min_fair_latency_violations_ev_only_per_episode[episode_id] = np.sum(max_min_fair_simulation.jobs_lost_EV_only)
    max_min_fair_rewards_per_episode[episode_id] = np.mean(max_min_fair_simulation.rewards)

    delay_sensitive_mean_throughput_per_episode[episode_id] = np.mean(delay_sensitive_simulation.sum_capacity)
    delay_sensitive_datarate_satisfaction_per_episode[episode_id] = np.sum(delay_sensitive_simulation.datarate_satisfaction)
    delay_sensitive_latency_violations_per_episode[episode_id] = np.sum(delay_sensitive_simulation.jobs_lost)
    delay_sensitive_latency_violations_ev_only_per_episode[episode_id] = np.sum(delay_sensitive_simulation.jobs_lost_EV_only)
    delay_sensitive_rewards_per_episode[episode_id] = np.mean(delay_sensitive_simulation.rewards)

    # Reset simulation for the next episode-----------------------------------------------------------------------------
    max_throughput_simulation.reset()
    max_min_fair_simulation.reset()
    delay_sensitive_simulation.reset()

    # Progress print----------------------------------------------------------------------------------------------------
    if episode_id % int(config.num_episodes / 100) == 0:
        completion = np.round((episode_id + 1) / config.num_episodes * 100, 1)
        print('\rProgress:', completion, '%', end='')

print('\r ..Done', flush=True)
print('MaxThroughput mean throughput', np.mean(max_throughput_mean_throughput_per_episode))
print('MaxMinFair mean throughput', np.mean(max_min_fair_mean_throughput_per_episode))
print('DelaySensitive mean throughput', np.mean(delay_sensitive_mean_throughput_per_episode))

print('MaxThroughput mean datarate', np.mean(max_throughput_datarate_satisfaction_per_episode))
print('MaxMinFair mean datarate', np.mean(max_min_fair_datarate_satisfaction_per_episode))
print('DelaySensitive mean datarate', np.mean(delay_sensitive_datarate_satisfaction_per_episode))

# Plotting--------------------------------------------------------------------------------------------------------------

fig_throughput_comparison = plt.figure()
plt.title('Algorithm Throughput Comparison')
norm_factor = max(np.mean(max_throughput_mean_throughput_per_episode),
                  np.mean(max_min_fair_mean_throughput_per_episode),
                  np.mean(delay_sensitive_mean_throughput_per_episode))
plt.bar(0, np.mean(max_throughput_mean_throughput_per_episode) / 1, color=config.ccolor0)
plt.bar(1, np.mean(max_min_fair_mean_throughput_per_episode) / 1, color=config.ccolor0)
plt.bar(2, np.mean(delay_sensitive_mean_throughput_per_episode) / 1, color=config.ccolor0)
plt.grid(alpha=.25, axis='y')
plt.xticks([0, 1, 2], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive'])
plt.ylabel('Normalized Throughput')
plt.tight_layout()

fig_timeouts_comparison = plt.figure()
plt.title('Latency constraint violations comparison')
plt.bar(0, np.mean(max_throughput_latency_violations_per_episode), color=config.ccolor0)
plt.bar(1, np.mean(max_min_fair_latency_violations_per_episode), color=config.ccolor0)
plt.bar(2, np.mean(delay_sensitive_latency_violations_per_episode), color=config.ccolor0)
plt.grid(alpha=.25, axis='y')
plt.xticks([0, 1, 2], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive'])
plt.ylabel('Mean violations per episode')
plt.tight_layout()

fig_datarate_satisfaction = plt.figure()
plt.title('Sum data rate satisfaction')
plt.bar(0, np.mean(max_throughput_datarate_satisfaction_per_episode), color=config.ccolor0)
plt.bar(1, np.mean(max_min_fair_datarate_satisfaction_per_episode), color=config.ccolor0)
plt.bar(2, np.mean(delay_sensitive_datarate_satisfaction_per_episode), color=config.ccolor0)
plt.grid(alpha=.25, axis='y')
plt.xticks([0, 1, 2], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive'])
plt.ylabel('Mean sum data rate satisfaction per episode')
plt.tight_layout()

fig_ev_timeouts, ax_ev_timeouts = plt.subplots()
ax_ev_timeouts.set_title('EV timeouts')
for sched_id, results in enumerate(
        [
            max_throughput_latency_violations_ev_only_per_episode,
            max_min_fair_latency_violations_ev_only_per_episode,
            delay_sensitive_latency_violations_ev_only_per_episode
        ]
):
    ax_ev_timeouts.bar(sched_id, np.mean(results), yerr=np.var(results))
ax_ev_timeouts.grid(alpha=.25, axis='y')
ax_ev_timeouts.set_xticks([0, 1, 2])
ax_ev_timeouts.set_xticklabels(['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive'])
ax_ev_timeouts.set_ylabel('Mean ev timeouts per episode')
fig_ev_timeouts.tight_layout()

fig_rewards, ax_rewards = plt.subplots()
ax_rewards.set_title('Rewards')
ax_rewards.bar(0, np.mean(max_throughput_rewards_per_episode), yerr=np.var(max_throughput_rewards_per_episode))
ax_rewards.bar(1, np.mean(max_min_fair_rewards_per_episode), yerr=np.var(max_min_fair_rewards_per_episode))
ax_rewards.bar(2, np.mean(delay_sensitive_rewards_per_episode), yerr=np.var(delay_sensitive_rewards_per_episode))
ax_rewards.grid(alpha=.25, axis='y')
ax_rewards.set_xticks([0, 1, 2])
ax_rewards.set_xticklabels(['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive'])
ax_rewards.set_ylabel('Mean rewards per episode')
fig_rewards.tight_layout()

plt.show()
