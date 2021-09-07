import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gzip, pickle
from os.path import join
from shutil import copy2

from scheduling.imports.simulation import Simulation
from scheduling.scheduling_config import Config
from scheduling.imports.dqn import DQN
from scheduling.imports.scheduling import maximum_throughput_scheduler
from scheduling.imports.scheduling import max_min_fair_scheduler
from scheduling.imports.scheduling import delay_sensitive_scheduler
from scheduling.imports.scheduling import ev_only_scheduler


class Runner:
    def __init__(self):
        self.config = Config()

        self.epsilon: float = 0

        self.dqn_simulation = Simulation(config=self.config)
        self.dqn = DQN(num_algorithms=4,
                       size_state=2*self.config.num_users+2,
                       num_hidden=self.config.num_hidden,
                       rainbow=self.config.rainbow,
                       num_dueling=self.config.num_dueling,
                       learning_rate=self.config.learning_rate,
                       batch_size=self.config.batch_size,
                       min_experiences=self.config.min_experiences,
                       max_experiences=self.config.max_experiences,
                       min_priority=self.config.min_priority,
                       prioritization_factors=self.config.prioritization_factors,
                       gamma=self.config.gamma,
                       loss_function=self.config.loss_function)

        # Statistics
        self.dqn_episode_rewards = np.zeros(self.config.num_episodes)

    def gather_state(self):
        state = np.array([])
        state = np.append(state, self.dqn_simulation.get_len_job_list())  # Queue length

        for user in self.dqn_simulation.users.values():  # Gather channel qualities
            state = np.append(state, user.channel_quality)

        for user in self.dqn_simulation.users.values():  # Gather mean time-to-timeout, min time-to-timeout
            user_job_timeouts = np.array([])
            for job in user.jobs:
                user_job_timeouts = np.append(user_job_timeouts, user.latency_max - job.delay)

            if user_job_timeouts.size > 0:  # if not empty
                state = np.append(state, np.mean(user_job_timeouts))
                state = np.append(state, min(user_job_timeouts))
            else:
                state = np.append(state, [10, 10])

        for user in self.dqn_simulation.users.values():  # Gather datarate satisfaction
            state = np.append(state, user.datarate_satisfaction)

        return state

    def step_simulations(self):
        dqn_action = self.dqn.get_action(state=self.gather_state(), epsilon=self.epsilon)

        if dqn_action == 0:
            resource_allocation = maximum_throughput_scheduler(users=list(self.dqn_simulation.users.values()),
                                                               resources_available=self.config.num_channels)
        elif dqn_action == 1:
            resource_allocation = max_min_fair_scheduler(users=list(self.dqn_simulation.users.values()),
                                                         resources_available=self.config.num_channels)
        elif dqn_action == 2:
            resource_allocation = delay_sensitive_scheduler(users=list(self.dqn_simulation.users.values()),
                                                            resources_available=self.config.num_channels)
        elif dqn_action == 3:
            resource_allocation = ev_only_scheduler(users=list(self.dqn_simulation.users.values()),
                                                    resources_available=self.config.num_channels)
        else:
            raise ValueError('Invalid dqn.get_action output value', dqn_action)

        self.dqn_simulation.step(allocation_vector=resource_allocation)
        self.dqn_simulation.generate_jobs(chance=self.config.new_job_chance)

        return dqn_action

    @staticmethod
    def should_save_cp(current_cp_reward, episode_id, best_checkpoint_reward, interval):
        cp_improvement = current_cp_reward / best_checkpoint_reward  # improvement in %
        if cp_improvement > 1.01:  # 1% better than the best cp
            # benchmark_improvement = current_cp_reward / self.random_total_rewards[
            #                                             max(0, episode_id - interval):(episode_id + 1)].mean()
            # if benchmark_improvement > 1.05:  # 5% better than the benchmark
            return True

        return False

    def train(self):
        # Work load self-check------------------------------------------------------------------------------------------
        mean_arrival_load = self.config.new_job_chance * (
                self.config.num_users_per_job['Normal'] * self.config.normal_job_size_max / 2 +
                self.config.num_users_per_job['High Datarate'] * self.config.high_datarate_job_size_max / 2 +
                self.config.num_users_per_job['Low Latency'] * self.config.low_latency_job_size_max / 2 +
                self.config.num_users_per_job['Emergency Vehicle'] * self.config.EV_job_size_max / 2)
        print('Mean arrival load per resource:', mean_arrival_load / self.config.num_channels)

        self.epsilon = self.config.epsilon

        checkpoint_counter: int = 0
        best_checkpoint_reward: float = 1e-20

        for episode_id in range(self.config.num_episodes):
            dqn_step_rewards = np.zeros(self.config.steps_per_episode)

            for step_id in range(self.config.steps_per_episode):
                # Save current state for experience buffer
                dqn_state_old = self.gather_state()

                # Step simulations and log actions taken for experience buffer
                dqn_actions_taken = self.step_simulations()

                # Log reward
                dqn_step_rewards[step_id] = self.dqn_simulation.rewards[-1]

                # Add Experience, train
                self.dqn.add_experience(state=dqn_state_old,
                                        action=dqn_actions_taken,
                                        reward=self.dqn_simulation.rewards[-1],
                                        following_state=self.gather_state())
                self.dqn.train()

                # Copy params to target net periodically
                if episode_id % self.config.copy_step == 0:
                    self.dqn.copy_weights()

            # Log rewards
            self.dqn_episode_rewards[episode_id] = np.sum(dqn_step_rewards)

            # Anneal Parameters
            # Decay random action probability
            self.epsilon = max(self.config.epsilon_min, self.epsilon - self.config.epsilon_decay)
            # Decay Learning Rate
            self.dqn.reduce_learning_rate(lr_decay=self.config.lr_decay, lr_min=self.config.lr_min)
            # Increase experience weight effect on sampling
            self.dqn.prioritization_factors[1] = min(
                1, self.dqn.prioritization_factors[1] + self.config.prioritization_factor_gain)

            # Reset simulation environments for the next episode
            self.dqn_simulation.reset()

            # Print gradient magnitude, analytics
            if (episode_id + 1) % 5 == 0:
                # if len(self.dqn.gradient_magnitude) > 0:  # Account for episodes without training
                #     print('Recent Max gradient: ' + str(np.max(self.dqn.gradient_magnitude)) +
                #           ', Recent mean gradient: ' + str(np.round(np.mean(self.dqn.gradient_magnitude), 1))
                #           )
                self.dqn.gradient_magnitude = np.array([], dtype=int)  # Reset log

            # Print rewards, analytics
            if (episode_id + 1) % 10 == 0:
                dqn_avg_rewards = self.dqn_episode_rewards[max(0, episode_id - 20):(episode_id + 1)].mean()

                print('Episode ' + str(episode_id + 1) +
                      ': Recent mean reward per episode: ' + str(np.round(dqn_avg_rewards, 1)) +
                      ', Current chance of random actions: ' + str(np.round(self.epsilon, 3))
                      )

            # Checkpoint-------------------------------------
            interval = 300
            dqn_avg_rewards_longterm = self.dqn_episode_rewards[max(0, episode_id - interval):(episode_id + 1)].mean()
            if self.should_save_cp(dqn_avg_rewards_longterm, episode_id, best_checkpoint_reward, interval=interval):
                self.dqn.q_primary.predict(np.random.rand(len(dqn_state_old))[np.newaxis])
                self.dqn.q_primary.save(join(self.config.model_path, 'q_primary', 'cp'), save_format='tf')

                checkpoint_counter += 1
                best_checkpoint_reward = dqn_avg_rewards_longterm

                print('Saved Checkpoint #' + str(checkpoint_counter - 1) +
                      ' at reward ' + str(round(best_checkpoint_reward, 1)))

        # Save q_primary---------------------------------
        # Load with tf.keras.models.load_model('path')
        # required bc custom model save bug:
        self.dqn.q_primary.predict(np.random.rand(len(dqn_state_old))[np.newaxis])
        self.dqn.q_primary.save(join(self.config.model_path, 'q_primary'), save_format='tf')

        # Save associated config----------------------------------------------------------------------------------------
        copy2('scheduling_config.py', self.config.model_path)

    def test(self):
        # Work load self-check------------------------------------------------------------------------------------------
        mean_arrival_load = self.config.new_job_chance * (
                self.config.num_users_per_job['Normal'] * self.config.normal_job_size_max / 2 +
                self.config.num_users_per_job['High Datarate'] * self.config.high_datarate_job_size_max / 2 +
                self.config.num_users_per_job['Low Latency'] * self.config.low_latency_job_size_max / 2 +
                self.config.num_users_per_job['Emergency Vehicle'] * self.config.EV_job_size_max / 2)

        print('Mean arrival load per resource:', mean_arrival_load / self.config.num_channels)

        self.dqn.q_primary = tf.keras.models.load_model(join(self.config.model_path, 'q_primary'))
        self.epsilon = 0

        # Simulations---------------------------------------------------------------------------------------------------
        max_throughput_simulation = Simulation(config=self.config)
        max_min_fair_simulation = Simulation(config=self.config)
        delay_sensitive_simulation = Simulation(config=self.config)
        ev_only_simulation = Simulation(config=self.config)

        # Statistics------------------------------------------------------------------------------------------------------------
        dqn_mean_throughput_per_episode = np.zeros(self.config.num_episodes)
        dqn_datarate_satisfaction_per_episode = np.zeros(self.config.num_episodes)
        dqn_latency_violations_per_episode = np.zeros(self.config.num_episodes)
        dqn_rewards_per_episode = np.zeros(self.config.num_episodes)
        dqn_ev_latency_violations_per_episode = np.zeros(self.config.num_episodes)

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

        ev_only_mean_throughput_per_episode = np.zeros(self.config.num_episodes)
        ev_only_datarate_satisfaction_per_episode = np.zeros(self.config.num_episodes)
        ev_only_latency_violations_per_episode = np.zeros(self.config.num_episodes)
        ev_only_rewards_per_episode = np.zeros(self.config.num_episodes)
        ev_only_ev_latency_violations_per_episode = np.zeros(self.config.num_episodes)

        algorithm_counters = np.array([0, 0, 0, 0])

        for episode_id in range(self.config.num_episodes):
            # Simulations-------------------------------------------------------------------------------------------------------
            # DQN Sim
            for _ in range(self.config.steps_per_episode):
                dqn_action = self.dqn.get_action(state=self.gather_state(), epsilon=self.epsilon)

                if dqn_action == 0:
                    algorithm_counters[0] += 1
                    resource_allocation = maximum_throughput_scheduler(users=list(self.dqn_simulation.users.values()),
                                                                       resources_available=self.config.num_channels)
                elif dqn_action == 1:
                    algorithm_counters[1] += 1
                    resource_allocation = max_min_fair_scheduler(users=list(self.dqn_simulation.users.values()),
                                                                 resources_available=self.config.num_channels)
                elif dqn_action == 2:
                    algorithm_counters[2] += 1
                    resource_allocation = delay_sensitive_scheduler(users=list(self.dqn_simulation.users.values()),
                                                                    resources_available=self.config.num_channels)
                elif dqn_action == 3:
                    algorithm_counters[3] += 1
                    resource_allocation = ev_only_scheduler(users=list(self.dqn_simulation.users.values()),
                                                            resources_available=self.config.num_channels)

                self.dqn_simulation.step(allocation_vector=resource_allocation)
                self.dqn_simulation.generate_jobs(chance=self.config.new_job_chance)

            # MaxThroughput Sim
            for _ in range(self.config.steps_per_episode):
                # max_throughput_simulation.get_len_job_list()
                allocation_vector = maximum_throughput_scheduler(users=list(max_throughput_simulation.users.values()),
                                                                 resources_available=self.config.num_channels)
                max_throughput_simulation.step(allocation_vector)
                max_throughput_simulation.generate_jobs(chance=self.config.new_job_chance)

            # MaxMinFair Sim
            for _ in range(self.config.steps_per_episode):
                allocation_vector = max_min_fair_scheduler(users=list(max_min_fair_simulation.users.values()),
                                                           resources_available=self.config.num_channels)
                max_min_fair_simulation.step(allocation_vector)
                max_min_fair_simulation.generate_jobs(chance=self.config.new_job_chance)

            # DelaySensitive Sim
            for _ in range(self.config.steps_per_episode):
                allocation_vector = delay_sensitive_scheduler(users=list(delay_sensitive_simulation.users.values()),
                                                              resources_available=self.config.num_channels)
                delay_sensitive_simulation.step(allocation_vector)
                delay_sensitive_simulation.generate_jobs(chance=self.config.new_job_chance)

            # EV only Sim
            for _ in range(self.config.steps_per_episode):
                allocation_vector = ev_only_scheduler(users=list(ev_only_simulation.users.values()),
                                                      resources_available=self.config.num_channels)
                ev_only_simulation.step(allocation_vector)
                ev_only_simulation.generate_jobs(chance=self.config.new_job_chance)

            # Collect statistics----------------------------------------------------------------------------------------
            dqn_mean_throughput_per_episode[episode_id] = np.mean(self.dqn_simulation.sum_capacity)
            dqn_datarate_satisfaction_per_episode[episode_id] = np.sum(self.dqn_simulation.datarate_satisfaction)
            dqn_latency_violations_per_episode[episode_id] = np.sum(self.dqn_simulation.jobs_lost)
            dqn_rewards_per_episode[episode_id] = np.sum(self.dqn_simulation.rewards)
            dqn_ev_latency_violations_per_episode[episode_id] = np.sum(self.dqn_simulation.jobs_lost_EV_only)

            max_throughput_mean_throughput_per_episode[episode_id] = np.mean(max_throughput_simulation.sum_capacity)
            max_throughput_datarate_satisfaction_per_episode[episode_id] = np.sum(
                max_throughput_simulation.datarate_satisfaction)
            max_throughput_latency_violations_per_episode[episode_id] = np.sum(max_throughput_simulation.jobs_lost)
            max_throughput_rewards_per_episode[episode_id] = np.sum(max_throughput_simulation.rewards)
            max_throughput_ev_latency_violations_per_episode[episode_id] = np.sum(max_throughput_simulation.jobs_lost_EV_only)

            max_min_fair_mean_throughput_per_episode[episode_id] = np.mean(max_min_fair_simulation.sum_capacity)
            max_min_fair_datarate_satisfaction_per_episode[episode_id] = np.sum(
                max_min_fair_simulation.datarate_satisfaction)
            max_min_fair_latency_violations_per_episode[episode_id] = np.sum(max_min_fair_simulation.jobs_lost)
            max_min_fair_rewards_per_episode[episode_id] = np.sum(max_min_fair_simulation.rewards)
            max_min_fair_ev_latency_violations_per_episode[episode_id] = np.sum(max_min_fair_simulation.jobs_lost_EV_only)

            delay_sensitive_mean_throughput_per_episode[episode_id] = np.mean(
                delay_sensitive_simulation.sum_capacity)
            delay_sensitive_datarate_satisfaction_per_episode[episode_id] = np.sum(
                delay_sensitive_simulation.datarate_satisfaction)
            delay_sensitive_latency_violations_per_episode[episode_id] = np.sum(
                delay_sensitive_simulation.jobs_lost)
            delay_sensitive_rewards_per_episode[episode_id] = np.sum(delay_sensitive_simulation.rewards)
            delay_sensitive_ev_latency_violations_per_episode[episode_id] = np.sum(delay_sensitive_simulation.jobs_lost_EV_only)

            ev_only_mean_throughput_per_episode[episode_id] = np.mean(ev_only_simulation.sum_capacity)
            ev_only_datarate_satisfaction_per_episode[episode_id] = np.sum(ev_only_simulation.datarate_satisfaction)
            ev_only_latency_violations_per_episode[episode_id] = np.sum(ev_only_simulation.jobs_lost)
            ev_only_rewards_per_episode[episode_id] = np.sum(ev_only_simulation.rewards)
            ev_only_ev_latency_violations_per_episode[episode_id] = np.sum(ev_only_simulation.jobs_lost_EV_only)

            # Reset simulation for the next episode---------------------------------------------------------------------
            self.dqn_simulation.reset()
            max_throughput_simulation.reset()
            max_min_fair_simulation.reset()
            delay_sensitive_simulation.reset()
            ev_only_simulation.reset()

            # Progress print--------------------------------------------------------------------------------------------
            if episode_id % int(self.config.num_episodes / 100) == 0:
                completion = np.round((episode_id + 1) / self.config.num_episodes * 100, 1)
                print('\rProgress:', completion, '%', end='')

        print('\r ..Done', flush=True)

        # Logging-------------------------------------------------------------------------------------------------------
        join(self.config.log_path, 'testing_algorithm_selection.gstor')
        with gzip.open(join(self.config.log_path, 'testing_dqn_results.gstor'), 'wb') as file:
            pickle.dump([dqn_mean_throughput_per_episode,
                         dqn_datarate_satisfaction_per_episode,
                         dqn_latency_violations_per_episode,
                         dqn_rewards_per_episode,
                         dqn_ev_latency_violations_per_episode],
                        file)
        with gzip.open(join(self.config.log_path, 'testing_max_throughput_results.gstor'), 'wb') as file:
            pickle.dump([max_throughput_mean_throughput_per_episode,
                         max_throughput_datarate_satisfaction_per_episode,
                         max_throughput_latency_violations_per_episode,
                         max_throughput_rewards_per_episode,
                         max_throughput_ev_latency_violations_per_episode],
                        file)
        with gzip.open(join(self.config.log_path, 'testing_maxminfair_results.gstor'), 'wb') as file:
            pickle.dump([max_min_fair_mean_throughput_per_episode,
                         max_min_fair_datarate_satisfaction_per_episode,
                         max_min_fair_latency_violations_per_episode,
                         max_min_fair_rewards_per_episode,
                         max_min_fair_ev_latency_violations_per_episode],
                        file)
        with gzip.open(join(self.config.log_path, 'testing_delay_sensitive_results.gstor'), 'wb') as file:
            pickle.dump([delay_sensitive_mean_throughput_per_episode,
                         delay_sensitive_datarate_satisfaction_per_episode,
                         delay_sensitive_latency_violations_per_episode,
                         delay_sensitive_rewards_per_episode,
                         delay_sensitive_ev_latency_violations_per_episode],
                        file)
        with gzip.open(join(self.config.log_path, 'testing_algorithm_selection.gstor'), 'wb') as file:
            pickle.dump(algorithm_counters, file)

        # Plotting------------------------------------------------------------------------------------------------------
        fig_dqn_algorithm_selection = plt.figure()
        plt.title('DQN Algorithm Selection')
        plt.bar([0, 1, 2, 3], algorithm_counters / sum(algorithm_counters))
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'EV First'])
        plt.ylabel('Percentage selected')
        plt.tight_layout()

        fig_throughput_comparison = plt.figure()
        plt.title('Algorithm Throughput comparison')
        plt.bar(0, np.mean(max_throughput_mean_throughput_per_episode), color=self.config.ccolor0)
        plt.bar(1, np.mean(max_min_fair_mean_throughput_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_mean_throughput_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(dqn_mean_throughput_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(ev_only_mean_throughput_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DQN Adaptive', 'EV First'])
        plt.ylabel('Normalized Throughput')
        plt.tight_layout()

        fig_timeouts_comparison = plt.figure()
        plt.title('Latency constraint violations comparison')
        plt.bar(0, np.mean(max_throughput_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(1, np.mean(max_min_fair_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(dqn_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(ev_only_latency_violations_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DQN Adaptive', 'EV First'])
        plt.ylabel('Mean violations per episode')
        plt.tight_layout()

        fig_datarate_satisfaction = plt.figure()
        plt.title('Sum data rate satisfaction comparison')
        plt.bar(0, np.mean(max_throughput_datarate_satisfaction_per_episode), color=self.config.ccolor0)
        plt.bar(1, np.mean(max_min_fair_datarate_satisfaction_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_datarate_satisfaction_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(dqn_datarate_satisfaction_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(ev_only_datarate_satisfaction_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DQN Adaptive', 'EV First'])
        plt.ylabel('Mean sum data rate satisfaction per episode')
        plt.tight_layout()

        fig_reward_comparison = plt.figure()
        plt.title('Combined metric comparison')
        plt.bar(0, np.mean(max_throughput_rewards_per_episode), color=self.config.ccolor0)
        plt.bar(1, np.mean(max_min_fair_rewards_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_rewards_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(dqn_rewards_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(ev_only_rewards_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DQN Adaptive', 'EV First'])
        plt.ylabel('Mean combined metric per episode')
        plt.tight_layout()

        fig_timeouts_ev_comparison = plt.figure()
        plt.title('Timeouts for EV only')
        plt.bar(0, np.mean(max_throughput_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(1, np.mean(max_min_fair_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(2, np.mean(delay_sensitive_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(3, np.mean(dqn_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.bar(4, np.mean(ev_only_ev_latency_violations_per_episode), color=self.config.ccolor0)
        plt.grid(alpha=.25, axis='y')
        plt.xticks([0, 1, 2, 3, 4], ['Maximum Throughput', 'Max-Min-Fair', 'Delay Sensitive', 'DQN Adaptive', 'EV First'])
        plt.ylabel('Mean EV timeouts per episode')
        plt.tight_layout()
