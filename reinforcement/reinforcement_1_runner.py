import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import gzip, pickle

from reinforcement.imports.dqn import DQN
from reinforcement.imports.simulation_1 import Simulation
from supervised.imports.wmmse import wmmse


class Runner:
    def __init__(self, config, online_normalization_buffer):
        self.config = config
        self.online_normalization_buffer = online_normalization_buffer

        if not self.config.rainbow['PrioReplay']:  # For Prioritized Experience Replay
            self.prioritization_factors = [0, 0]
        else:
            self.prioritization_factors = self.config.prioritization_factors

        self.epsilon: float = 0
        self.p_max_per_vehicle = self.config.p_max

        self.dqn_total_rewards = np.empty(self.config.num_episodes)
        self.random_total_rewards = np.empty(self.config.num_episodes)
        self.max_total_rewards = np.empty(self.config.num_episodes)
        self.wmmse_total_rewards = np.empty(self.config.num_episodes)

        # Simulations---------------------------------------------------------------------------------------------------
        self.simulation_dqn = Simulation(num_vehicles=self.config.num_vehicles,
                                         vehicles_initial=self.config.vehicles_initial,
                                         path_loss_exponent=self.config.path_loss_exponent,
                                         p_max_per_vehicle=self.p_max_per_vehicle,
                                         num_channels=self.config.num_channels,
                                         max_timing_constraint=self.config.max_timing_constraint,
                                         online_normalization_buffer=self.online_normalization_buffer)

        self.simulation_random = Simulation(num_vehicles=self.config.num_vehicles,
                                            vehicles_initial=self.config.vehicles_initial,
                                            path_loss_exponent=self.config.path_loss_exponent,
                                            p_max_per_vehicle=self.p_max_per_vehicle,
                                            num_channels=self.config.num_channels,
                                            max_timing_constraint=self.config.max_timing_constraint,
                                            online_normalization_buffer=self.online_normalization_buffer)

        self.simulation_max = Simulation(num_vehicles=self.config.num_vehicles,
                                         vehicles_initial=self.config.vehicles_initial,
                                         path_loss_exponent=self.config.path_loss_exponent,
                                         p_max_per_vehicle=self.p_max_per_vehicle,
                                         num_channels=self.config.num_channels,
                                         max_timing_constraint=self.config.max_timing_constraint,
                                         online_normalization_buffer=self.online_normalization_buffer)

        self.simulation_wmmse = Simulation(num_vehicles=self.config.num_vehicles,
                                           vehicles_initial=self.config.vehicles_initial,
                                           path_loss_exponent=self.config.path_loss_exponent,
                                           p_max_per_vehicle=self.p_max_per_vehicle,
                                           num_channels=self.config.num_channels,
                                           max_timing_constraint=self.config.max_timing_constraint,
                                           online_normalization_buffer=self.online_normalization_buffer)

        # DQN-----------------------------------------------------------------------------------------------------------
        self.dqn = DQN(num_channels=self.config.num_channels,
                       num_levels=self.config.num_levels,
                       size_state=len(self.simulation_dqn.states[0]),
                       num_hidden=self.config.num_hidden,
                       rainbow=self.config.rainbow,
                       num_dueling=self.config.num_dueling,
                       learning_rate=self.config.lr,
                       batch_size=self.config.batch_size,
                       min_experiences=self.config.min_experiences,
                       max_experiences=self.config.max_experiences,
                       min_priority=self.config.min_priority,
                       prioritization_factors=self.prioritization_factors,
                       gamma=self.config.gamma,
                       loss_function=self.config.loss_function)

    def get_max_channel(self, state):
        # Returns action: Best channel, max power
        max_channel = np.argmax(state)
        if self.config.num_levels == 1:
            action = max_channel
        else:
            action = (self.config.num_levels - 1) * (max_channel + 1)

        return action

    def should_save_cp(self, current_cp_reward, episode_id, best_checkpoint_reward, interval):
        cp_improvement = current_cp_reward / best_checkpoint_reward  # improvement in %
        if cp_improvement > 1.01:  # 1% better than the best cp
            benchmark_improvement = current_cp_reward / self.random_total_rewards[
                                                        max(0, episode_id - interval):(episode_id + 1)].mean()
            if benchmark_improvement > 1.05:  # 5% better than the benchmark
                return True

        return False

    def step_simulations(self):
        # DQN simulation------------------------------------------------------------------------------------------------
        # Determine actors' actions-----------------------------
        dqn_actions = []  # DQN Output
        dqn_actions_channel = []  # DQN Output translated to channel choice
        dqn_actions_usage = []  # DQN Output translated to power level choice in % of p_max
        for vehicle_id in range(self.config.vehicles_controlled):  # Controlled vehicles
            dqn_actions.append(self.dqn.get_action(state=self.simulation_dqn.states[vehicle_id], epsilon=self.epsilon))
            dqn_action_channel, dqn_action_usage = self.dqn.translate_action(dqn_actions[-1])
            dqn_actions_channel.append(dqn_action_channel)
            dqn_actions_usage.append(dqn_action_usage)

        for vehicle_id in range(self.config.vehicles_controlled, self.config.num_vehicles):  # Random vehicles
            dqn_actions.append(self.dqn.get_action(state=self.simulation_dqn.states[vehicle_id], epsilon=1))
            dqn_action_channel, dqn_action_usage = self.dqn.translate_action(dqn_actions[-1])
            dqn_actions_channel.append(dqn_action_channel)
            dqn_actions_usage.append(dqn_action_usage)

        # Simulate environment developing-----------------------
        self.simulation_dqn.step(vehicles_step=self.config.vehicles_step,
                                 p_max_per_vehicle=self.p_max_per_vehicle, sigma_per_vehicle=self.config.sigma,
                                 actions_channel=dqn_actions_channel, actions_usage=dqn_actions_usage,
                                 lambda_reward=self.config.lambda_reward,
                                 update_p_max_per_vehicle=self.config.random_power_assignment,
                                 online_normalization_buffer=self.online_normalization_buffer)

        # Random action simulation--------------------------------------------------------------------------------------
        random_actions = []
        random_actions_channel = []
        random_actions_usage = []
        for vehicle_id in range(self.config.num_vehicles):
            random_actions.append(self.dqn.get_action(state=self.simulation_random.states[vehicle_id], epsilon=1))
            random_action_channel, random_action_usage = self.dqn.translate_action(random_actions[-1])
            random_actions_channel.append(random_action_channel)
            random_actions_usage.append(random_action_usage)

        # Simulate environment developing-----------------------
        self.simulation_random.step(vehicles_step=self.config.vehicles_step,
                                    p_max_per_vehicle=self.p_max_per_vehicle, sigma_per_vehicle=self.config.sigma,
                                    actions_channel=random_actions_channel, actions_usage=random_actions_usage,
                                    lambda_reward=self.config.lambda_reward,
                                    update_p_max_per_vehicle=self.config.random_power_assignment,
                                    online_normalization_buffer=self.online_normalization_buffer)

        # Best Channel Max Power simulation-----------------------------------------------------------------------------
        max_actions = []
        max_actions_channel = []
        max_actions_usage = []
        for vehicle_id in range(self.config.num_vehicles):
            max_actions.append(self.get_max_channel(self.simulation_max.csi[vehicle_id]))
            max_action_channel, max_action_usage = self.dqn.translate_action(max_actions[-1])
            max_actions_channel.append(max_action_channel)
            max_actions_usage.append(max_action_usage)

        # Simulate environment developing-----------------------
        self.simulation_max.step(vehicles_step=self.config.vehicles_step,
                                 p_max_per_vehicle=self.p_max_per_vehicle, sigma_per_vehicle=self.config.sigma,
                                 actions_channel=max_actions_channel, actions_usage=max_actions_usage,
                                 lambda_reward=self.config.lambda_reward,
                                 update_p_max_per_vehicle=self.config.random_power_assignment,
                                 online_normalization_buffer=self.online_normalization_buffer)

        if self.config.num_channels == 1:
            # wmmse simulation------------------------------------------------------------------------------------------
            h_csi = np.reshape(np.repeat(np.sqrt(self.simulation_wmmse.csi), self.config.num_vehicles),
                               (self.config.num_vehicles, self.config.num_vehicles))
            wmmse_actions, _ = wmmse(P_max=self.p_max_per_vehicle[0], num_channels=self.config.num_vehicles, h_csi=h_csi,
                                  sigma=self.config.sigma, alpha=np.ones(self.config.num_vehicles))
            wmmse_actions_channel = np.zeros(self.config.num_vehicles, dtype=int).tolist()
            wmmse_actions_usage = wmmse_actions / self.p_max_per_vehicle
            # print(wmmse_actions_usage)

            # Simulate environment developing-----------------------
            self.simulation_wmmse.step(vehicles_step=self.config.vehicles_step,
                                       p_max_per_vehicle=self.p_max_per_vehicle, sigma_per_vehicle=self.config.sigma,
                                       actions_channel=wmmse_actions_channel, actions_usage=wmmse_actions_usage,
                                       lambda_reward=self.config.lambda_reward,
                                       update_p_max_per_vehicle=self.config.random_power_assignment,
                                       online_normalization_buffer=self.online_normalization_buffer)

        return dqn_actions

    def train(self):
        self.epsilon = self.config.epsilon

        fig, ax = plt.subplots()
        vehicles_trajectory = np.array([[vehicle.position for vehicle in self.simulation_dqn.vehicles]])
        colormap = cm.get_cmap('viridis', self.config.num_vehicles)
        ax.scatter(vehicles_trajectory[-1, :, 0], vehicles_trajectory[-1, :, 1],
                   cmap=colormap, c=np.arange(self.config.num_vehicles), marker='x')

        checkpoint_counter: int = 0
        best_checkpoint_reward: float = 1e-20

        for episode_id in range(self.config.num_episodes):
            dqn_episode_rewards = 0
            random_episode_rewards = 0
            max_episode_rewards = 0

            for _ in range(self.config.steps_per_episode):
                dqn_state_old = self.simulation_dqn.states.copy()

                dqn_actions = self.step_simulations()

                dqn_episode_rewards += self.simulation_dqn.rewards.mean()
                random_episode_rewards += self.simulation_random.rewards.mean()
                max_episode_rewards += self.simulation_max.rewards.mean()

                self.p_max_per_vehicle = self.simulation_dqn.p_max_per_vehicle

                # Add new experiences-----------------------------------
                for vehicle_id in range(self.config.num_vehicles):
                    self.dqn.add_experience(state=dqn_state_old[vehicle_id],
                                            action=dqn_actions[vehicle_id],
                                            reward=self.simulation_dqn.rewards[vehicle_id],
                                            following_state=self.simulation_dqn.states[vehicle_id])
                self.dqn.train()

                # Copy params to target net periodically----------------
                if episode_id % self.config.copy_step == 0:
                    self.dqn.copy_weights()

                # First episode, record trajectory for plot--------------
                if episode_id == 0:
                    vehicles_trajectory = np.append(vehicles_trajectory, np.array(
                        [[vehicle.position for vehicle in self.simulation_dqn.vehicles]]), axis=0)

            # Log rewards------------------------------------
            self.dqn_total_rewards[episode_id] = dqn_episode_rewards
            self.random_total_rewards[episode_id] = random_episode_rewards
            self.max_total_rewards[episode_id] = max_episode_rewards

            # Anneal Parameters------------------------------
            # Decay random action probability
            self.epsilon = max(self.config.epsilon_min, self.epsilon - self.config.epsilon_decay)
            # Decay Learning Rate
            self.dqn.reduce_learning_rate(lr_decay=self.config.lr_decay, lr_min=self.config.lr_min)
            # Increase experience weight effect on sampling
            self.dqn.prioritization_factors[1] = min(
                1, self.dqn.prioritization_factors[1] + self.config.prioritization_factor_gain)

            # Log and plot trajectory of first episode--------
            if episode_id == 0:  # First Episode
                ax.scatter(vehicles_trajectory[-1, :, 0], vehicles_trajectory[-1, :, 1],
                           cmap=colormap, c=np.arange(self.config.num_vehicles), marker='o')
                for vehicle_id in range(self.config.num_vehicles):
                    ax.plot(vehicles_trajectory[:, vehicle_id, 0], vehicles_trajectory[:, vehicle_id, 1],
                            c=colormap(vehicle_id), linewidth=1)
                ax.scatter(self.simulation_dqn.towers[:, 0], self.simulation_dqn.towers[:, 1],
                           marker='^', s=200, c='black')
                ax.grid(alpha=.25)
                plt.title('Vehicle Trajectories')
                fig.tight_layout()

                with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\trajectories_vehicles.gstor',
                               'wb') as file:
                    pickle.dump(vehicles_trajectory, file)
                with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\trajectories_towers.gstor',
                               'wb') as file:
                    pickle.dump(self.simulation_dqn.towers, file)

            # Reset simulation environments for the next episode
            self.simulation_dqn.reset(self.p_max_per_vehicle, self.online_normalization_buffer)
            self.simulation_random.reset(self.p_max_per_vehicle, self.online_normalization_buffer)
            self.simulation_max.reset(self.p_max_per_vehicle, self.online_normalization_buffer)

            # Print gradient magnitude, analytics------------
            if (episode_id + 1) % 5 == 0:
                # if len(self.dqn.gradient_magnitude) > 0:  # Account for episodes without training
                #     print('Recent Max gradient: ' + str(np.max(self.dqn.gradient_magnitude)) +
                #           ', Recent mean gradient: ' + str(np.round(np.mean(self.dqn.gradient_magnitude), 1))
                #           )
                self.dqn.gradient_magnitude = np.array([], dtype=int)  # Reset log

            # Print rewards, analytics-----------------------
            if (episode_id + 1) % 10 == 0:
                dqn_avg_rewards = self.dqn_total_rewards[max(0, episode_id - 20):(episode_id + 1)].mean()
                random_avg_rewards = self.random_total_rewards[max(0, episode_id - 20):(episode_id + 1)].mean()
                max_avg_rewards = self.max_total_rewards[max(0, episode_id - 20):(episode_id + 1)].mean()

                print('Episode ' + str(episode_id + 1) +
                      ': Recent mean reward per episode: ' + str(np.round(dqn_avg_rewards, 1)) +
                      ', Current chance of random actions: ' + str(np.round(self.epsilon, 3)) +
                      ', Random mean reward: ' + str(np.round(random_avg_rewards, 1)) +
                      ', Max mean reward: ' + str(np.round(max_avg_rewards, 1))
                      )

            # Checkpoint-------------------------------------
            interval = 300
            dqn_avg_rewards_longterm = self.dqn_total_rewards[max(0, episode_id - interval):(episode_id + 1)].mean()
            if self.should_save_cp(dqn_avg_rewards_longterm, episode_id, best_checkpoint_reward, interval=interval):
                checkpoint_path = self.config.checkpoint_path + '_cp' + str(checkpoint_counter) + \
                                  '_' + str(round(dqn_avg_rewards_longterm, 1)) + \
                                  '_ep' + str(episode_id)
                self.dqn.q_primary.predict(np.random.rand(len(self.simulation_dqn.states[0]))[np.newaxis])
                self.dqn.q_primary.save(checkpoint_path, save_format='tf')

                checkpoint_counter += 1
                best_checkpoint_reward = dqn_avg_rewards_longterm

                print('Saved Checkpoint #' + str(checkpoint_counter - 1) +
                      ' at reward ' + str(round(best_checkpoint_reward, 1)))

        # Save logs--------------------------------------
        with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\training_episode_rewards_dqn.gstor',
                       'wb') as file:
            pickle.dump(self.dqn_total_rewards, file)
        with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\training_episode_rewards_random.gstor',
                       'wb') as file:
            pickle.dump(self.random_total_rewards, file)
        with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\training_episode_rewards_maxpower.gstor',
                       'wb') as file:
            pickle.dump(self.max_total_rewards, file)

        # Save q_primary---------------------------------
        # Load with tf.keras.models.load_model('path')
        # required bc custom model save bug:
        self.dqn.q_primary.predict(np.random.rand(len(self.simulation_dqn.states[0]))[np.newaxis])
        self.dqn.q_primary.save(self.config.model_path, save_format='tf')

    def test(self):
        self.dqn.q_primary = tf.keras.models.load_model(self.config.model_path)
        self.epsilon = 0

        for episode_id in range(self.config.num_episodes):
            dqn_rewards = 0
            random_rewards = 0
            max_rewards = 0
            wmmse_rewards = 0

            for _ in range(self.config.steps_per_episode):
                self.step_simulations()

                dqn_rewards += self.simulation_dqn.rewards.mean()
                random_rewards += self.simulation_random.rewards.mean()
                max_rewards += self.simulation_max.rewards.mean()
                wmmse_rewards += self.simulation_wmmse.rewards.mean()

            # Log rewards--------------------------------------------------------
            self.dqn_total_rewards[episode_id] = dqn_rewards
            self.random_total_rewards[episode_id] = random_rewards
            self.max_total_rewards[episode_id] = max_rewards
            self.wmmse_total_rewards[episode_id] = wmmse_rewards

            # Reset simulation environments for the next episode-----------------
            self.simulation_dqn.reset(self.config.p_max, self.online_normalization_buffer)
            self.simulation_random.reset(self.config.p_max, self.online_normalization_buffer)
            self.simulation_max.reset(self.config.p_max, self.online_normalization_buffer)
            self.simulation_wmmse.reset(self.config.p_max, self.online_normalization_buffer)

        with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\testing_episode_rewards_dqn.gstor',
                       'wb') as file:
            pickle.dump(self.dqn_total_rewards, file)
        with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\testing_episode_rewards_random.gstor',
                       'wb') as file:
            pickle.dump(self.random_total_rewards, file)
        with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\testing_episode_rewards_maxpower.gstor',
                       'wb') as file:
            pickle.dump(self.max_total_rewards, file)
        with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\testing_episode_rewards_wmmse.gstor',
                       'wb') as file:
            pickle.dump(self.wmmse_total_rewards, file)
