import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from reinforcement.reinforcement_1_config import Config
from reinforcement.reinforcement_1_runner import Runner
from reinforcement.imports.online_normalization_buffer import OnlineNormalizationBuffer
from reinforcement.imports.simulation_1 import Simulation


def main():
    config = Config()
    config.num_episodes = 2000
    config.random_power_assignment = False

    # Initialize Buffer------------------------------------
    online_normalization_buffer = OnlineNormalizationBuffer(max_sizes=[100_000])
    online_normalization_buffer.power.set_statistics(min=config.snr_range[0], max=config.snr_range[1])
    init_sim = Simulation(num_vehicles=config.num_vehicles,
                          vehicles_initial=config.vehicles_initial,
                          path_loss_exponent=config.path_loss_exponent,
                          p_max_per_vehicle=config.p_max,
                          num_channels=config.num_channels,
                          max_timing_constraint=config.max_timing_constraint,
                          online_normalization_buffer=online_normalization_buffer)
    for _ in range(100):
        for _ in range(config.steps_per_episode):
            init_sim.step(vehicles_step=config.vehicles_step,
                          p_max_per_vehicle=config.p_max, sigma_per_vehicle=config.sigma,
                          actions_channel=[0 for _ in range(config.num_vehicles)],
                          actions_usage=[0 for _ in range(config.num_vehicles)],
                          lambda_reward=config.lambda_reward,
                          update_p_max_per_vehicle=config.random_power_assignment,
                          online_normalization_buffer=online_normalization_buffer)
            online_normalization_buffer.channel_state_information.add_values(np.concatenate(init_sim.csi))
        init_sim.reset(p_max_per_vehicle=config.p_max, online_normalization_buffer=online_normalization_buffer)

    num_simulations = 1

    dqn_mean_reward = np.zeros(num_simulations)
    random_mean_reward = np.zeros(num_simulations)
    max_power_mean_reward = np.zeros(num_simulations)
    wmmse_mean_reward = np.zeros(num_simulations)

    for simulation_id in range(num_simulations):

        runner = Runner(config=config, online_normalization_buffer=online_normalization_buffer)
        runner.test()

        dqn_mean_reward[simulation_id] = np.mean(runner.dqn_total_rewards)
        random_mean_reward[simulation_id] = np.mean(runner.random_total_rewards)
        max_power_mean_reward[simulation_id] = np.mean(runner.max_total_rewards)
        wmmse_mean_reward[simulation_id] = np.mean(runner.wmmse_total_rewards)

        if (simulation_id + 1) % 2 == 0:
            print(str(simulation_id + 1) + " / " + str(num_simulations))


if __name__ == '__main__':
    main()
