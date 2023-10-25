import numpy as np
import matplotlib.pyplot as plt

from reinforcement.reinforcement_1_config import Config
from reinforcement.reinforcement_1_runner import Runner
from reinforcement.imports.online_normalization_buffer import OnlineNormalizationBuffer
from reinforcement.imports.simulation_1 import Simulation


# remove future rewards -> remove target networks? -> [1]
# [1] A Deep Reinforcement Learning-based Approach to Dynamic eMBB/URLLC Multiplexing in 5G NR
# Huber loss
# normalize inputs? with largest seen input? [2] input scaling has been implemented, but not true normalization
# current implementation makes the first few experiences trash - doesnt work at all
# [2] Online Normalization for Training Neural Networks
# subtract a large number from reward to encourage natural exploration?


def main():
    config = Config()

    online_normalization_buffer = OnlineNormalizationBuffer(max_sizes=[100_000])
    # Initialize Buffer------------------------------------
    print('Initializing normalization buffer.. 0.0 %', end='')
    online_normalization_buffer.power.set_statistics(min=config.snr_range[0], max=config.snr_range[1])
    init_sim = Simulation(num_vehicles=config.num_vehicles,
                          vehicles_initial=config.vehicles_initial,
                          path_loss_exponent=config.path_loss_exponent,
                          p_max_per_vehicle=config.p_max,
                          num_channels=config.num_channels,
                          max_timing_constraint=config.max_timing_constraint,
                          online_normalization_buffer=online_normalization_buffer)
    for episode_id in range(100):
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
        if np.mod(episode_id, 10) == 0:
            progress = episode_id / 100 * 100
            print('\rInitializing normalization buffer..', np.round(progress, 1), '%', end='')
    print('\rInitializing normalization buffer.. done', flush=True)

    # print(online_normalization_buffer.channel_state_information.buffer_max_log)
    # print(online_normalization_buffer.channel_state_information.buffer_max_no_outliers_log)
    # print(online_normalization_buffer.channel_state_information.buffer_min_log)
    # print(online_normalization_buffer.channel_state_information.buffer_min_no_outliers_log)
    # print(np.histogram(online_normalization_buffer.channel_state_information.buffer))
    runner = Runner(config=config, online_normalization_buffer=online_normalization_buffer)
    runner.train()

    # Stats----------------------------------------------------------------
    dqn_recent_mean_total = np.array([])
    random_recent_mean_total = np.array([])
    max_recent_mean_total = np.array([])
    for episode_id in range(config.num_episodes):
        dqn_recent_mean_total = np.append(
            dqn_recent_mean_total, runner.dqn_total_rewards[max(0, episode_id - 40):(episode_id + 1)].mean())
        random_recent_mean_total = np.append(
            random_recent_mean_total, runner.random_total_rewards[max(0, episode_id - 40):(episode_id + 1)].mean())
        max_recent_mean_total = np.append(
            max_recent_mean_total, runner.max_total_rewards[max(0, episode_id - 40):(episode_id + 1)].mean())

    # Plotting-------------------------------------------------------------
    fig, ax1 = plt.subplots()

    plt.plot(range(config.num_episodes), dqn_recent_mean_total, color=config.ccolor2)
    plt.plot(range(config.num_episodes), random_recent_mean_total, color=config.ccolor3)
    plt.plot(range(config.num_episodes), max_recent_mean_total, color=config.ccolor4)
    # plt.scatter(range(config.num_episodes), runner.dqn_total_rewards, color=config.ccolor2)
    # plt.scatter(range(config.num_episodes), runner.random_total_rewards, color=config.ccolor3)
    # plt.scatter(range(config.num_episodes), runner.max_total_rewards, color=config.ccolor4)

    ax2 = ax1.twinx()
    # ax1.set_ylim([.6, 1.9])
    ax2.set_ylim([-0.07, 1.1])
    ax2.plot(range(config.num_episodes), np.where(
        config.epsilon - (config.epsilon_decay * np.arange(config.num_episodes)) > config.epsilon_min,
        config.epsilon - (config.epsilon_decay * np.arange(config.num_episodes)),
        config.epsilon_min),
             color=config.ccolor1)  # linear decay

    plt.grid(alpha=0.25)
    fig.legend(['DQN Actors', 'Random Actors', 'Max Channel'], loc='lower right')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Mean Reward')
    ax2.set_ylabel('% Random actions taken', color=config.ccolor1)

    fig.set_size_inches(w=5.78853, h=3.5)  # \textwidth is w=5.78853 in
    plt.tight_layout()
    # plt.savefig('plots/early_random_v_trained.pdf', bbox_inches='tight', dpi=800)
    # plt.savefig('plots/early_random_v_trained.pgf')

    plt.show()


# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'sans-serif',
#     'text.usetex': True,  # inline math
#     'pgf.rcfonts': False,  # Override font configuration
# })

if __name__ == '__main__':
    main()
