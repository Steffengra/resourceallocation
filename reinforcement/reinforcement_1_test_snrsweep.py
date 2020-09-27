import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import gzip, pickle

from reinforcement.reinforcement_1_config import Config
from reinforcement.reinforcement_1_runner import Runner
from reinforcement.imports.online_normalization_buffer import OnlineNormalizationBuffer
from reinforcement.imports.simulation_1 import Simulation


def main():
    config = Config()
    config.num_episodes = 20
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

    num_simulations = 10
    snr = np.linspace(config.snr_range[0], config.snr_range[1], num_simulations)  # SNR in dB

    dqn_mean_reward = np.zeros(num_simulations)
    random_mean_reward = np.zeros(num_simulations)
    max_power_mean_reward = np.zeros(num_simulations)
    wmmse_mean_reward = np.zeros(num_simulations)

    for simulation_id in range(num_simulations):
        p_max = np.float_power(10, snr[simulation_id]/10)
        config.p_max = p_max * np.ones(config.num_vehicles)

        runner = Runner(config=config, online_normalization_buffer=online_normalization_buffer)
        runner.test()

        dqn_mean_reward[simulation_id] = np.mean(runner.dqn_total_rewards)
        random_mean_reward[simulation_id] = np.mean(runner.random_total_rewards)
        max_power_mean_reward[simulation_id] = np.mean(runner.max_total_rewards)
        wmmse_mean_reward[simulation_id] = np.mean(runner.wmmse_total_rewards)

        if (simulation_id + 1) % 2 == 0:
            print(str(simulation_id + 1) + " / " + str(num_simulations))

    with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\snrsweep_testing_mean_episode_rewards_dqn.gstor', 'wb') as file:
        pickle.dump(dqn_mean_reward, file)
    with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\snrsweep_testing_mean_episode_rewards_random.gstor', 'wb') as file:
        pickle.dump(random_mean_reward, file)
    with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\snrsweep_testing_mean_episode_rewards_maxpower.gstor', 'wb') as file:
        pickle.dump(max_power_mean_reward, file)
    with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\snrsweep_testing_mean_episode_rewards_wmmse.gstor', 'wb') as file:
        pickle.dump(wmmse_mean_reward, file)

    dqn_mean_energy_efficiency = dqn_mean_reward / snr
    random_mean_energy_efficiency = dqn_mean_reward / snr
    max_power_mean_energy_efficiency = dqn_mean_reward / snr

    # Plotting-------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.plot(snr, dqn_mean_reward, color=config.ccolor2)
    ax.plot(snr, random_mean_reward, color=config.ccolor3)
    ax.plot(snr, max_power_mean_reward, color=config.ccolor4)
    if config.num_channels == 1:
        ax.plot(snr, wmmse_mean_reward, color='red')

    ax.grid(alpha=.25)
    ax.legend(['DQN', 'Random Action', 'Max Power', 'WMMSE Allocation'])
    ax.set_xlabel('SNR /dB')
    ax.set_ylabel('Mean Reward')

    fig.set_size_inches(w=5.78853, h=3.5)  # \textwidth is w=5.78853 in
    fig.tight_layout()
    # fig.savefig('plots/test.pdf', bbox_inches='tight', dpi=800)
    # fig.savefig('plots/test.pgf')


    # fig2, ax2 = plt.subplots()
    # ax2.plot(snr, dqn_mean_energy_efficiency, color=config.ccolor2)
    # ax2.plot(snr, random_mean_energy_efficiency, color=config.ccolor3)
    # ax2.plot(snr, max_power_mean_energy_efficiency, color=config.ccolor4)
    #
    # ax2.grid(alpha=.25)
    # ax2.legend(['DQN', 'Random Action', 'Max Power'])
    # ax2.set_xlabel('SNR /dB')
    # ax2.set_ylabel('Mean Energy Efficiency')
    #
    # fig2.set_size_inches(w=5.78853, h=3.5)  # \textwidth is w=5.78853 in
    # fig2.tight_layout()
    # fig2.savefig('plots/test.pdf', bbox_inches='tight', dpi=800)
    # fig2.savefig('plots/test.pgf')

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
