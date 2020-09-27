import numpy as np


class Config:
    def __init__(self):
        # Simulation Parameters-----------------------------------------------------------------------------------------
        self.num_episodes: int = 10_000
        self.steps_per_episode: int = 100
        self.num_vehicles: int = 7  # (3, 4, 1), (5, 4, 2), (5, 3, 3), (7, 1, 2)
        self.num_channels: int = 1  # Base Stations
        self.num_levels: int = 2  # Power usage levels [0,..., 1], len=num_levels
        self.max_timing_constraint: int = 5  # rand(1, max_timing_constraint)

        self.vehicles_controlled: int = -1  # -1 = All vehicles
        self.vehicles_initial: tuple = (100, 10)  # (Mean, Var) of vehicles and towers initial positions
        self.vehicles_step: float = .1  # size of vehicles movement steps

        self.path_loss_exponent: float = 0.00  # PL = 1/dist^PLE

        self.random_power_assignment: bool = True  # Assign new max power per episode
        self.snr_range: tuple = (-10, 30)  # SNR range if random_power_assignment == True
        self.p_max: np.ndarray = 20 * np.ones(self.num_vehicles)
        self.sigma: np.ndarray = np.ones(self.num_vehicles)

        # Training Parameters-------------------------------------------------------------------------------------------
        self.num_hidden: list = [300, 300, 300, 300, ]  # Hidden layers' numbers of nodes
        self.epsilon: float = 0.03  # Initial chance at random exploration
        self.epsilon_min: float = 0.03  # Minimum chance at random exploration
        # linear decay to epsilon_min after x% episodes:
        self.epsilon_decay: float = (self.epsilon - self.epsilon_min) / (0.20 * self.num_episodes)
        self.gamma: float = 0.99  # Future reward discount
        self.loss_function: str = 'mse'  # mse, huber
        self.lr: float = 1 * 1e-5  # learning rate
        self.lr_min = 0.9 * self.lr  # Minimum learning rate
        self.lr_decay: float = 0  # lr is adjusted every episode linearly, lr_new = lr_old-lr_decay
        self.min_experiences: int = 100  # Min experiences in buffer to start training
        self.max_experiences: int = 10_000  # Max experience buffer size, fifo
        self.batch_size: int = 30  # Training batch size
        self.copy_step: int = 25  # To Target Network
        self.lambda_reward: tuple = (1, 0)
        # Weighting of sum-rate capacity (1) and timing constraint (2),
        # 1/(2*max_timing_constraint) has proven good for (2)

        self.model_path: str = 'SavedModels\\reinforcement_1\\q_primary'  # Path for saving the final trained model
        self.checkpoint_path: str = 'SavedModels\\reinforcement_1\\cps\\q_primary'  # Path for saving checkpoints

        # DQN enhancement parameters------------------------------------------------------------------------------------
        self.rainbow: dict = {'Double-Q': True,
                              'DuelingDQN': True,
                              'PrioReplay': True,
                              'OnlineInputNormalization': False,  # Not working, need to manually enable
                              }
        # :DuelingDQN implements a two path output NN.
        # Usage requires adjusting num_hidden, num_dueling. Both value- and advantage stream
        # will each get layers with num_dueling nodes, so 2*num_dueling nodes are added to the network.
        # Without DuelingDQN==True, num_dueling can be ignored
        self.num_dueling: list = [200, ]  # Number of nodes for DuelingDQN V and A streams
        # :PrioReplay implements weighted experience sampling.
        # prioritization_factors [alpha, beta] determine how heavily weights influence sampling probability.
        # Without PrioReplay==True, prioritization_factors can be ignored
        self.min_priority: float = 1e-7  # For Prioritized Experience Replay
        self.prioritization_factors: list = [.5, .5]  # [alpha, beta]
        # Linear increase of "beta", weighting of experiences, toward a max of 1 after x% episodes:
        self.prioritization_factor_gain: float = (1 - self.prioritization_factors[1]) / (0.8 * self.num_episodes)
        # :OnlineInputNormalization normalizes inputs according to their maximum and minimum values
        # encountered during training. Normalization is (x - x_min) / (x_max - x_min) -> [0, 1]
        # Defaults from Rayleigh Distribution and Path Loss for csi normalization factors
        self.default_normalization_min = 0 * 1 / np.power(
            np.sqrt(2 * (2 * self.vehicles_initial[1]) ** 2), self.path_loss_exponent)
        self.default_normalization_max = 6 * 1

        # Plotting------------------------------------------------------------------------------------------------------
        # Branding Palette
        self.color0 = '#000000'  # black
        self.color1 = '#21467a'  # blue
        self.color2 = '#c4263a'  # red
        self.color3 = '#008700'  # green
        self.color4 = '#caa023'  # gold

        # Colorblind Palette
        self.ccolor0 = '#000000'  # use for lines, black
        self.ccolor1 = '#d01b88'  # use for lines, pink
        self.ccolor2 = '#254796'  # use for scatter, blue
        self.ccolor3 = '#307b3b'  # use for scatter, green
        self.ccolor4 = '#caa023'  # use for scatter, gold

        # post init sanity check----------------------------------------------------------------------------------------
        if self.vehicles_controlled > self.num_vehicles:
            print('Config error: vehicles_controlled > num_vehicles')
            self.vehicles_controlled = -1
        if self.vehicles_controlled == -1:
            self.vehicles_controlled = self.num_vehicles
