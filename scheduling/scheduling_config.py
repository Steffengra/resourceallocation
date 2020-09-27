
class Config:
    def __init__(self):
        self.num_episodes: int = 20_000
        self.steps_per_episode: int = 50

        self.user_snr: float = 20
        self.path_loss_exponent: float = 1

        self.num_channels: int = 16
        self.new_job_chance: float = .20

        self.num_users_normal: int = 5
        self.num_users_high_datarate: int = 2
        self.num_users_low_latency: int = 2
        self.num_users = self.num_users_normal + self.num_users_high_datarate + self.num_users_low_latency

        # Normal user profile-------------------
        self.normal_datarate: int = 9
        self.normal_latency: int = 20
        self.normal_job_size_max: int = 30
        # HighDatarate user profile-------------
        self.high_datarate_datarate: int = 11
        self.high_datarate_latency: int = self.normal_latency
        self.high_datarate_job_size_max: int = 40
        # LowLatency user profile---------------
        self.low_latency_datarate: int = self.normal_datarate
        self.low_latency_latency: int = 1
        self.low_latency_job_size_max: int = 20

        # Training Parameters-------------------------------------------------------------------------------------------
        self.num_hidden: list = [300, 300, 300, 300]  # Hidden layers' numbers of nodes
        self.epsilon: float = .99  # Initial chance at random exploration
        self.epsilon_min: float = 0.00  # Minimum chance at random exploration
        # linear decay to epsilon_min after x% episodes:
        self.epsilon_decay: float = (self.epsilon - self.epsilon_min) / (0.8 * self.num_episodes)
        self.gamma: float = 0.90  # Future reward discount
        self.loss_function: str = 'mse'  # mse, huber
        self.learning_rate: float = 10 * 1e-5  # learning rate
        self.lr_min = 0.9 * self.learning_rate  # Minimum learning rate
        self.lr_decay: float = 0  # lr is adjusted every episode linearly, lr_new = lr_old-lr_decay
        self.min_experiences: int = 100  # Min experiences in buffer to start training
        self.max_experiences: int = 10_000  # Max experience buffer size, fifo
        self.batch_size: int = 30  # Training batch size
        self.copy_step: int = 25  # To Target Network
        self.lambda_reward: tuple = (1, 1, .25)  # Weighting of reward sum components

        self.model_path: str = 'SavedModels\\dqn\\q_primary'  # Path for saving the final trained model
        self.checkpoint_path: str = 'SavedModels\\dqn\\cps\\q_primary'  # Path for saving checkpoints

        # DQN enhancement parameters------------------------------------------------------------------------------------
        self.rainbow: dict = {'Double-Q': True,
                              'DuelingDQN': True,
                              'PrioReplay': True,
                              'OnlineInputNormalization': False,
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
