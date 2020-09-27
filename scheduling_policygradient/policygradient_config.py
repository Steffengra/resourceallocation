
class Config:
    def __init__(self):
        self.num_episodes: int = 20_000
        self.steps_per_episode: int = 50

        self.user_snr: float = 20
        self.path_loss_exponent: float = 1

        self.num_channels: int = 16
        self.new_job_chance: float = .20

        self.lambda_reward: tuple = (1, 1, .25)  # Weighting of reward sum components

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
        self.low_latency_job_size_max: int = self.num_channels

        # Training Parameters-------------------------------------------------------------------------------------------
        self.num_hidden: list = [300, 300, 300, 300, 400, 300]  # Hidden layers' numbers of nodes
        self.learning_rate_q: float = 1 * 1e-5  # learning rate
        self.learning_rate_a: float = 1 * 1e-6  # learning rate for actor
        self.gamma: float = 0.00  # Future reward discount
        self.loss_function: str = 'mse'  # mse, huber
        self.min_experiences: int = 100  # Min experiences in buffer to start training
        self.max_experiences: int = 100_000  # Max experience buffer size, fifo
        self.batch_size: int = 128  # Training batch size
        self.copy_step: int = 10  # To Target Network
        self.initial_noise_multiplier: float = 1.5  # allocation + multiplier * rand
        self.noise_decay_to_zero_threshold: float = 0.5  # decay noise multiplier to zero by x% episodes
        self.noise_decay: float = self.initial_noise_multiplier / (  # decay per episode
                self.noise_decay_to_zero_threshold * self.num_episodes)

        self.min_priority: float = 1e-7  # For Prioritized Experience Replay
        self.prioritization_factors: list = [.5, .5]  # [alpha, beta]
        self.prioritization_factor_gain: float = (1 - self.prioritization_factors[1]) / (0.8 * self.num_episodes)

        self.model_path_critic: str = 'SavedModels\\ddpg\\q_primary'  # Path for saving the final trained model
        self.model_path_actor: str = 'SavedModels\\ddpg\\actor'  # Path for saving the final trained model
        self.checkpoint_path: str = 'SavedModels\\ddpg\\cps\\q_primary'  # Path for saving checkpoints

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
