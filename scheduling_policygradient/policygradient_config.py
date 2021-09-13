
from os.path import (join, dirname)
from numpy import floor


class Config:
    def __init__(self):
        simulation_title: str = 'ddpg_2'

        self.num_episodes: int = 10_000
        self.steps_per_episode: int = 50

        self.user_snr: float = 20
        self.path_loss_exponent: float = 1

        self.num_channels: int = 16
        self.new_job_chance: float = .20

        self.num_users_per_job: dict = {'Normal': 5,
                                        'High Datarate': 2,
                                        'Low Latency': 2,
                                        'Emergency Vehicle': 1}

        self.lambda_reward: dict = {'Sum Capacity': .25,
                                    'Packet Rate': .25,
                                    'Packet Timeouts': 1,
                                    'EV Packet Timeouts': 1}

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
        self.low_latency_latency: int = 2
        self.low_latency_job_size_max: int = int(floor(self.num_channels / 2))
        # EmergencyVehicle user profile---------
        self.EV_datarate: int = self.normal_datarate
        self.EV_latency: int = 1
        self.EV_job_size_max: int = self.num_channels

        # Training Parameters-------------------------------------------------------------------------------------------
        self.num_hidden: list = [300, 300, 300, 300, 400, 300]  # Hidden layers' numbers of nodes
        self.learning_rate_q: float = 1 * 1e-4  # learning rate for critic
        self.learning_rate_a: float = 1 * 1e-5  # learning rate for actor
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

        # Setup (don't change)------------------------------------------------------------------------------------------
        self.num_users: int = sum(self.num_users_per_job.values())

        self.model_path: str = join(dirname(__file__), 'SavedModels', simulation_title)
        self.log_path: str = join(dirname(__file__), 'logs')

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
