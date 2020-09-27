import numpy as np

from scheduling.scheduling_config import Config
from scheduling.imports.job import Job


class User:
    def __init__(self, user_id: int, pos_base_station: list):
        config = Config()

        self.user_id: int = user_id
        self.type: str = 'none'
        self.color: str = config.ccolor0

        self.pos: np.ndarray = np.random.randint(1, 200, 2)
        self.path_loss_exponent: float = config.path_loss_exponent
        self.signal_noise_ratio: float = config.user_snr

        self.channel_quality: float = 0
        self.update_channel_quality(pos_base_station=pos_base_station)

        self.job_size_max: int = 0
        self.jobs: list = []

        self.datarate_min: int = 0
        self.latency_max: int = 0

        # Statistics
        self.runtime: int = 0
        self.units_requested: int = 0
        self.units_requested_lifetime: int = 0
        self.units_received: int = 0
        self.datarate: float = 0
        self.datarate_satisfaction: float = 0
        self.jobs_lost_to_timeout: int = 0

    def update_channel_quality(self, pos_base_station):
        rayleigh_fading = abs((1 * np.random.randn() + 0) + 1j * (1 * np.random.randn() + 0))

        distance_to_base_station = np.sqrt(np.sum(np.power(self.pos - pos_base_station, 2))) + 0.01  # +0.01 avoids 0
        path_loss = max(1, np.power(distance_to_base_station, self.path_loss_exponent))

        self.channel_quality = rayleigh_fading / path_loss

    def update_statistics(self):
        # units_requested--------------------
        units_requested = 0
        for job in self.jobs:
            units_requested += job.size
        self.units_requested = units_requested
        # datarate---------------------------
        if self.runtime != 0:
            self.datarate = self.units_received / self.runtime
            # self.datarate_satisfaction = self.datarate / self.datarate_min
            self.datarate_satisfaction = self.units_received / self.units_requested_lifetime

    def generate_job(self):
        size = np.random.randint(1, self.job_size_max + 1)
        job = Job(size=size)
        self.jobs.append(job)
        self.units_requested_lifetime += job.size
        self.update_statistics()


class UserNormal(User):
    def __init__(self, user_id, pos_base_station):
        super().__init__(user_id=user_id, pos_base_station=pos_base_station)
        self.type = 'Normal'
        config = Config()
        self.color = config.ccolor2
        self.datarate_min = config.normal_datarate
        self.latency_max = config.normal_latency

        self.job_size_max = config.normal_job_size_max


class UserLowLatency(User):
    def __init__(self, user_id, pos_base_station):
        super().__init__(user_id=user_id, pos_base_station=pos_base_station)
        self.type = 'LowLatency'
        config = Config()
        self.color = config.ccolor3
        self.datarate_min = config.low_latency_datarate
        self.latency_max = config.low_latency_latency

        self.job_size_max = config.low_latency_job_size_max


class UserHighDatarate(User):
    def __init__(self, user_id, pos_base_station):
        super().__init__(user_id=user_id, pos_base_station=pos_base_station)
        self.type = 'HighDatarate'
        config = Config()
        self.color = config.ccolor4
        self.datarate_min = config.high_datarate_datarate
        self.latency_max = config.high_datarate_latency

        self.job_size_max = config.high_datarate_job_size_max
