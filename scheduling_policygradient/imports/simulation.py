import numpy as np

from scheduling_policygradient.imports.user import UserNormal, UserLowLatency, UserHighDatarate
from scheduling_policygradient.imports.base_station import BaseStation


class Simulation:
    def __init__(self, config):
        self.config = config
        # Generate Users------------------------------------------------------------------------------------------------
        self.base_station = BaseStation()
        user_counter = 0
        self.users = dict()
        for _ in range(config.num_users_normal):
            self.users[user_counter] = UserNormal(pos_base_station=self.base_station.pos, user_id=user_counter)
            user_counter += 1
        for _ in range(config.num_users_high_datarate):
            self.users[user_counter] = (UserHighDatarate(pos_base_station=self.base_station.pos, user_id=user_counter))
            user_counter += 1
        for _ in range(config.num_users_low_latency):
            self.users[user_counter] = (UserLowLatency(pos_base_station=self.base_station.pos, user_id=user_counter))
            user_counter += 1

        # Generate Initial Job Queue------------------------------------------------------------------------------------
        self.generate_jobs(chance=config.new_job_chance)

        # Statistics----------------------------------------------------------------------------------------------------
        self.total_units_requested = np.array([])
        self.sum_capacity = np.array([])
        self.jobs_lost = np.array([])  # Number of jobs lost from violating latency constraints
        self.datarate_satisfaction = np.array([])
        self.rewards = np.array([])

    def generate_jobs(self, chance):
        for user in self.users.values():  # add a job per user..
            if np.random.random() < chance:  # .. at an initial chance
                user.generate_job()

    def get_len_job_list(self):
        units_requested = 0
        for user in self.users.values():
            units_requested += user.units_requested
        # print('Total resources requested: '+str(int(units_requested)))

        return units_requested

    def build_reward(self):
        reward = 0
        # print('c', self.sum_capacity[-1])
        # print('j', self.jobs_lost[-1])
        # print('d', self.datarate_satisfaction[-1])
        reward += self.config.lambda_reward[0] * self.sum_capacity[-1]  # Throughput
        reward -= self.config.lambda_reward[1] * self.jobs_lost[-1]  # Latency constraint violations
        reward += self.config.lambda_reward[2] * self.datarate_satisfaction[-1]
        # print('r', reward)

        return reward

    def step(self, allocation_vector):
        sum_capacity: float = 0
        jobs_lost: int = 0
        datarate_satisfaction: float = 0
        allocation_vector_copy = allocation_vector.copy()

        self.total_units_requested = self.get_len_job_list()

        for user in self.users.values():
            # Increment user runtime while they have a job in queue -> data rates not tied to chance
            if user.jobs:
                user.runtime += 1
            # Allocate resources to jobs, oldest first------------------------------------------------------------------
            job_delays = []
            for job in user.jobs:  # List delay of jobs
                job_delays += [job.delay]

            while allocation_vector_copy[user.user_id] > 0:  # Allocate blocks of jobs, oldest jobs first
                highest_delay_id = int(np.argmax(job_delays))
                units_sent = min(allocation_vector_copy[user.user_id], user.jobs[highest_delay_id].size)

                allocation_vector_copy[user.user_id] -= units_sent
                user.jobs[highest_delay_id].size -= units_sent
                if user.jobs[highest_delay_id].size == 0:  # Job fully processed
                    user.jobs.pop(highest_delay_id)  # Remove from job list
                    job_delays.pop(highest_delay_id)  # Remove from delay ranking

                user.units_received += units_sent  # Update internal statistics
                user.update_statistics()
            user.update_channel_quality(pos_base_station=self.base_station.pos)

            # Increment delay metric to remaining jobs, remove jobs over delay------------------------------------------
            for job in user.jobs:
                job.delay += 1
                if job.delay >= user.latency_max:
                    user.jobs.pop(user.jobs.index(job))
                    user.jobs_lost_to_timeout += 1
                    jobs_lost += 1  # Count latency constraint violation
                    user.update_statistics()

            # Update sum_capacity metric--------------------------------------------------------------------------------
            sum_capacity += allocation_vector[user.user_id] * np.log10(
                1 + user.channel_quality * user.signal_noise_ratio)

            # Update datarate_satisfaction metric-----------------------------------------------------------------------
            datarate_satisfaction += min(user.datarate_satisfaction, 1)  # 1 for minimum datarate met, less else

        self.sum_capacity = np.append(self.sum_capacity, sum_capacity)
        self.jobs_lost = np.append(self.jobs_lost, jobs_lost)
        self.datarate_satisfaction = np.append(self.datarate_satisfaction, datarate_satisfaction)
        self.rewards = np.append(self.rewards, self.build_reward())

    def reset(self):
        self.__init__(config=self.config)
