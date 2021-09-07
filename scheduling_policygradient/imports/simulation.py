import numpy as np

from scheduling_policygradient.imports.user import UserNormal, UserLowLatency, UserHighDatarate, UserEmergencyVehicle
from scheduling_policygradient.imports.base_station import BaseStation


class Simulation:
    def __init__(self, config):
        self.config = config
        # Generate Users------------------------------------------------------------------------------------------------
        self.base_station = BaseStation()
        user_counter = 0
        self.users = dict()
        for _ in range(config.num_users_per_job['Normal']):
            self.users[user_counter] = UserNormal(pos_base_station=self.base_station.pos, user_id=user_counter)
            user_counter += 1
        for _ in range(config.num_users_per_job['High Datarate']):
            self.users[user_counter] = UserHighDatarate(pos_base_station=self.base_station.pos, user_id=user_counter)
            user_counter += 1
        for _ in range(config.num_users_per_job['Low Latency']):
            self.users[user_counter] = UserLowLatency(pos_base_station=self.base_station.pos, user_id=user_counter)
            user_counter += 1
        for _ in range(config.num_users_per_job['Emergency Vehicle']):
            self.users[user_counter] = UserEmergencyVehicle(pos_base_station=self.base_station.pos, user_id=user_counter)
            user_counter += 1

        # Generate Initial Job Queue------------------------------------------------------------------------------------
        self.generate_jobs(chance=config.new_job_chance)

        # Statistics----------------------------------------------------------------------------------------------------
        self.total_units_requested: np.ndarray = np.array([])
        self.sum_capacity: np.ndarray = np.array([])
        self.jobs_lost: np.ndarray = np.array([])  # Number of jobs lost from violating latency constraints
        self.datarate_satisfaction: np.ndarray = np.array([])
        self.rewards: np.ndarray = np.array([])
        self.jobs_lost_EV_only: np.ndarray = np.array([])

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
        reward = 0 \
                 + self.config.lambda_reward['Sum Capacity'] * self.sum_capacity[-1] \
                 - self.config.lambda_reward['Packet Timeouts'] * self.jobs_lost[-1] \
                 + self.config.lambda_reward['Packet Rate'] * self.datarate_satisfaction[-1] \
                 - self.config.lambda_reward['EV Packet Timeouts'] * self.jobs_lost_EV_only[-1]
        # print('r', reward)

        return reward

    def step(self, allocation_vector):
        sum_capacity: float = 0
        jobs_lost: int = 0
        jobs_lost_EV_only: int = 0
        datarate_satisfaction: float = 0
        allocation_vector_copy = allocation_vector.copy()

        self.total_units_requested = self.get_len_job_list()

        for user in self.users.values():
            # Increment user runtime while they have a job in queue-----------------------------------------------------
            if user.jobs:
                user.runtime += 1

            # Allocate resources to jobs according to allocation solution, oldest jobs first----------------------------
            # Fetch job delays or ages per job for user
            job_delays = []
            for job in user.jobs:
                job_delays += [job.delay]

            # Allocate blocks of jobs, oldest jobs first
            while allocation_vector_copy[user.user_id] > 0:
                # Determine oldest job
                highest_delay_id = int(np.argmax(job_delays))

                # Send as many resources to this job as allowed, remove from pool
                units_sent = min(allocation_vector_copy[user.user_id], user.jobs[highest_delay_id].size)
                allocation_vector_copy[user.user_id] -= units_sent
                user.jobs[highest_delay_id].size -= units_sent

                # Check if job fully processed
                if user.jobs[highest_delay_id].size == 0:
                    user.jobs.pop(highest_delay_id)  # Remove from job list
                    job_delays.pop(highest_delay_id)  # Remove from delay ranking

                # Update users internal stats
                user.units_received += units_sent
                user.update_statistics()

            # Increment delay metric to remaining jobs, remove jobs over delay------------------------------------------
            for job in user.jobs:
                job.delay += 1
                if job.delay >= user.latency_max:
                    user.jobs.pop(user.jobs.index(job))
                    user.jobs_lost_to_timeout += 1
                    jobs_lost += 1  # Count latency constraint violation
                    user.update_statistics()

                    if user.type == 'EmergencyVehicle':
                        jobs_lost_EV_only += 1

            # Update sum_capacity metric--------------------------------------------------------------------------------
            sum_capacity += allocation_vector[user.user_id] * np.log10(
                1 + user.channel_quality * user.signal_noise_ratio)

            # Update datarate_satisfaction metric-----------------------------------------------------------------------
            datarate_satisfaction += min(user.datarate_satisfaction, 1)  # 1 for minimum datarate met, less else

            # Roll new channel according to position, fading------------------------------------------------------------
            user.update_channel_quality(pos_base_station=self.base_station.pos)

            # Move user to new position---------------------------------------------------------------------------------
            user.grid_move(step_size=1)

        self.sum_capacity = np.append(self.sum_capacity, sum_capacity)
        self.jobs_lost = np.append(self.jobs_lost, jobs_lost)
        self.datarate_satisfaction = np.append(self.datarate_satisfaction, datarate_satisfaction)
        self.jobs_lost_EV_only = np.append(self.jobs_lost_EV_only, jobs_lost_EV_only)
        self.rewards = np.append(self.rewards, self.build_reward())

    def reset(self):
        self.__init__(config=self.config)
