import numpy as np

from reinforcement.imports.vehicle import Vehicle


class Simulation:
    def __init__(self, num_vehicles: int, vehicles_initial: tuple,
                 path_loss_exponent: float, p_max_per_vehicle: np.ndarray,
                 num_channels: int, max_timing_constraint: int, online_normalization_buffer):

        self.num_vehicles: int = num_vehicles
        self.vehicles_initial: tuple = vehicles_initial
        self.num_channels: int = num_channels
        self.path_loss_exponent: float = path_loss_exponent

        self.vehicles: list = []
        self.towers: list = []

        self.max_timing_constraint: int = max_timing_constraint
        self.time_constraints: list = list(  # [(max delay, remaining delay) per vehicle]
            zip(np.zeros(self.num_vehicles), np.zeros(self.num_vehicles)))

        # Generate initial positions, timings, csi---------------------------
        self.vehicles = [Vehicle(vehicles_initial[0], vehicles_initial[1]) for _ in range(self.num_vehicles)]
        self.towers = np.array([self.vehicles_initial[1] * np.random.randn(2) + self.vehicles_initial[0]
                                for _ in range(self.num_channels)])
        self.update_timings()
        self.csi = self.create_csi()
        self.p_max_per_vehicle = p_max_per_vehicle

        # Update normalization buffer---------------------------
        online_normalization_buffer.channel_state_information.add_values(np.concatenate(self.csi))

        # Generate initial states--------------------------------------------
        self.rewards = np.zeros(num_vehicles)
        self.states = []
        for vehicle_id in range(num_vehicles):
            i_per_channel_per_vehicle = np.zeros(self.num_channels)
            n_per_channel_per_vehicle = np.zeros(self.num_channels)

            state_flattened = []
            # state_flattened.append(p_max_per_vehicle[vehicle_id])
            state_flattened.append(
                (10 * np.log10(self.p_max_per_vehicle[vehicle_id]) - online_normalization_buffer.power.min) / (
                        online_normalization_buffer.power.max - online_normalization_buffer.power.min))
            # state_flattened.extend(i_per_channel_per_vehicle)
            # state_flattened.extend(n_per_channel_per_vehicle)
            # state_flattened.extend(self.csi[vehicle_id])
            state_flattened.extend(
                (np.log10(self.csi[vehicle_id]) - online_normalization_buffer.channel_state_information.buffer_min_no_outliers_log) / (
                    online_normalization_buffer.channel_state_information.buffer_max_no_outliers_log - online_normalization_buffer.channel_state_information.buffer_min_no_outliers_log))

            self.states.append(state_flattened)

    def create_csi(self):
        csi = []
        for vehicle_id in range(self.num_vehicles):
            # Rayleigh-fading---------------
            csi_current_vehicle = abs((1 * np.random.randn(self.num_channels) + 0)
                                      + 1j * (1 * np.random.randn(self.num_channels) + 0))

            # Path loss---------------------
            distance_to_towers = np.sqrt(np.sum(np.square(self.vehicles[vehicle_id].position - self.towers), axis=1))
            csi_current_vehicle = csi_current_vehicle / np.maximum(
                np.power(distance_to_towers, self.path_loss_exponent), np.ones(self.num_channels))  # cap path loss at 1 to avoid path gain

            csi.append(csi_current_vehicle)

        return csi

    def update_timings(self):
        # Roll random time constraints for vehicles that have none, ---------
        # update others------------------------------------------------------
        for vehicle_id in range(self.num_vehicles):
            if self.time_constraints[vehicle_id][1] <= 0:
                t_lim = np.random.randint(low=1, high=self.max_timing_constraint + 1)
                t_rem = t_lim
            else:
                t_lim = self.time_constraints[vehicle_id][0]
                t_rem = self.time_constraints[vehicle_id][1] - 1
            self.time_constraints[vehicle_id] = (t_lim, t_rem)

    def step(self, vehicles_step: float,
             p_max_per_vehicle: np.ndarray, sigma_per_vehicle: np.ndarray,
             actions_channel: list, actions_usage: list,
             lambda_reward: tuple, update_p_max_per_vehicle: bool, online_normalization_buffer):

        # Move vehicles in a grid fashion------------------------------------
        for vehicle in self.vehicles:
            vehicle.grid_move(step_size=vehicles_step)

        # Calculate new state------------------------------------------------
        n_per_channel = np.zeros(self.num_channels)
        p_per_channel = np.zeros(self.num_channels)
        sinr = []
        for vehicle_id in range(self.num_vehicles):
            # Calculate usage per channel-----------------------
            n_per_channel[actions_channel[vehicle_id]] += 1

            # Calculate power per channel-----------------------
            p_per_channel[actions_channel[vehicle_id]] += \
                actions_usage[vehicle_id] * p_max_per_vehicle[vehicle_id] * self.csi[vehicle_id][actions_channel[vehicle_id]]

            # Calculate capacity per vehicle--------------------
            sumterm = sigma_per_vehicle[vehicle_id] ** 2
            for vehicle_id2 in range(self.num_vehicles):
                if vehicle_id != vehicle_id2 and actions_channel[vehicle_id] == actions_channel[vehicle_id2]:
                    sumterm += actions_usage[vehicle_id2] * p_max_per_vehicle[vehicle_id2] * self.csi[vehicle_id2][actions_channel[vehicle_id2]]
            sinr.append(actions_usage[vehicle_id] * p_max_per_vehicle[vehicle_id] * self.csi[vehicle_id][actions_channel[vehicle_id]] / sumterm)
        capacity = 1 * np.log10(1 + np.array(sinr))

        # Calculate interference per channel--------------------
        i_per_channel_per_vehicle = []
        n_per_channel_per_vehicle = []
        for vehicle_id in range(self.num_vehicles):
            # i_per_channel is p_per_channel minus own contribution
            p_per_channel_adjusted = p_per_channel.copy()
            p_per_channel_adjusted[actions_channel[vehicle_id]] -= \
                actions_usage[vehicle_id] * p_max_per_vehicle[vehicle_id] * self.csi[vehicle_id][actions_channel[vehicle_id]]
            i_per_channel_per_vehicle.append(p_per_channel_adjusted)

            n_per_channel_adjusted = n_per_channel.copy()
            n_per_channel_adjusted[actions_channel[vehicle_id]] -= 1
            n_per_channel_per_vehicle.append(n_per_channel_adjusted)

        # Create new csi per vehicle----------------------------
        self.csi = self.create_csi()

        # Create new p_max_per_vehicle--------------------------
        if update_p_max_per_vehicle:
            snr = np.random.randint(online_normalization_buffer.power.min, online_normalization_buffer.power.max + 1)
            self.p_max_per_vehicle = np.float_power(10, snr / 10) * np.ones(self.num_vehicles)

        # Update normalization buffer---------------------------
        online_normalization_buffer.channel_state_information.add_values(np.concatenate(self.csi))

        # Assemble new state information per vehicle------------
        self.states = []
        for vehicle_id in range(self.num_vehicles):
            self.rewards[vehicle_id] = lambda_reward[0] * sum(capacity) - lambda_reward[1] * (
                    self.time_constraints[vehicle_id][0] - self.time_constraints[vehicle_id][1])

            state_flattened = []
            # state_flattened.append(p_max_per_vehicle[vehicle_id])
            state_flattened.append(
                (10 * np.log10(self.p_max_per_vehicle[vehicle_id]) - online_normalization_buffer.power.min) / (
                        online_normalization_buffer.power.max - online_normalization_buffer.power.min))
            # state_flattened.extend(i_per_channel_per_vehicle)
            # state_flattened.extend(n_per_channel_per_vehicle)
            # state_flattened.extend(self.csi[vehicle_id])
            state_flattened.extend(
                (np.log10(self.csi[vehicle_id]) - online_normalization_buffer.channel_state_information.buffer_min_no_outliers_log) / (
                        online_normalization_buffer.channel_state_information.buffer_max_no_outliers_log - online_normalization_buffer.channel_state_information.buffer_min_no_outliers_log))

            self.states.append(state_flattened)
        # Update delay constraints-------------------------------------------
        self.update_timings()

    def reset(self, p_max_per_vehicle, online_normalization_buffer):
        self.__init__(num_vehicles=self.num_vehicles,
                      vehicles_initial=self.vehicles_initial,
                      path_loss_exponent=self.path_loss_exponent,
                      p_max_per_vehicle=p_max_per_vehicle,
                      num_channels=self.num_channels,
                      max_timing_constraint=self.max_timing_constraint,
                      online_normalization_buffer=online_normalization_buffer)
