import numpy as np


# WMMSE optimizes Throughput via power allocation
# SINR = |h_kk|^2 * p_k / (sum_j<>k |h_jk|^2 * p_j + sigma_k^2)
# -> max throughput:
#    max_p_i sum_k alpha_k log( 1 + sinr_k )
#    s.t. 0 <= p_k <= P_max
# -> wmmse transforms problem into different space to solve
def wmmse(P_max, num_channels, h_csi, sigma, alpha):
    abs_h_csi = abs(h_csi)
    abs_h_csi_sq = abs_h_csi ** 2

    # Initialize--------------------------------------------------------------------------------------------------------
    # Initialize v_k <= sqrt(P_max)
    sqrt_p = np.sqrt(P_max) * np.ones(num_channels)  # v

    # Initialize u_k
    sqrt_sinr = np.zeros(num_channels)  # u
    for vehicle_id_sinr in range(num_channels):
        sumterm = sigma[vehicle_id_sinr] ** 2
        for vehicle_id_itfrnc in range(num_channels):
            sumterm += abs_h_csi_sq[vehicle_id_itfrnc, vehicle_id_sinr] * sqrt_p[vehicle_id_itfrnc]**2
        sqrt_sinr[vehicle_id_sinr] = abs_h_csi[vehicle_id_sinr, vehicle_id_sinr] * sqrt_p[vehicle_id_sinr] / sumterm

    # Initialize w_k
    inv_sinr = np.zeros(num_channels)  # w
    for vehicle_id_sinr in range(num_channels):
        inv_sinr[vehicle_id_sinr] = 1 / (
                1 - sqrt_sinr[vehicle_id_sinr] * abs_h_csi[vehicle_id_sinr, vehicle_id_sinr] * sqrt_p[vehicle_id_sinr])

    # Initialize objective function at p_k = P_max for all k
    p = P_max * np.ones(num_channels)
    sinr = np.zeros(num_channels)
    for vehicle_id_sinr in range(num_channels):
        sumterm = sigma[vehicle_id_sinr]**2
        for vehicle_id_itfrnc in range(num_channels):
            if vehicle_id_itfrnc != vehicle_id_sinr:
                sumterm += abs_h_csi_sq[vehicle_id_itfrnc, vehicle_id_sinr] * p[vehicle_id_itfrnc]
        sinr[vehicle_id_sinr] = abs_h_csi_sq[vehicle_id_sinr, vehicle_id_sinr] * p[vehicle_id_sinr] / sumterm

    capacity = alpha * np.log10(1 + sinr)
    objective_new = sum(capacity)

    # Iterate-----------------------------------------------------------------------------------------------------------
    t = 0
    error = 1
    while t < 500 and error > 1e-5:
        t += 1
        objective_old = objective_new

        # Update v_k
        for vehicle_id_sinr in range(num_channels):
            sumterm = 0
            for vehicle_id_itfrnc in range(num_channels):
                sumterm += alpha[vehicle_id_itfrnc] * inv_sinr[vehicle_id_itfrnc] * \
                           sqrt_sinr[vehicle_id_itfrnc]**2 * abs_h_csi_sq[vehicle_id_itfrnc, vehicle_id_sinr]
            sqrt_p[vehicle_id_sinr] = alpha[vehicle_id_sinr] * inv_sinr[vehicle_id_sinr] * \
                                      sqrt_sinr[vehicle_id_sinr] * abs_h_csi[vehicle_id_sinr, vehicle_id_sinr] / sumterm
            # case: Cut off at 0..sqrt(P_max) to ensure constraint, [2]
            sqrt_p[vehicle_id_sinr] = min(max(0, sqrt_p[vehicle_id_sinr]), np.sqrt(P_max))

        # Update u_k
        for vehicle_id_sinr in range(num_channels):
            sumterm = sigma[vehicle_id_sinr]**2
            for vehicle_id_itfrnc in range(num_channels):
                sumterm += abs_h_csi_sq[vehicle_id_itfrnc, vehicle_id_sinr] * sqrt_p[vehicle_id_itfrnc]**2
            sqrt_sinr[vehicle_id_sinr] = abs_h_csi[vehicle_id_sinr, vehicle_id_sinr] * sqrt_p[vehicle_id_sinr] / sumterm

        # Update w_k
        for vehicle_id_sinr in range(num_channels):
            inv_sinr[vehicle_id_sinr] = 1 / (1 - sqrt_sinr[vehicle_id_sinr] * abs_h_csi[vehicle_id_sinr, vehicle_id_sinr] * sqrt_p[vehicle_id_sinr])

        # Calculate iterated powers, test objective function
        p = np.round(sqrt_p**2, 7)
        sinr = np.zeros(num_channels)
        for vehicle_id_sinr in range(num_channels):
            sumterm = sigma[vehicle_id_sinr] ** 2
            for vehicle_id_itfrnc in range(num_channels):
                if vehicle_id_itfrnc != vehicle_id_sinr:
                    sumterm += abs_h_csi_sq[vehicle_id_itfrnc, vehicle_id_sinr] * p[vehicle_id_itfrnc]
            sinr[vehicle_id_sinr] = abs_h_csi_sq[vehicle_id_sinr, vehicle_id_sinr] * p[vehicle_id_sinr] / sumterm

        capacity = alpha * np.log10(1 + sinr)
        objective_new = sum(capacity)
        error = objective_new - objective_old

    return p, t
