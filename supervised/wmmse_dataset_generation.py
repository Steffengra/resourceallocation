import numpy as np
import pickle, gzip

from supervised.imports.wmmse import wmmse


# [1] LOW COMPLEXITY WMMSE POWER ALLOCATION IN NOMA-FD SYSTEMS, F. Saggese
# [2] Learning to Optimize: Training Deep Neural Networks for Wireless Resource Management, H. Sun


def create_csi(num_channels, type):
    # Creates reciprocal num_channels x num_channels matrix h_csi, with coefficients distributed according
    # to type from {'rayleigh', 'gaussian'}
    if type == 'rayleigh':
        h_csi = (1 * np.random.randn(num_channels, num_channels) + 0) \
                + 1j * (1 * np.random.randn(num_channels, num_channels) + 0)
    elif type == 'gaussian':
        h_csi = 1 * np.random.randn(num_channels, num_channels) + 0
    else:
        print('Invalid channel type, valid types are: rayleigh, gaussian')
        return -1

    # Make reciprocal
    for ii in range(num_channels):
        for jj in range(ii + 1, num_channels):
            h_csi[ii, jj] = h_csi[jj, ii]
    return h_csi


def generate_dataset(num_channels, channel_type, P_max, N_iter, filename, compress):
    h_csi = []
    P_per_channel = []
    iterations_per_step = np.zeros(N_iter)
    print('Progress: 0.0 %', end='')
    for gen_loop_id in range(N_iter):
        # Generate data point
        h_csi.append(create_csi(num_channels, type=channel_type))
        p_alloc, iterations = wmmse(P_max=P_max, num_channels=num_channels,
                                    h_csi=h_csi[-1], sigma=np.ones(num_channels),
                                    alpha=np.ones(num_channels))
        # Save data point
        P_per_channel.append(p_alloc)
        iterations_per_step[gen_loop_id] = iterations

        # Print progress
        if np.mod(gen_loop_id, N_iter / 50) == 0:
            progress = np.round(gen_loop_id / (N_iter - 1) * 100, 1)
            print('\rProgress:', progress, '%', end='', flush=True)

    print('\rProgress: 100.0 %', flush=True)
    for ii in range(N_iter):
        h_csi[ii] = abs(h_csi[ii])

    dataset = [np.array(h_csi, dtype=np.float32),
               np.array(P_per_channel, dtype=np.float32)]

    # Save dataset
    if compress:
        f = gzip.open(filename + '.gstor', 'wb')
    else:
        f = open(filename + '.stor', 'wb')
    pickle.dump(dataset, f)
    f.close()

    # Save log data
    with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\supervised\\logs\\num_iterations_wmmse.gstor', 'wb') as file:
        pickle.dump(iterations_per_step, file)


num_channels = 10
channel_type = 'rayleigh'  # {'rayleigh', 'gaussian'}
P_max = 1  # Per Channel
N_iter = 5_000_000
name = 'wmmse'
filename = 'C:\\Py\\MasterThesis\\resourceallocation\\supervised\\datasets\\dataset_' + name + '_' + channel_type + '_C' + str(
    num_channels) + '_P' + str(P_max)
compress = True
generate_dataset(num_channels=num_channels, channel_type=channel_type, P_max=P_max, N_iter=N_iter, filename=filename,
                 compress=compress)
