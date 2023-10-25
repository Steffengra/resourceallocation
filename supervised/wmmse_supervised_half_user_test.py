
import gzip
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf

from supervised.imports.wmmse import wmmse


P_max = 1
num_channels = 10
num_data_points = 100_000


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


def relu_clamp(x):
    return tf.keras.backend.relu(x, max_value=P_max)


def calculate_throughput(abs_h_csi, P_alloc, sigma):
    alpha = np.ones(len(P_alloc))

    abs_h_csi_sq = abs_h_csi ** 2
    throughput = 0
    for kk in range(len(P_alloc)):
        sumterm = sigma[kk] ** 2
        for jj in range(len(P_alloc)):
            if jj != kk: sumterm += abs_h_csi_sq[kk, jj] * P_alloc[jj]
        throughput += alpha[kk] * np.log(1 + abs_h_csi_sq[kk, kk] * P_alloc[kk] / sumterm)

    return throughput


def post_processor(p_alloc, p_max):
    p_alloc_processed = p_alloc.copy()
    p_alloc_processed = p_alloc_processed / p_max
    for p_id in range(len(p_alloc)):
        if p_alloc_processed[p_id] > 0.5:
            p_alloc_processed[p_id] = 1
        else:
            p_alloc_processed[p_id] = 0
    p_alloc_processed = p_alloc_processed * p_max

    return p_alloc_processed


model = tf.keras.models.load_model(Path(Path(__file__), 'SavedModels', 'dnn1keras'))  # Load model
# Generate data set-----------------------------------------------------------------------------------------------------
csis = []
csis_padded = []
for csi_id in range(num_data_points):
    csi_current = np.array(abs(create_csi(num_channels=int(num_channels / 2), type='rayleigh')), dtype='float32')
    csi_current_padded = np.zeros((num_channels, num_channels), dtype='float32')
    csi_current_padded[0:int(num_channels / 2), 0:int(num_channels / 2)] = csi_current

    csis.append(csi_current)
    csis_padded.append(csi_current_padded)

y_pred = model.predict(np.array(csis_padded))

print('Calculating throughputs.. 0%', end='')
Throughput_wmmse, Throughput_dnn, Throughput_dnn_postprocessed = [], [], []
for ii in range(len(y_pred)):
    p_wmmse, _ = wmmse(P_max,
                       int(num_channels / 2),
                       csis[ii],
                       np.ones(int(num_channels / 2)),
                       np.ones(int(num_channels / 2)))
    Throughput_wmmse.append(
        calculate_throughput(csis[ii], p_wmmse, sigma=np.ones(int(num_channels / 2))))
    Throughput_dnn.append(
        calculate_throughput(csis_padded[ii], y_pred[ii], sigma=np.ones(num_channels)))
    Throughput_dnn_postprocessed.append(
        calculate_throughput(csis_padded[ii], post_processor(y_pred[ii], p_max=P_max), sigma=np.ones(num_channels)))
    if np.mod(ii, int(len(y_pred) / 20)) == 0:
        print('\rCalculating throughputs.. ' + str(round(ii / len(y_pred) * 100, 1)) + '%', end='')
print('\rCalculating throughputs.. done')

# Log throughput
with gzip.open(Path(Path(__file__), 'logs', 'throughput_model_half_user.gstor'), 'wb') as file:
    pickle.dump(Throughput_dnn, file)
with gzip.open(Path(Path(__file__), 'logs', 'throughput_model_postprocessed_half_user.gstor'), 'wb') as file:
    pickle.dump(Throughput_dnn_postprocessed, file)
with gzip.open(Path(Path(__file__), 'logs', 'throughput_wmmse_half_user.gstor'), 'wb') as file:
    pickle.dump(Throughput_wmmse, file)
