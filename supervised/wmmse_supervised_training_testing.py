
import gzip
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from timeit import default_timer

from supervised.imports.wmmse import wmmse

# Load Data
# [1] Efficient BackProp, LeCun
P_max = 1
num_channels = 10
channel_type = 'rayleigh'  # {'rayleigh', 'gaussian'}
file_path = Path(Path(__file__), 'datasets', f'dataset_wmmse_{channel_type}_C{num_channels}_P{P_max}.gstor')
checkpoint_path = Path(Path(__file__), 'SavedModels', 'dnn1keras', 'checkpoints', 'model.ckpt')
validation_size, test_size = 0.1, 0.1  # relative

num_episodes: int = 300


def relu_clamp(x):
    return tf.keras.backend.relu(x, max_value=P_max)


def load_data(path, test_size):
    if path.split('.')[-1] == 'gstor':
        with gzip.open(path, 'rb') as f:
            Data = pickle.load(f)
    else:
        with open(path, 'rb') as f:
            Data = pickle.load(f)
    X = Data[0][:, :, :]  # Scale to 0.. 1? [1]
    y = Data[1][:, :]

    return train_test_split(X, y, test_size=test_size)


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


# Training--------------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = load_data(file_path, test_size)

with tf.device('/GPU:0'):
    loss_fn = tf.keras.losses.mean_squared_error
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=.001, rho=0.9)
    callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          # min_delta=1e-9,
                                                          patience=5,
                                                          verbose=1)
    callback_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=X_train[0].shape),
        tf.keras.layers.Dense(200, activation='relu', kernel_initializer='glorot_normal'),
        # Truncation of glorot-normal is unclear, presumably 2*stddev = 2
        tf.keras.layers.Dense(200, activation='relu', kernel_initializer='glorot_normal'),
        tf.keras.layers.Dense(200, activation='relu', kernel_initializer='glorot_normal'),
        tf.keras.layers.Dense(10, activation=relu_clamp, kernel_initializer='glorot_normal')
    ])
    model.summary()

    model.compile(optimizer=optimizer, loss=loss_fn)
    model.fit(X_train, y_train,
              callbacks=[
                  callback_earlystop,
                  # callback_checkpoint,
              ],
              validation_split=validation_size / (1 - test_size),  # .9 * x = .1 of entire data set
              validation_freq=1,
              batch_size=1000,
              epochs=num_episodes,
              verbose=2)
    model.save('SavedModels/dnn1keras/')

    # Testing-----------------------------------------------------------------------------------------------------------
    y_pred = model.predict(X_test)  # batch testing

    # Run time testing
    # print('Testing run times.. 0%', end='')
    # run_time_tests = 100_000
    # run_time_model = np.zeros(run_time_tests)
    # run_time_wmmse = np.zeros(run_time_tests)
    # for test_id in range(run_time_tests):  # more annoying than necessary but i want run time for every pass through
    #     start_time = default_timer()
    #     model(X_test[test_id][np.newaxis], training=False)
    #     end_time = default_timer()
    #     run_time_model[test_id] = end_time - start_time
    #
    #     start_time = default_timer()
    #     wmmse(P_max=P_max, num_channels=num_channels, h_csi=X_test[test_id],
    #           sigma=np.ones(num_channels), alpha=np.ones(num_channels))
    #     end_time = default_timer()
    #     run_time_wmmse[test_id] = end_time - start_time
    #     if np.mod(test_id, 50) == 0:
    #         print('\rTesting run times.. ' + str(round(test_id/run_time_tests * 100, 1)) + '%', end='')
    # print('\rTesting run times.. done', flush=True)

    # Save log data
    # with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\supervised\\logs\\run_time_model.gstor', 'wb') as file:
    #     pickle.dump(run_time_model, file)
    # with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\supervised\\logs\\run_time_wmmse.gstor', 'wb') as file:
    #     pickle.dump(run_time_wmmse, file)

    # model = tf.keras.models.load_model('saved_model/my_model') # Load model

print('Calculating throughputs.. 0%', end='')
Throughput_wmmse, Throughput_dnn, Throughput_dnn_postprocessed, Throughput_Pmax, Throughput_random = [], [], [], [], []
for ii in range(len(y_pred)):
    Throughput_wmmse.append(
        calculate_throughput(X_test[ii], y_test[ii], sigma=np.ones(num_channels)))
    Throughput_dnn.append(
        calculate_throughput(X_test[ii], y_pred[ii], sigma=np.ones(num_channels)))
    Throughput_dnn_postprocessed.append(
            calculate_throughput(X_test[ii], post_processor(y_pred[ii], p_max=P_max), sigma=np.ones(num_channels)))
    Throughput_Pmax.append(
        calculate_throughput(X_test[ii], P_max * np.ones(num_channels), sigma=np.ones(num_channels)))
    Throughput_random.append(
        calculate_throughput(X_test[ii], P_max * np.random.rand(num_channels), sigma=np.ones(num_channels)))
    if np.mod(ii, int(len(y_pred)/20)) == 0:
        print('\rCalculating throughputs.. '+str(round(ii/len(y_pred)*100, 1))+'%', end='')
print('\rCalculating throughputs.. done')

# Log throughput
with gzip.open(Path(Path(__file__), 'logs', 'throughput_model.gstor'), 'wb') as file:
    pickle.dump(Throughput_dnn, file)
with gzip.open(Path(Path(__file__), 'logs', 'throughput_model_postprocessed.gstor'), 'wb') as file:
    pickle.dump(Throughput_dnn_postprocessed, file)
with gzip.open(Path(Path(__file__), 'logs', 'throughput_wmmse.gstor'), 'wb') as file:
    pickle.dump(Throughput_wmmse, file)
with gzip.open(Path(Path(__file__), 'logs', 'throughput_pmax.gstor'), 'wb') as file:
    pickle.dump(Throughput_Pmax, file)
with gzip.open(Path(Path(__file__), 'logs', 'throguhput_random.gstor'), 'wb') as file:
    pickle.dump(Throughput_random, file)

# Plotting---------------------------------------------------------------------------------
import matplotlib.pyplot as plt

plt.style.use('default')
color1 = '#21467a'
color2 = '#c4263aff'
color3 = '#008700'
color4 = '#660078'

dnn_amounts, dnn_bin_edges = np.histogram(Throughput_dnn, bins=40)
dnn_cumulative = np.cumsum(dnn_amounts) / len(y_pred)
dnn_postprocessed_amounts, dnn_postprocessed_bin_edges = np.histogram(Throughput_dnn, bins=40)
dnn_postprocessed_cumulative = np.cumsum(dnn_postprocessed_amounts) / len(y_pred)
wmmse_amounts, wmmse_bin_edges = np.histogram(Throughput_wmmse, bins=40)
wmmse_cumulative = np.cumsum(wmmse_amounts) / len(y_pred)
Pmax_amounts, Pmax_bin_edges = np.histogram(Throughput_Pmax, bins=40)
Pmax_cumulative = np.cumsum(Pmax_amounts) / len(y_pred)
random_amounts, random_bin_edges = np.histogram(Throughput_random, bins=40)
random_cumulative = np.cumsum(random_amounts) / len(y_pred)

f = plt.figure()
plt.plot(dnn_bin_edges[:-1], dnn_cumulative, c=color1, figure=f)
plt.plot(wmmse_bin_edges[:-1], wmmse_cumulative, c=color2, figure=f)
plt.plot(Pmax_bin_edges[:-1], Pmax_cumulative, c=color3, figure=f)
plt.plot(random_bin_edges[:-1], random_cumulative, c=color4, figure=f)
plt.xlabel('Throughput')
plt.ylabel('Cumulative Probability')
plt.legend(['DNN', 'WMMSE', 'Max Power', 'Random Power'], loc='lower right')
plt.grid(ls='-', alpha=0.25)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
# plt.savefig(
#     'plots/wmmse1_'+channel_type+'_C'+str(num_channels)+'_P'+str(P_max)+'_cumprob.pdf', bbox_inches='tight', dpi=800)
plt.show()
