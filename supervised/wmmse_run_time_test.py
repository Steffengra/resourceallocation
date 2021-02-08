import pickle, gzip
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from timeit import default_timer

from supervised.imports.wmmse import wmmse


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

# Testing-----------------------------------------------------------------------------------------------------------
P_max = 1
num_channels = 10
channel_type = 'rayleigh'  # {'rayleigh', 'gaussian'}
file_path = 'C:\\Py\\MasterThesis\\resourceallocation\\supervised\\datasets\\dataset_wmmse_'+channel_type+'_C'+str(num_channels)+'_P'+str(P_max)+'.gstor'
checkpoint_path = 'C:\\Py\\MasterThesis\\resourceallocation\\supervised\\SavedModels\\dnn1keras\\checkpoints\\model.ckpt'
validation_size, test_size = 0.1, 0.1  # relative

X_train, X_test, y_train, y_test = load_data(file_path, test_size)
model = tf.keras.models.load_model('SavedModels/dnn1keras')  # Load model

# Run time testing
print('Testing run times.. 0%', end='')
run_time_tests = 50
run_time_model = np.zeros(run_time_tests)
run_time_wmmse = np.zeros(run_time_tests)
for test_id in range(run_time_tests):  # more annoying than necessary but i want run time for every pass through
    start_time = default_timer()
    model(X_test[test_id][np.newaxis], training=False)
    end_time = default_timer()
    run_time_model[test_id] = end_time - start_time

    start_time = default_timer()
    wmmse(P_max=P_max, num_channels=num_channels, h_csi=X_test[test_id],
          sigma=np.ones(num_channels), alpha=np.ones(num_channels))
    end_time = default_timer()
    run_time_wmmse[test_id] = end_time - start_time
    if np.mod(test_id, 50) == 0:
        print('\rTesting run times.. ' + str(round(test_id/run_time_tests * 100, 1)) + '%', end='')
print('\rTesting run times.. done', flush=True)

print(np.mean(run_time_model))
print(np.mean(run_time_wmmse))

# Save log data
# with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\supervised\\logs\\run_time_model.gstor', 'wb') as file:
#     pickle.dump(run_time_model, file)
# with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\supervised\\logs\\run_time_wmmse.gstor', 'wb') as file:
#     pickle.dump(run_time_wmmse, file)
