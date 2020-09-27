import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import gzip, pickle


num_tests = 100_000


def main():
    q_primary = tf.keras.models.load_model('SavedModels\\reinforcement_1\\q_primary')
    csis = np.zeros(num_tests)
    choices = np.zeros(num_tests)

    print('Testing.. 0.0 %', end='')
    for test_id in range(num_tests):
        csi = abs((1 * np.random.randn(1) + 0) + 1j * (1 * np.random.randn(1) + 0))
        choice = np.argmax(q_primary(np.array([20, csi], dtype='float32')[np.newaxis]))

        csis[test_id] = csi
        choices[test_id] = choice
        if np.mod(test_id, 21) == 0:
            progress = np.round(test_id/num_tests*100, 1)
            print('\rTesting..', progress, '%', end='')
    print('\rTesting.. done', flush=True)

    plt.scatter(csis, choices)
    plt.show()

    with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\testing_policy_csis.gstor',
                   'wb') as file:
        pickle.dump(csis, file)
    with gzip.open('C:\\Py\\MasterThesis\\resourceallocation\\reinforcement\\logs\\testing_policy_choices.gstor',
                   'wb') as file:
        pickle.dump(choices, file)


if __name__ == '__main__':
    main()
