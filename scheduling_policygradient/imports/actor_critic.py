import tensorflow as tf
import numpy as np

from scheduling_policygradient.imports.experience_buffer import ExperienceBuffer


class QModel(tf.keras.Model):  # the critic
    def __init__(self, input_shape, num_hidden):  # input shape is state+action
        super(QModel, self).__init__()
        self.hidden_layers = []
        for size in num_hidden:
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output


class ActorModel(tf.keras.Model):  # the actor
    def __init__(self, input_shape, num_hidden, output_shape):  # input shape is state
        super(ActorModel, self).__init__()
        self.hidden_layers = []
        for size in num_hidden:
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(output_shape, activation='softmax')

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output


class ActorCritic:
    def __init__(self, size_state, size_action, num_hidden,
                 learning_rate_q, learning_rate_a, batch_size, min_experiences, max_experiences, min_priority,
                 prioritization_factors: list, gamma, loss_function):
        self.size_state = size_state
        self.size_action = size_action

        self.batch_size: int = batch_size
        self.min_experiences: int = min_experiences
        self.prioritization_factors: list = prioritization_factors
        self.gamma: float = gamma
        self.c_optimizer = tf.optimizers.Adam(lr=learning_rate_q)
        self.a_optimizer = tf.optimizers.Adam(lr=learning_rate_a)

        if loss_function == 'mse':
            self.loss = self.mse_loss
        elif loss_function == 'huber':
            self.loss = self.huber_loss
        else:
            print('Invalid Loss Function')
            quit()

        self.q_primary = QModel(input_shape=size_state+size_action, num_hidden=num_hidden)
        # self.q_target = QModel(input_shape=size_state+size_action, num_hidden=num_hidden)

        self.actor_primary = ActorModel(input_shape=size_state, num_hidden=num_hidden, output_shape=size_action)
        # self.actor_target = ActorModel(input_shape=size_state, num_hidden=num_hidden, output_shape=size_action)

        # self.copy_weights()

        self.experience_buffer = ExperienceBuffer(max_experiences=max_experiences,
                                                  min_priority=min_priority,
                                                  alpha=self.prioritization_factors[0])

        self.gradient_magnitude: np.ndarray = np.array([], dtype=int)  # Performance metric

    def get_q(self, state, action):
        if np.ndim(state) == 1:
            state = np.array(state)[np.newaxis]
            action = np.array(action)[np.newaxis]
        input_vector = np.concatenate([state, action], axis=1)
        input_vector = np.array(input_vector, dtype='float32')

        return self.q_primary(input_vector)

    def get_action(self, state):
        if np.ndim(state) == 1:
            state = np.array(state, dtype='float32')[np.newaxis]
        else:
            state = np.array(state, dtype='float32')

        return self.actor_primary(state)

    def add_experience(self, state, action, reward, following_state):
        self.experience_buffer.add_experience(state, action, reward, following_state)

    def copy_weights(self):
        variables1 = self.q_target.trainable_variables
        variables2 = self.q_primary.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

        variables1 = self.actor_target.trainable_variables
        variables2 = self.actor_primary.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    @staticmethod
    def mse_loss(td_error, importance_sampling_weights):
        squared_error = importance_sampling_weights * tf.square(td_error)

        return tf.math.reduce_mean(squared_error)

    @staticmethod
    def huber_loss(td_error, importance_sampling_weights):  # According to wiki/huber_loss
        delta = 1  # Where to clip
        absolute_error = delta * (tf.abs(td_error) - .5 * delta)
        squared_error = .5 * tf.square(td_error)

        indicator = absolute_error < delta
        huber_loss = tf.where(indicator, squared_error, absolute_error)

        return tf.math.reduce_mean(importance_sampling_weights * huber_loss)

    def train(self):
        if len(self.experience_buffer.state) < self.min_experiences:
            return 0  # Don't train before enough experiences

        # Select batch_size experiences randomly from memory----
        experience_ids, states, actions, rewards, following_states, importance_sampling_weights = \
            self.experience_buffer.sample(batch_size=self.batch_size, beta=self.prioritization_factors[1])

        # Calculate Loss critic---------------------------------
        input_vector = np.concatenate([states, actions], axis=1)
        input_vector = np.array(input_vector, dtype='float32')

        # next_actions = self.actor_target(following_states)
        # next_input = np.array(np.concatenate([following_states, next_actions], axis=1), dtype='float32')
        # future_rewards_estimate = tf.squeeze(self.q_target(next_input))

        c_parameters = self.q_primary.trainable_variables
        with tf.GradientTape() as tape:
            r_predicted = tf.squeeze(self.q_primary(input_vector))

            # td_error = (rewards + self.gamma * future_rewards_estimate) - r_predicted
            td_error = rewards - r_predicted
            c_loss = self.loss(td_error=td_error, importance_sampling_weights=importance_sampling_weights)

        c_gradients = tape.gradient(c_loss, c_parameters)  # d_loss / d_parameters

        # Calculate loss actor----------------------------------
        a_parameters = self.actor_primary.trainable_variables
        with tf.GradientTape() as tape:
            actor_action = self.actor_primary(states)
            input_vector = tf.concat([states, actor_action], axis=1)
            critic_critique = tf.squeeze(self.q_primary(input_vector))

            a_loss = -tf.math.reduce_mean(critic_critique)

        a_gradients = tape.gradient(a_loss, a_parameters)

        # Add priority to experience buffer---------------------
        self.experience_buffer.adjust_priority(ids=experience_ids, new_priority=td_error.numpy())

        # Check for exploding gradients-------------------------
        self.gradient_magnitude = np.append(
            self.gradient_magnitude, int(tf.reduce_sum([tf.reduce_sum(gradient**2) for gradient in c_gradients])**.5))

        # Apply Gradient Update---------------------------------
        self.c_optimizer.apply_gradients(zip(c_gradients, c_parameters))
        self.a_optimizer.apply_gradients(zip(a_gradients, a_parameters))
