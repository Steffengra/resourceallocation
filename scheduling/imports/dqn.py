import tensorflow as tf
import numpy as np

from scheduling.imports.experience_buffer import ExperienceBuffer


class Model(tf.keras.Model):
    def __init__(self, state_shape, num_hidden, num_actions):
        super(Model, self).__init__()
        self.input_layer = tf.keras.layers.Flatten(input_shape=(1, state_shape))
        self.hidden_layers = []
        for size in num_hidden:
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation='relu'))
        self.output_layer = tf.keras.layers.Dense(num_actions)

    @tf.function
    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output


class ModelDueling(tf.keras.Model):
    # DuelingDQN separates the NN output into Value and Action calculation, then recombines them to form Q
    # [1] Dueling Network Architectures for Deep Reinforcement Learning, wang, 2016
    def __init__(self, state_shape, num_hidden, num_dueling, num_actions):
        super(ModelDueling, self).__init__()
        self.input_layer = tf.keras.layers.Flatten(input_shape=(1, state_shape))
        self.hidden_layers = []
        for size in num_hidden:
            self.hidden_layers.append(tf.keras.layers.Dense(size, activation='relu'))

        self.advantage_layers = []
        self.value_layers = []
        for size in num_dueling:
            self.advantage_layers.append(tf.keras.layers.Dense(size, activation='relu'))
            self.value_layers.append(tf.keras.layers.Dense(size, activation='relu'))

        self.advantage_layers.append(tf.keras.layers.Dense(num_actions))  # Output layers
        self.value_layers.append(tf.keras.layers.Dense(1))

    @tf.function
    def call(self, inputs):
        # Joint layers--------
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        # Advantage Stream----
        a = tf.identity(x)  # Copy of x
        for layer in self.advantage_layers:
            a = layer(a)
        # Value Stream.-------
        v = tf.identity(x)  # Copy of x
        for layer in self.value_layers:
            v = layer(v)

        return v + (a - tf.reduce_mean(a))


class DQN:
    # A DQN contains two neural nets, q_primary and q_target
    def __init__(self, num_algorithms, size_state, num_hidden, rainbow, num_dueling,
                 learning_rate, batch_size, min_experiences, max_experiences, min_priority,
                 prioritization_factors: list, gamma, loss_function):
        self.num_hidden: list = num_hidden
        self.rainbow: dict = rainbow
        self.num_dueling: list = num_dueling
        self.num_algorithms = num_algorithms
        self.size_state: int = size_state

        self.batch_size: int = batch_size
        self.min_experiences: int = min_experiences
        self.prioritization_factors: list = prioritization_factors
        self.gamma: float = gamma
        self.optimizer = tf.optimizers.Adam(lr=learning_rate)

        if loss_function == 'mse':
            self.loss = self.mse_loss
        elif loss_function == 'huber':
            self.loss = self.huber_loss
        else:
            print('Invalid Loss Function')
            quit()

        # q_primary is the primary network used for making predictions. q_target is used for training stability.
        if not self.rainbow['DuelingDQN']:  # DuelingDQN Yes/No
            self.q_primary = Model(state_shape=self.size_state,
                                   num_hidden=self.num_hidden, num_actions=self.num_algorithms)
            self.q_target = Model(state_shape=self.size_state,
                                  num_hidden=self.num_hidden, num_actions=self.num_algorithms)
        else:
            self.q_primary = ModelDueling(state_shape=self.size_state,
                                          num_hidden=self.num_hidden, num_dueling=self.num_dueling,
                                          num_actions=self.num_algorithms)
            self.q_target = ModelDueling(state_shape=self.size_state,
                                         num_hidden=self.num_hidden, num_dueling=self.num_dueling,
                                         num_actions=self.num_algorithms)
        self.copy_weights()

        self.experience_buffer = ExperienceBuffer(max_experiences=max_experiences,
                                                  min_priority=min_priority,
                                                  alpha=self.prioritization_factors[0])

        self.gradient_magnitude: np.ndarray = np.array([], dtype=int)  # Performance metric

    def get_action(self, state, epsilon):
        # Get a random action at chance epsilon, or the DQN predicted action
        if np.random.rand() < epsilon:
            return np.random.choice(self.num_algorithms)
        else:
            return np.argmax(self.q_primary(np.asarray(state, dtype='float32')[np.newaxis]))

    def add_experience(self, state, action, reward, following_state):
        self.experience_buffer.add_experience(state, action, reward, following_state)

    def copy_weights(self):
        variables1 = self.q_target.trainable_variables
        variables2 = self.q_primary.trainable_variables
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

    def reduce_learning_rate(self, lr_decay: float, lr_min: float):
        old_lr = self.optimizer.lr.read_value()
        new_lr = max(lr_min, old_lr - lr_decay)  # cap at lr_min
        self.optimizer.lr.assign(new_lr)
        # print('Reduced learning rate to ' + str(self.optimizer.lr.read_value().numpy()))

    def train(self):
        if len(self.experience_buffer.state) < self.min_experiences:
            return 0  # Don't train before enough experiences

        # Select batch_size experiences randomly from memory----
        experience_ids, states, actions, rewards, following_states, importance_sampling_weights = \
            self.experience_buffer.sample(batch_size=self.batch_size, beta=self.prioritization_factors[1])

        # Calculate Loss----------------------------------------
        # L = ((r + gamma * max(Q(s+)) - Q_primary(s, a))^2
        #   = (Q(s,a) - f(s, params))^2
        # gamma is discount rate, r is rewards
        # Semi-gradient, using Q_target for stability.
        # Optimize by deriving toward params -> optimizer
        if not self.rainbow['Double-Q']:  # Double-Q Learning, "Corrects for overestimation", rainbow
            q_target_of_s_plus = np.max(self.q_target(following_states), axis=1)  # max(Q_target(s+))
        else:
            action_ids = np.argmax(self.q_primary(following_states), axis=1)
            q_target_of_s_plus = self.q_target(following_states).numpy()[range(len(action_ids)), [action_ids]][0]
        q_target_of_s = rewards + self.gamma * q_target_of_s_plus  # r + gamma * max(Q_target(s+))

        with tf.GradientTape() as tape:
            q_primary_of_s = tf.math.reduce_sum(
                self.q_primary(states) * tf.one_hot(actions, self.num_algorithms), axis=1
            )  # Q_primary(s, action) -> What would Q_primary predict for the action from memory?

            td_error = q_target_of_s - q_primary_of_s
            loss = self.loss(td_error=td_error, importance_sampling_weights=importance_sampling_weights)

        # Add priority to experience buffer---------------------
        self.experience_buffer.adjust_priority(ids=experience_ids, new_priority=td_error.numpy())

        # Adjust parameters to minimize loss--------------------
        parameters = self.q_primary.trainable_variables
        gradients = tape.gradient(loss, parameters)

        # Check for exploding gradients-------------------------
        self.gradient_magnitude = np.append(
            self.gradient_magnitude, int(tf.reduce_sum([tf.reduce_sum(gradient**2) for gradient in gradients])**.5))

        # Apply Gradient Update---------------------------------
        self.optimizer.apply_gradients(zip(gradients, parameters))
