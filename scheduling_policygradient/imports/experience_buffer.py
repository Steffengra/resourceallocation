import numpy as np


class ExperienceBuffer:
    def __init__(self, max_experiences, min_priority, alpha):
        self.state: list = []
        self.action: list = []
        self.reward: list = []
        self.following_state: list = []
        self.priority: np.ndarray = np.array([])

        self.alpha = alpha  # Priority adjustment factor, 0..1+
        self.min_priority: float = min_priority  # Minimum priority
        self.max_experiences: int = max_experiences  # Max buffer size
        self.sum_priority: float = 0  # holds sum_k p_k
        self.probability: np.ndarray = np.array([])  # holds priority converted to probability, P(i) = p_i/sum_k p_k

        self.maximum_priority: float = min_priority  # Maximum encountered priority so far, assigned to new exp.

    def add_experience(self, state, action, reward, following_state):
        self.state.append(state)
        self.action.append(action)
        self.reward.append(reward)
        self.following_state.append(following_state)
        self.priority = np.append(self.priority, self.maximum_priority)

        if len(self.state) > self.max_experiences:
            self.state.pop(0)
            self.action.pop(0)
            self.reward.pop(0)
            self.following_state.pop(0)
            self.priority = np.delete(self.priority, 0)

    def adjust_priority(self, ids, new_priority):
        for ii in range(len(ids)):
            # print('a', abs(new_priority[ii]))
            # print('m', self.maximum_priority)
            if abs(new_priority[ii]) > self.maximum_priority:
                self.maximum_priority = abs(new_priority[ii]) + self.min_priority
            self.priority[ids[ii]] = self.maximum_priority  # Adjust priority

    def update_probabilities(self):
        self.sum_priority = np.sum(np.power(self.priority, self.alpha))  # Update Priority sum
        self.probability = np.power(self.priority, self.alpha) / self.sum_priority  # Update probabilities

    def sample(self, batch_size, beta):
        self.update_probabilities()

        experience_ids = np.random.choice(len(self.state), size=batch_size, p=self.probability)
        states = np.asarray([self.state[ii] for ii in experience_ids], dtype='float32')
        actions = np.asarray([self.action[ii] for ii in experience_ids])
        rewards = np.asarray([self.reward[ii] for ii in experience_ids])
        following_states = np.asarray([self.following_state[ii] for ii in experience_ids], dtype='float32')

        probabilities = np.asarray([self.probability[ii] for ii in experience_ids])
        importance_sampling_weights = 1 / np.power(len(self.probability) * probabilities, beta)
        # Scale down weights per batch
        importance_sampling_weights = importance_sampling_weights / np.max(importance_sampling_weights)

        return experience_ids, states, actions, rewards, following_states, importance_sampling_weights
