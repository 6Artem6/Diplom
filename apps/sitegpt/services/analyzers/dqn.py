import random
import numpy as np
from collections import deque


class DQN:
    def __init__(self, state_size, action_size, hidden_size=256, gamma=0.9, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        # Простейшая модель — матрица Q-значений
        self.q_table = np.zeros((state_size, action_size))

    def predict(self, state):
        return self.q_table[state]

    def update(self, state, action, reward, next_state):
        q_next = np.max(self.q_table[next_state])
        q_target = reward + self.gamma * q_next
        self.q_table[state, action] += self.lr * (
            q_target - self.q_table[state, action]
        )


class ExperienceReplay:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))


class ActionGraphOptimizer:
    def __init__(self, state_size=512, action_size=10):
        self.q_network = DQN(state_size, action_size)
        self.experience_buffer = ExperienceReplay()

    def encode_state(self, page_graph, user_intent):
        # упрощение: хэшируем граф+интент в индекс
        return hash(str(page_graph) + str(user_intent)) % self.q_network.state_size

    def generate_candidates(self, state):
        # создаём случайные последовательности действий
        return [random.randint(0, self.q_network.action_size - 1) for _ in range(5)]

    def simulate_execution(self, sequence):
        # toy reward: чем ближе к "submit", тем выше
        if sequence == 3:  # например, 3 = "click login"
            return 10
        return -1  # бесполезное действие

    def optimize_action_sequence(self, page_graph, user_intent):
        state = self.encode_state(page_graph, user_intent)

        action_sequences = self.generate_candidates(state)

        for action in action_sequences:
            reward = self.simulate_execution(action)
            next_state = (state + 1) % self.q_network.state_size
            self.experience_buffer.add(state, action, reward, next_state)
            self.q_network.update(state, action, reward, next_state)

        return self.select_best_sequence(state)

    def select_best_sequence(self, state):
        q_values = self.q_network.predict(state)
        return np.argmax(q_values)
