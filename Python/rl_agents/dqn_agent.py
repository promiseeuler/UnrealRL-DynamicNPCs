"""Deep Q-Network agent used by the training examples."""

from collections import deque
import random

import numpy as np
import tensorflow as tf


class DQNAgent:
    """A compact DQN implementation with epsilon-greedy exploration."""

    def __init__(self, state_size, action_size, memory_size=10_000):
        if state_size <= 0 or action_size <= 0:
            raise ValueError("state_size and action_size must be positive")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store one transition in the bounded replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose an action using an epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_size)
        action_values = self.model.predict(state, verbose=0)
        return int(np.argmax(action_values[0]))

    def replay(self, batch_size):
        """Train the network on a random batch of stored transitions."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.memory) < batch_size:
            return None
        minibatch = random.sample(self.memory, batch_size)
        states = np.vstack([item[0] for item in minibatch])
        next_states = np.vstack([item[3] for item in minibatch])
        actions = np.asarray([item[1] for item in minibatch], dtype=int)
        rewards = np.asarray([item[2] for item in minibatch], dtype=float)
        dones = np.asarray([item[4] for item in minibatch], dtype=bool)
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        targets[np.arange(batch_size), actions] = rewards + self.gamma * np.max(next_q_values, axis=1) * (~dones)
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return history

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path)
