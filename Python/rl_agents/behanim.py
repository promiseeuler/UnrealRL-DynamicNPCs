import numpy as np
import tensorflow as tf
from unreal_interface import UnrealEnvWrapper

class ProceduralAnimationTrainer:
    def __init__(self, state_size, action_size, env_config):
        self.state_size = state_size
        self.action_size = action_size
        self.env = UnrealEnvWrapper(env_config)
        self.memory = []
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        """
        Construct the neural network model for the reinforcement learning agent.
        This example uses a simple feedforward neural network with two hidden layers.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store the experience tuple (state, action, reward, next_state, done) in the agent's memory.
        """
        self.memory.append((state, action, reward, next_state, done))





 # Train the agent
    trainer.train(episodes=1000, batch_size=32)

    # Save the trained model
    trainer.save('procedural_animation_model.h5')