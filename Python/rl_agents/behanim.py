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


    def act(self, state):
        """
        Determine the action to take based on the current state.
        The action is selected either randomly (for exploration) or by following the current policy (for exploitation).
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """
        Perform experience replay to update the agent's policy.
        Samples a batch of experiences from the agent's memory, computes the target Q-values, and updates the model.
        """
        if len(self.memory) < batch_size:
            return

        batch = np.random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        targets = rewards + self.gamma * (np.amax(self.model.predict(next_states), axis=1) * (1 - np.array(dones)))
        target_f = self.model.predict(states)
        target_f[np.arange(batch_size), actions] = targets
        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



    def train(self, episodes, batch_size):
        """
        Train the agent by interacting with the environment and performing experience replay.
        """
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            for time in range(500):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    print(f"episode: {e}/{episodes}, score: {time}")
                    break

                if len(self.memory) > batch_size:
                    self.replay(batch_size)


    def save(self, name):
        """
        Save the trained model to a file.
        """
        self.model.save_weights(name)

    def load(self, name):
        """
        Load a pre-trained model from a file.
        """
        self.model.load_weights(name)


    if __:
    # Set up the environment configuration
    env_config = {
        'host': 'localhost',
        'port': 8000,
        'scene': 'MyScene',
        'character': 'MyCharacter'
    }

    # Initialize the trainer
    state_size = 12  # Example state size (e.g., joint angles, velocities)
    action_size = 6  # Example action size (e.g., joint torques)
    trainer = ProceduralAnimationTrainer(state_size, action_size, env_config)




 # Train the agent
    trainer.train(episodes=1000, batch_size=32)

    # Save the trained model
    trainer.save('procedural_animation_model.h5')