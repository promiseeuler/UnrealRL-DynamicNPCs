from rl_agents.dqn_agent import DQNAgent
import numpy as np


def train_agent(episodes=1000, max_steps=500, batch_size=32):
    state_size = 4  # Example state size
    action_size = 2  # Example action size
    agent = DQNAgent(state_size, action_size)
    
    for e in range(episodes):
        state = np.random.rand(1, state_size)  # Get initial state
        for time in range(max_steps):
            action = agent.act(state)
            
            # Get next_state, reward, done from environment
            next_state = np.random.rand(1, state_size)  # Placeholder
            reward = np.random.rand()  # Placeholder
            done = False  # Placeholder
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            

            if done:
                print(f"episode: {e}/{episodes}, score: {time}")
                break
            agent.replay(batch_size)

if __name__ == "__main__":
    train_agent()
