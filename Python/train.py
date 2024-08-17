from rl_agents.dqn_agent import DQNAgent
import numpy as np

def train_agent():
    state_size = 4  # Example state size
    action_size = 2  # Example action size
    agent = DQNAgent(state_size, action_size)
    
    episodes = 1000
    batch_size = 32

    

    for e in range(episodes):
        state = np.random.rand(1, state_size)  # Get initial state
        for time in range(500):  # 500 timesteps per episode
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
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

if __name__ == "__main__":
    train_agent()