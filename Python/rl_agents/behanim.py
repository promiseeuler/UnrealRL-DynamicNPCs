"""Procedural-animation trainer for Unreal-compatible environments."""

from typing import Protocol, Tuple

import numpy as np

from .dqn_agent import DQNAgent


class Environment(Protocol):
    """Minimal environment contract required by the trainer."""

    def reset(self) -> np.ndarray:
        ...

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        ...


class ProceduralAnimationTrainer:
    """Train a DQN agent against an injected Unreal environment adapter."""

    def __init__(self, state_size: int, action_size: int, environment: Environment):
        self.state_size = state_size
        self.environment = environment
        self.agent = DQNAgent(state_size, action_size)

    def train(self, episodes: int, batch_size: int, max_steps: int = 500):
        """Run training and return the number of steps per episode."""
        episode_lengths = []
        for episode in range(episodes):
            state = np.asarray(self.environment.reset()).reshape(1, self.state_size)
            steps = 0
            for step in range(max_steps):
                action = self.agent.act(state)
                next_state, reward, done, _ = self.environment.step(action)
                next_state = np.asarray(next_state).reshape(1, self.state_size)
                self.agent.remember(state, action, reward, next_state, done)
                self.agent.replay(batch_size)
                state = next_state
                steps = step + 1
                if done:
                    break
            episode_lengths.append(steps)
            print(f"episode: {episode + 1}/{episodes}, steps: {steps}")
        return episode_lengths

    def save(self, path: str):
        self.agent.save(path)

    def load(self, path: str):
        self.agent.load(path)
