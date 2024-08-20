# API reference

## `DQNAgent`

Located in `Python/rl_agents/dqn_agent.py`.

- `DQNAgent(state_size, action_size, memory_size=10000)` creates an agent and validates its dimensions.
- `remember(state, action, reward, next_state, done)` appends a transition to the bounded replay buffer.
- `act(state)` selects a random or model-predicted discrete action.
- `replay(batch_size)` performs one gradient update when enough transitions exist.
- `save(path)` and `load(path)` persist TensorFlow weights.

States passed to `act` should have shape `(1, state_size)`. Actions are integer indexes in `[0, action_size)`.

## `ProceduralAnimationTrainer`

Located in `Python/rl_agents/behanim.py`.

- `ProceduralAnimationTrainer(state_size, action_size, environment)` wraps a `DQNAgent` around an environment adapter.
- `train(episodes, batch_size, max_steps=500)` runs training and returns the step count for each episode.
- `save(path)` and `load(path)` delegate weight persistence to the agent.

The environment must implement `reset()` and `step(action)`. `step` returns `(next_state, reward, done, info)`.

## Unreal `URLComponent`

Located under `Source/UnrealRLNPCs`.

- `PerformAction(int32 ActionIndex)` is the Blueprint-callable action hook.
- `ReceiveReward(float Reward)` is the Blueprint-callable reward hook.
- `ActionSpace` contains project-defined action labels.
- `StateSize` records the expected observation width.

Both C++ methods are extension points; their current bodies intentionally contain no project-specific transport or gameplay logic.
