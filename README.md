# UnrealRL Dynamic NPCs

An experimental reinforcement-learning toolkit for prototyping adaptive NPC behaviour with Unreal Engine. The repository combines a TensorFlow DQN reference implementation with an Unreal `ActorComponent` scaffold for game-side integration.

> **Project status:** research prototype. The Python agent and training loop are usable examples; the Unreal component is an integration starting point and still requires a project-specific transport and environment adapter.

## What is included

- A Deep Q-Network agent with bounded experience replay and epsilon-greedy exploration
- A reusable trainer that accepts any environment implementing `reset()` and `step()`
- A small standalone training example for validating the Python setup
- An Unreal Engine component scaffold exposing action and reward hooks to Blueprints
- Focused setup and API documentation

## Architecture

```text
Unreal NPC / Blueprint
        |
        | observations, rewards, actions
        v
Environment adapter (project-specific)
        |
        v
ProceduralAnimationTrainer -> DQNAgent -> TensorFlow model
```

The environment adapter is intentionally application-defined. It is responsible for translating Unreal state into fixed-size NumPy observations and converting discrete actions back into game behaviour.

## Quick start

Requirements: Python 3.9+ and a virtual environment are recommended.

```bash
git clone https://github.com/promiseeuler/UnrealRL-DynamicNPCs.git
cd UnrealRL-DynamicNPCs
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Requirements/requirements.txt
python Python/train.py
```

The included `train.py` uses generated observations as a smoke test; it is not an Unreal environment. See [Getting started](Docs/GettingStarted.md) to connect an adapter.

## Repository structure

```text
Python/                     DQN agent and training examples
Source/UnrealRLNPCs/        Unreal C++ integration scaffold
Docs/                       Setup and API documentation
Requirements/               Python dependencies
```

## Current limitations

- No network or IPC transport between Unreal and Python is bundled.
- The C++ source is not yet packaged as a standalone `.uplugin` module.
- The example uses a discrete action space and a single online DQN network.
- Production training should add target networks, checkpointing, metrics, and deterministic environment seeding.

## Documentation

- [Getting started](Docs/GettingStarted.md)
- [API reference](Docs/API_Reference.md)

## Contributing

Issues and focused pull requests are welcome. Please describe the environment, Unreal version, and reproducible test steps when reporting integration problems.

## License

Released under the [MIT License](LICENSE).
