# Getting started

## 1. Set up Python

From the repository root, create an isolated environment and install the pinned dependency ranges:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Requirements/requirements.txt
```

Run `python Python/train.py` to confirm that TensorFlow loads and the example training loop starts.

## 2. Implement an environment adapter

`ProceduralAnimationTrainer` accepts an object with two methods:

```python
class UnrealEnvironment:
    def reset(self):
        """Return the initial one-dimensional NumPy observation."""

    def step(self, action):
        """Return (next_state, reward, done, info)."""
```

The transport is deliberately left to the host project. A local socket, Unreal Remote Control, or a custom Unreal subsystem can all work, provided calls are synchronous from the trainer's perspective.

## 3. Create the trainer

```python
from rl_agents.behanim import ProceduralAnimationTrainer

environment = UnrealEnvironment()
trainer = ProceduralAnimationTrainer(
    state_size=12,
    action_size=6,
    environment=environment,
)
trainer.train(episodes=100, batch_size=32)
trainer.save("npc.weights.h5")
```

Run this script with `Python` on the import path, for example `PYTHONPATH=Python python your_training_script.py`.

## 4. Integrate Unreal

The files under `Source/UnrealRLNPCs` demonstrate the Blueprint-facing action and reward hooks. Copy them into an existing Unreal C++ module, replace the placeholder bodies with project-specific behaviour, and expose observations through your chosen adapter.

This repository does not yet contain the `.uplugin`, module rules, or transport required for drop-in installation.
