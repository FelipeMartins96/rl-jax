from dataclasses import dataclass


@dataclass
class HyperparametersSAC:
    algorithm_name: str = "Soft Actor Critic"
    environment_name: str = "Pendulum-v0"
    total_training_steps: int = int(1e6)
    replay_capacity: int = int(1e6)
    min_replay_size: int = 64
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 3e-4
    temperature: float = 1.0
    tau: float = 1e-3
    seed: int = 0
