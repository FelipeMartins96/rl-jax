from dataclasses import dataclass


@dataclass
class HyperparametersDDPG:
    algorithm_name: str = "Deep Deterministic Policy Gradient"
    environment_name: str = "Pendulum-v0"
    total_training_steps: int = int(1e6)
    replay_capacity: int = int(1e6)
    min_replay_size: int = 64
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 3e-4
    noise_sigma: float = 0.2
    tau: float = 1e-3
    seed: int = 0
    double_q: bool = False
