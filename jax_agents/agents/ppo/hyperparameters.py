from dataclasses import dataclass


@dataclass
class HyperparametersPPO:
    algorithm_name: str = "Proximal Policy Optimization"
    environment_name: str = "Pendulum-v0"
    total_training_steps: int = int(2e6)
    n_rollout_steps: int = 2048
    update_epochs: int = 10
    n_mini_batches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    maximum_gradient_norm: float = 0.5
    adam_epsilon: float = 1e-5
    learning_rate: float = 3e-4
    clip_coefficient: float = 0.2
    entropy_loss_coefficient: float = 0.0
    value_loss_coefficient: float = 0.5
    seed: int = 0
