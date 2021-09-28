from dataclasses import dataclass
import gym.spaces as spaces


@dataclass
class HyperparametersPPO:
    algorithm_name: str = "Proximal Policy Optimization"
    environment_name: str = "Pendulum-v0"
    rollout_steps: int = 128
