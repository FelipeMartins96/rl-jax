import abc
from typing import Any

import numpy as np
from rl_jax.replay import Transition

Action = Any  # np.ndarray | int


class Agent(abc.ABC):
    """RL Agent which can act and learn"""

    @abc.abstractmethod
    def select_action(self, observation: np.ndarray) -> Action:
        """Sample an action from the policy"""

    @abc.abstractmethod
    def observe(self, step: Transition):
        """Observe an environment step transition"""

    @abc.abstractmethod
    def update(self):
        """Update the agent parameters"""
