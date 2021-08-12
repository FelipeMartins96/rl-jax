import collections
import logging

import gym
import numpy as np

Transition = collections.namedtuple(
    "Transition", ["state", "action", "reward", "discount", "next_state"]
)
Batch = collections.namedtuple(
    "Batch", ["states", "actions", "rewards", "discounts", "next_states"]
)


class ReplayBuffer:
    """Circular replay buffer for gym environments transitions"""

    def __init__(self, environment: gym.Env, capacity: int):
        """Initialize a replay buffer for the given environment

        Args:
            environment: gym environment.
            capacity: maximum number of transitions to store in the buffer.
        """
        self._capacity = capacity
        self._num_added = 0
        self._index = 0

        if isinstance(environment.action_space, gym.spaces.Discrete):
            action_dim = (1,)
        else:
            action_dim = environment.action_space.shape

        state_dim = environment.observation_space.shape

        # Preallocate memory
        self._states = np.empty((capacity, *state_dim), dtype=np.float32)
        self._actions = np.empty((capacity, *action_dim), dtype=np.float32)
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._discounts = np.empty((capacity, 1), dtype=np.float32)
        self._next_states = np.empty((capacity, *state_dim), dtype=np.float32)

    def add(self, transition: Transition):
        """Add a transition to the buffer."""
        self._states[self._index] = transition.state
        self._actions[self._index] = transition.action
        self._rewards[self._index] = transition.reward
        self._discounts[self._index] = transition.discount
        self._next_states[self._index] = transition.next_state

        self._index = (self._index + 1) % self._capacity
        self._num_added = self._num_added + 1

    def sample(self, batch_size: int) -> Batch:
        """Sample a batch of transitions uniformly."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        if self.size < batch_size:
            logging.warn("Number of transitions on buffer is smaller than batch size")

        indices = np.random.randint(0, self.size, size=batch_size)
        return Batch(
            states=self._states[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            discounts=self._discounts[indices],
            next_states=self._next_states[indices],
        )

    @property
    def size(self) -> int:
        """Number of transitions in the buffer"""
        return min(self._capacity, self._num_added)
