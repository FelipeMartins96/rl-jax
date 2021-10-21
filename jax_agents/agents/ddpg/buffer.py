import gym.spaces as spaces
import numpy as np


class ReplayBuffer:
    """Circular replay buffer for gym environments transitions"""

    def __init__(self, env_observation_space, env_action_space, capacity):
        """Initialize a replay buffer for the given environment.

        Args:
            env_observation_space: Environment observation space.
            env_action_space: Environment action space.
            capacity: Number of steps per rollout.
        """
        # Tests
        assert isinstance(env_observation_space, spaces.Box)
        assert isinstance(env_action_space, spaces.Box)

        self._capacity = capacity
        self._num_added = 0
        self._index = 0
        action_shape = env_action_space.shape
        observation_shape = env_observation_space.shape

        # Preallocate memory
        self._observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self._actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._dones = np.empty((capacity, 1), dtype=np.float32)
        self._next_observations = np.empty(
            (capacity, *observation_shape), dtype=np.float32
        )

    def add(self, observation, action, logprob, reward, done, next_observation):
        """Add a transition to the buffer."""
        assert self._num_added < self._capacity
        # TODO: Assert if transitions ranks are consistent

        self._observations[self._num_added] = observation
        self._actions[self._num_added] = action
        self._rewards[self._num_added] = reward
        self._dones[self._num_added] = done
        self._next_observations[self._num_added] = next_observation

        self._num_added += 1

    def get_batch(self, batch_size):
        """Sample a batch of transitions uniformly."""
        assert self.size >= batch_size

        batch_indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self._observations[batch_indices],
            self._actions[batch_indices],
            self._rewards[batch_indices],
            self._dones[batch_indices],
            self._next_observations[batch_indices],
        )

    @property
    def size(self) -> int:
        """Number of transitions in the buffer"""
        return min(self._capacity, self._num_added)
