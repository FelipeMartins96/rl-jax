import flax.linen as nn
import jax.numpy as jnp
import distrax


class PolicyModule(nn.Module):
    "SAC Paper Actor Network"
    actions_dim: int

    @nn.compact
    def __call__(self, o):
        x = nn.Dense(256)(o)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        mean = nn.Dense(self.actions_dim)(x)
        log_std = nn.Dense(self.actions_dim)(x)

        return mean, log_std

    def evaluate(self, o, rng, epsilon=1e-6):
        mean, log_std = self(o)
        std = jnp.exp(log_std)

        normal = distrax.Normal(mean, std)
        z = normal.sample(seed=rng)
        action = nn.tanh(z)
        log_prob = normal.log_prob(z) - jnp.log(1 - jnp.power(action, 2) + epsilon)

        return action, log_prob.sum(), z, mean, log_std

    def get_action(self, o, rng):
        mean, log_std = self(o)
        std = jnp.exp(log_std)

        normal = distrax.Normal(mean, std)
        z = normal.sample(seed=rng)
        action = nn.tanh(z)

        return action


class QValueModule(nn.Module):
    "SAC Paper Critic Network"

    @nn.compact
    def __call__(self, o, a):
        x = jnp.concatenate([o, a], axis=-1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        return x


class ValueModule(nn.Module):
    "SAC Paper Value Network"

    @nn.compact
    def __call__(self, o):
        x = nn.Dense(256)(o)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)

        return x
