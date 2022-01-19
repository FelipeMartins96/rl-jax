import jax
import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import uniform, lecun_uniform


def bias_init_fn(fan_in):
    # DDPG paper bias init
    range = jnp.sqrt(jnp.array(1.0 / fan_in))
    return uniform(range)


class PolicyModule(nn.Module):
    """DDPG Paper Actor(Policy) Network"""

    action_dim: int

    @nn.compact
    def __call__(self, o):
        x = nn.Dense(
            256,
            kernel_init=lecun_uniform(),
            bias_init=bias_init_fn(fan_in=o.shape[-1]),
        )(o)
        x = nn.relu(x)
        x = nn.Dense(
            256, kernel_init=lecun_uniform(), bias_init=bias_init_fn(fan_in=256)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            512, kernel_init=lecun_uniform(), bias_init=bias_init_fn(fan_in=256)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            512, kernel_init=lecun_uniform(), bias_init=bias_init_fn(fan_in=512)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            256, kernel_init=lecun_uniform(), bias_init=bias_init_fn(fan_in=512)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=uniform(scale=3e-3),
            bias_init=uniform(scale=3e-3),
        )(x)
        x = nn.tanh(x)

        return x


class QValueModule(nn.Module):
    """DDPG Paper Q Value (Critic) Network"""

    @nn.compact
    def __call__(self, o, a):
        x = nn.Dense(
            512,
            kernel_init=lecun_uniform(),
            bias_init=bias_init_fn(fan_in=o.shape[-1]),
        )(o)
        x = nn.relu(x)
        x = nn.Dense(
            512, kernel_init=lecun_uniform(), bias_init=bias_init_fn(fan_in=512)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            1024, kernel_init=lecun_uniform(), bias_init=bias_init_fn(fan_in=512)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            512,
            kernel_init=lecun_uniform(),
            bias_init=bias_init_fn(fan_in=1024 + a.shape[-1]),
        )(jnp.concatenate([x, a], axis=-1))
        x = nn.relu(x)
        x = nn.Dense(
            512, kernel_init=lecun_uniform(), bias_init=bias_init_fn(fan_in=512)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=uniform(scale=3e-4), bias_init=uniform(scale=3e-4))(
            x
        )

        return x


class DoubleQValueModule(nn.Module):
    """Double Q Value (Critic) Network"""

    @nn.compact
    def __call__(self, o, a):
        critic1 = QValueModule()(o, a)
        critic2 = QValueModule()(o, a)

        return critic1, critic2


def target_params_sync_fn(params, tgt_params, tau):
    """Soft target network update."""

    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), params, tgt_params
    )

    return new_target_params
