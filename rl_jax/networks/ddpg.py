import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import uniform, lecun_uniform


def bias_init_fn(fan_in):
    range = jnp.sqrt(jnp.array(1.0 / fan_in))
    return uniform(range)


class DDPGActor(nn.Module):
    """DDPG Paper Actor Network"""

    action_dim: int

    @nn.compact
    def __call__(self, obs):
        hidden_layer1 = nn.Dense(
            400,
            kernel_init=lecun_uniform(),
            bias_init=bias_init_fn(fan_in=obs.shape[-1]),
        )
        hidden_layer2 = nn.Dense(
            300, kernel_init=lecun_uniform(), bias_init=bias_init_fn(fan_in=400)
        )
        final_layer = nn.Dense(
            1, kernel_init=uniform(scale=3e-3), bias_init=uniform(scale=3e-3)
        )

        x = obs
        x = nn.relu(hidden_layer1(x))
        x = nn.relu(hidden_layer2(x))
        x = final_layer(x)

        return x


class DDPGCritic(nn.Module):
    """DDPG Paper Critic Network"""

    @nn.compact
    def __call__(self, obs, act):
        hidden_layer1 = nn.Dense(
            400,
            kernel_init=lecun_uniform(),
            bias_init=bias_init_fn(fan_in=obs.shape[-1]),
        )
        hidden_layer2 = nn.Dense(
            300,
            kernel_init=lecun_uniform(),
            bias_init=bias_init_fn(fan_in=300 + act.shape[-1]),
        )
        final_layer = nn.Dense(
            1, kernel_init=uniform(scale=3e-4), bias_init=uniform(scale=3e-4)
        )

        x = obs
        x = nn.relu(hidden_layer1(x))
        x = jnp.concatenate([x, act], axis=-1)
        x = nn.relu(hidden_layer2(x))
        x = final_layer(x)

        return x
