import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


class PolicyModule(nn.Module):
    action_dims: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(120, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)  # 120
        x = nn.tanh(x)
        x = nn.Dense(84, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)  # 84
        x = nn.tanh(x)
        mean = nn.tanh(
            nn.Dense(self.action_dims, kernel_init=nn.initializers.orthogonal(0.01))(x)
        )
        logstd = self.param(
            "logstd", lambda rng, shape: jnp.zeros(shape), self.action_dims
        )
        return mean, jnp.exp(logstd)


class ValueModule(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1))(x)
        return x


def get_optimizer_step_fn(optim):
    # return function to update the parameters from a batch of gradients
    def update_step(params, grads, opt_state):
        mean_grads = jax.tree_map(lambda x: x.mean(axis=0), grads)
        mean_grads, opt_state = optim.update(mean_grads, opt_state)
        params = optax.apply_updates(params, mean_grads)
        return params, opt_state

    return update_step
