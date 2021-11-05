import jax.numpy as jnp
import jax


def get_gaussian_noise_fn(action_space, scale):
    def add_gaussian_noise(key, action):
        noise = jax.random.normal(key, action_space.shape) * scale
        return jnp.clip(
            action + noise, -1.0, 1.0
        )  # TODO: Assumes -1, 1 bounded action space

    return add_gaussian_noise
