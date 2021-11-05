import jax.numpy as jnp
import jax


def get_gaussian_noise_fn(action_space, sigma):
    def add_gaussian_noise(key, state, action):
        noise = jax.random.normal(key, action_space.shape) * sigma

        # TODO: Assumes -1, 1 bounded action space
        return jnp.clip(action + noise, -1.0, 1.0), None

    return add_gaussian_noise


# Based on https://stable-baselines3.readthedocs.io/en/v1.0/_modules/stable_baselines3/common/noise.html#OrnsteinUhlenbeckActionNoise
def get_ornstein_uhlenbeck_noise_fn(action_space, sigma, theta, dt):
    mu = jnp.zeros(action_space.shape)  # TODO: Assumes 0 mean

    def add_ornstein_uhlenbeck_noise(key, prev_noise, action):
        noise = (
            prev_noise
            + theta * (mu - prev_noise) * dt
            + sigma * jnp.sqrt(dt) * jax.random.normal(key, action_space.shape)
        )

        # TODO: Assumes -1, 1 bounded action space
        return jnp.clip(action + noise, -1.0, 1.0), noise

    return add_ornstein_uhlenbeck_noise
