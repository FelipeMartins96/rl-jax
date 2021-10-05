import jax
import numpy as np


def get_calculate_gae_fn(v_model, gamma, gae_lambda, n_rollout_steps):
    get_v = v_model.apply
    get_v_vmap = jax.jit(jax.vmap(get_v, in_axes=(None, 0)))

    def gae_advantages(v_params, rollout):
        observations, actions, logprobs, rewards, dones, next_observations = rollout
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)

        values = get_v_vmap(v_params, observations)
        next_values = get_v_vmap(v_params, next_observations)

        lastgaelam = 0
        for t in reversed(range(n_rollout_steps)):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
            lastgaelam = advantages[t]
            returns[t] = advantages[t] + values[t]

        return (
            observations,
            actions,
            logprobs,
            returns,
            advantages,
            values,
        )

    return gae_advantages
