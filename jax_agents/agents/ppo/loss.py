import jax.numpy as jnp
import distrax


def get_ppo_loss_fn(policy_model, value_model, clip_coef, c1, c2):
    def ppo_loss(
        p_params,
        v_params,
        observation,
        action,
        logprob,
        target_value,
        old_values,
        advantage,
    ):
        mean, sigma = policy_model.apply(p_params, observation)
        dist = distrax.MultivariateNormalDiag(mean, sigma)
        new_logprob = dist.log_prob(action)
        ratio = jnp.exp(new_logprob - logprob)
        p_loss1 = ratio * advantage
        p_loss2 = jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef) * advantage
        l_clip = jnp.fmin(p_loss1, p_loss2)

        value = value_model.apply(v_params, observation)
        l_vf_1 = jnp.square(value - target_value)
        v_clipped = old_values + jnp.clip(value - old_values, -clip_coef, clip_coef)
        l_vf_2 = jnp.square(v_clipped - target_value)
        l_vf = 0.5 * jnp.fmax(l_vf_1, l_vf_2)

        S = jnp.expand_dims(dist.entropy(), 0)

        l1 = -l_clip
        l2 = c1 * l_vf
        l3 = -c2 * S

        loss = l1 + l2 + l3

        a_kl = logprob - new_logprob

        info = dict(l_clip=l1, l_vf=l_vf, S=S, loss=loss, approx_kl=a_kl)

        return loss.squeeze(), info

    return ppo_loss
