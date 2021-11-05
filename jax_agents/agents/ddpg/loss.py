import jax.numpy as jnp
from functools import partial


def get_policy_loss_fn(policy, q_value, is_double_q):
    def policy_loss(policy_params, q_value_params, observation, double_q):
        new_action = policy.apply(policy_params, observation)

        if double_q:
            q1, q2 = q_value.apply(q_value_params, observation, new_action)
            new_q = jnp.minimum(q1, q2)
        else:
            new_q = q_value.apply(q_value_params, observation, new_action)

        loss = -new_q
        return loss.squeeze(), {"agent/policy_loss": loss}

    return partial(policy_loss, double_q=is_double_q)


def get_q_value_loss_fn(policy, q_value, gamma, is_double_q):
    def q_value_loss(
        q_value_params,
        tgt_policy_params,
        tgt_q_value_params,
        observation,
        action,
        reward,
        done,
        next_observation,
        double_q,
    ):
        next_action = policy.apply(tgt_policy_params, next_observation)

        if double_q:
            new_next_q1, new_next_q2 = q_value.apply(
                tgt_q_value_params, next_observation, next_action
            )
            new_next_q = jnp.minimum(new_next_q1, new_next_q2)
            target_q = reward + (1 - done) * gamma * new_next_q
            new_q1, new_q2 = q_value.apply(q_value_params, observation, action)
            td_loss1, td_loss2 = new_q1 - target_q, new_q2 - target_q
            loss = jnp.square(td_loss1) + jnp.square(td_loss2)

        else:
            new_next_q = q_value.apply(
                tgt_q_value_params, next_observation, next_action
            )
            target_q = reward + (1 - done) * gamma * new_next_q
            new_q = q_value.apply(q_value_params, observation, action)
            td_loss = new_q - target_q
            loss = jnp.square(td_loss)

        return loss.squeeze(), {"agent/q_value_loss": loss}

    return partial(q_value_loss, double_q=is_double_q)
