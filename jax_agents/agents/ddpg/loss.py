import jax.numpy as jnp


def get_policy_loss_fn(policy, q_value):
    def policy_loss(policy_params, q_value_params, observation):
        new_action = policy.apply(policy_params, q_value_params, observation)
        new_q = q_value.apply(q_value_params, observation, new_action)
        loss = -new_q
        return loss.squeeze(), {"agent/policy_loss": loss}

    return policy_loss


def get_q_value_loss_fn(policy, q_value, gamma):
    def q_value_loss(
        q_value_params,
        tgt_policy_params,
        tgt_q_value_params,
        observation,
        action,
        reward,
        done,
        next_observation,
    ):
        new_q = q_value.apply(q_value_params, observation, action)
        next_action = policy.apply(tgt_policy_params, next_observation)
        new_next_q = q_value.apply(tgt_q_value_params, next_observation, next_action)
        target_q = reward + (1 - done) * gamma * new_next_q

        td_loss = new_q - target_q
        loss = jnp.square(td_loss)

        return loss.squeeze(), {"agent/q_value_loss": loss}

    return q_value_loss
