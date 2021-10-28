import jax.numpy as jnp


def get_policy_loss_fn(policy, q_value, temperature):
    def policy_loss(policy_params, q_value_params, observation, rng):
        new_action, new_action_log_prob, _, _, _ = policy.apply(
            policy_params, observation, rng, method=policy.evaluate
        )
        new_q = q_value.apply(q_value_params, observation, new_action)
        loss = new_action_log_prob * temperature - new_q
        return loss.squeeze(), {
            "agent/policy_loss": loss,
            "agent/policy_entropy": -new_action_log_prob,
        }

    return policy_loss


def get_q_value_loss_fn(q_value, value, gamma):
    def q_value_loss(
        q_value_params,
        tgt_value_params,
        observation,
        action,
        reward,
        done,
        next_observation,
    ):
        next_v = value.apply(tgt_value_params, next_observation)
        target_q = reward + (1 - done) * gamma * next_v
        new_q = q_value.apply(q_value_params, observation, action)
        td_loss = new_q - target_q
        loss = jnp.square(td_loss)

        return loss.squeeze(), {"agent/q_value_loss": loss}

    return q_value_loss


def get_value_loss_fn(policy, q_value, value):
    def value_loss(value_params, policy_params, q_value_params, observation, rng):
        new_action = policy.apply(
            policy_params, observation, rng, method=policy.get_action
        )
        target_v = q_value.apply(q_value_params, observation, new_action)
        new_value = value.apply(value_params, observation)
        td_loss = new_value - target_v
        loss = jnp.square(td_loss)
        return loss.squeeze(), {"agent/value_loss": loss}

    return value_loss
