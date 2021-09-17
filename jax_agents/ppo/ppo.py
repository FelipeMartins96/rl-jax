import jax.numpy as jnp
import distrax
import numpy as np

from jax.config import config

config.update("jax_debug_nans", True)  # break on nans
# config.update('jax_disable_jit', True)


def build_ppo_loss(policy_model, value_model, epsilon, c1, c2):
    def ppo_loss(
        p_params, v_params, observation, action, logprob, target_value, advantage
    ):
        mean, sigma = policy_model.apply(p_params, observation)
        dist = distrax.MultivariateNormalDiag(mean, sigma)
        new_logprob = dist.log_prob(action)
        ratio = jnp.exp(new_logprob - logprob)
        p_loss1 = ratio * advantage
        p_loss2 = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
        l_clip = jnp.fmin(p_loss1, p_loss2)

        value = value_model.apply(v_params, observation)
        l_vf = jnp.square(value - target_value)

        S = dist.entropy()

        loss = +l_clip - c1 * l_vf + c2 * S

        info = dict(l_clip=l_clip, l_vf=l_vf, S=S, loss=loss)

        return loss.squeeze(), info

    return ppo_loss


def build_sample_action(policy_model):
    def sample_action(rng, p_params, observation):
        mean, sigma = policy_model.apply(p_params, observation)
        dist = distrax.MultivariateNormalDiag(mean, sigma)
        action = dist.sample(seed=rng)
        logprob = dist.log_prob(action)
        return action, logprob

    return sample_action


import flax.linen as nn


class Policy(nn.Module):
    action_dims: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(120)(x)  # 120
        x = nn.tanh(x)
        x = nn.Dense(84)(x)  # 84
        x = nn.tanh(x)
        mean = nn.tanh(nn.Dense(self.action_dims)(x))
        logstd = self.param(
            "logstd", lambda rng, shape: jnp.zeros(shape), self.action_dims
        )
        return mean, jnp.exp(logstd)


class Value(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


import jax
import gym
import optax


def optim_update_fcn(optim):
    @jax.jit
    def update_step(params, grads, opt_state):
        mean_grads = jax.tree_map(lambda x: x.mean(axis=0), grads)
        mean_grads, opt_state = optim.update(mean_grads, opt_state)
        params = optax.apply_updates(params, mean_grads)
        return params, opt_state

    return update_step


import wandb

if __name__ == "__main__":
    env = gym.wrappers.RecordVideo(gym.make("LunarLanderContinuous-v2"), "./monitor/")
    # env = gym.make("LunarLanderContinuous-v2")
    # wandb.init(project="cleanrl.benchmark", entity="felipemartins", mode="disabled")
    wandb.init(project="cleanrl.benchmark", entity="felipemartins", monitor_gym=True)
    rng = jax.random.PRNGKey(0)
    epsilon = 0.1
    c1 = 0.25
    c2 = 0.01
    gamma = 0.99
    update_epochs = 3
    learning_rate = 7e-4
    max_grad_norm = 0.5

    p_model = Policy(env.action_space.shape[0])
    v_model = Value()

    rng, p_key, v_key = jax.random.split(rng, 3)
    p_params = p_model.init(p_key, env.observation_space.sample())
    v_params = v_model.init(v_key, env.observation_space.sample())

    ppo_loss = build_ppo_loss(p_model, v_model, epsilon, c1, c2)
    ppo_loss_grad = jax.jit(jax.grad(ppo_loss, argnums=[0, 1], has_aux=True))
    ppo_loss_grad_vmap = jax.vmap(ppo_loss_grad, in_axes=(None, None, 0, 0, 0, 0, 0))

    optim = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.scale_by_adam(),
        optax.scale(learning_rate),
    )

    p_opt_state = optim.init(p_params)
    v_opt_state = optim.init(v_params)

    optim_update_step = optim_update_fcn(optim)

    sample_action = jax.jit(build_sample_action(p_model))
    get_v = jax.jit(v_model.apply)

    import time

    st = time.time()
    for ep in range(50000):
        done = False
        s = env.reset()

        s_v = []
        a_v = []
        logprob_v = []
        v_v = []
        r_v = []

        while not done:
            rng, a_key = jax.random.split(rng, 2)
            a, logprob = sample_action(a_key, p_params, s)
            # Scaling to environment, assumes env action_space is simmetric around 0
            clipped_a = np.clip(
                a * env.action_space.high, env.action_space.low, env.action_space.high
            )
            s_, r, done, _ = env.step(np.array(clipped_a))
            v = get_v(v_params, s)
            s_v.append(s)
            a_v.append(a)
            logprob_v.append(logprob)
            v_v.append(v)
            r_v.append(r)

        steps = len(r_v)
        returns = np.zeros_like(r_v)
        returns[-1] = r_v[-1]
        for t in reversed(range(steps - 1)):
            returns[t] = r_v[t] + gamma * returns[t + 1]
        # advantages are returns - baseline, value estimates in our case
        advantages = [returns[i] - v_v[i] for i in range(steps)]

        s_j, a_j, lp_j, r_j, adv_j = (
            jnp.array(s_v),
            jnp.array(a_v),
            jnp.array(logprob_v),
            jnp.array(returns),
            jnp.array(advantages),
        )

        for i in range(update_epochs):
            (p_grad, v_grad), info = ppo_loss_grad_vmap(
                p_params, v_params, s_j, a_j, lp_j, r_j, adv_j
            )

            p_params, p_opt_state = optim_update_step(p_params, p_grad, p_opt_state)
            v_params, v_opt_state = optim_update_step(v_params, v_grad, v_opt_state)

        et = time.time()
        info_mean = jax.tree_map(lambda x: x.mean(axis=0), info)
        info_mean.update(
            dict(
                rw=sum(r_v),
                mean_v=sum(v_v) / steps,
                steps_s=steps / (et - st),
                mean_return=sum(returns) / steps,
                mean_adv=adv_j.mean(),
                logstd_param0=p_params["params"]["logstd"][0],
                logstd_param1=p_params["params"]["logstd"][1],
            )
        )
        wandb.log(info_mean)
        st = et
