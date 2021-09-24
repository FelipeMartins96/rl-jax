import time

import distrax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pybullet_envs
from jax.config import config
import tqdm

import wandb

# config.update("jax_debug_nans", True)  # break on nans
# config.update('jax_disable_jit', True)


def build_ppo_loss(policy_model, value_model, epsilon, c1, c2):
    def ppo_loss(
        p_params, v_params, observation, action, logprob, target_value, old_values, advantage
    ):
        mean, sigma = policy_model.apply(p_params, observation)
        dist = distrax.MultivariateNormalDiag(mean, sigma)
        new_logprob = dist.log_prob(action)
        ratio = jnp.exp(new_logprob - logprob)
        p_loss1 = ratio * advantage
        p_loss2 = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage
        l_clip = jnp.fmin(p_loss1, p_loss2)

        value = value_model.apply(v_params, observation)
        l_vf_1 = jnp.square(value - target_value)
        v_clipped = old_values + jnp.clip(value - old_values, -epsilon, epsilon)
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


def build_sample_action(policy_model):
    def sample_action(rng, p_params, observation):
        mean, sigma = policy_model.apply(p_params, observation)
        dist = distrax.MultivariateNormalDiag(mean, sigma)
        action = dist.sample(seed=rng)
        logprob = dist.log_prob(action)
        return action, logprob

    return sample_action


class Policy(nn.Module):
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


class Value(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.tanh(x)
        x = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1))(x)
        return x


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizedEnv(gym.core.Wrapper):
    def __init__(
        self,
        env,
        ob=True,
        ret=True,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
    ):
        super(NormalizedEnv, self).__init__(env)
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=(1,)) if ret else None
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(())
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        infos["real_reward"] = rews
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(np.array([self.ret].copy()))
            rews = np.clip(
                rews / np.sqrt(self.ret_rms.var + self.epsilon),
                -self.cliprew,
                self.cliprew,
            )
        self.ret = self.ret * (1 - float(dones))
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon),
                -self.clipob,
                self.clipob,
            )
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(())
        obs = self.env.reset()
        return self._obfilt(obs)


class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np

        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)


def optim_update_fcn(optim):
    @jax.jit
    def update_step(params, grads, opt_state):
        mean_grads = jax.tree_map(lambda x: x.mean(axis=0), grads)
        mean_grads, opt_state = optim.update(mean_grads, opt_state)
        params = optax.apply_updates(params, mean_grads)
        return params, opt_state

    return update_step


def get_calculate_gae_fn(v_model, gamma, gae_lambda, num_steps):
    get_v = jax.jit(v_model.apply)
    get_v_vmap = jax.vmap(get_v, in_axes=(None, 0))

    def gae_advantages(v_params, rollout):
        observations, actions, logprobs, rewards, dones, next_observation = rollout
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        values = get_v_vmap(
            v_params, np.concatenate([observations, next_observation], axis=0)
        )
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
            lastgaelam = advantages[t]
            returns[t] = advantages[t] + values[t]

        return (
            jnp.array(observations),
            jnp.array(actions),
            jnp.array(logprobs),
            jnp.array(returns),
            jnp.array(advantages),
            jnp.array(values[:-1]),
        )

    return gae_advantages


class RolloutBuffer:
    """Circular replay buffer for gym environments transitions"""

    def __init__(self, environment, capacity):
        """Initialize a replay buffer for the given environment

        Args:
            environment: gym environment.
            capacity: maximum number of transitions to store in the buffer.
        """
        self._capacity = capacity
        self._num_added = 0

        if isinstance(environment.action_space, gym.spaces.Discrete):
            action_dim = (1,)
        else:
            action_dim = environment.action_space.shape

        state_dim = environment.observation_space.shape

        # Preallocate memory
        self._observations = np.empty((capacity, *state_dim), dtype=np.float32)
        self._actions = np.empty((capacity, *action_dim), dtype=np.float32)
        self._logprobs = np.empty((capacity, 1), dtype=np.float32)
        self._rewards = np.empty((capacity, 1), dtype=np.float32)
        self._dones = np.empty((capacity, 1), dtype=np.float32)
        self._next_observation = np.empty((1, *state_dim), dtype=np.float32)

    def add(self, observation, action, logprob, reward, done, next_observation):
        """Add a transition to the buffer."""
        if self._num_added >= self._capacity:
            raise ValueError("Adding transition to full buffer")

        self._observations[self._num_added] = observation
        self._actions[self._num_added] = action
        self._logprobs[self._num_added] = logprob
        self._rewards[self._num_added] = reward
        self._dones[self._num_added] = done
        self._next_observation[0] = next_observation

        self._num_added += 1

    def get_rollout(self):
        """Sample a batch of transitions uniformly."""
        if self.size != self._capacity:
            raise ValueError("Incomplete Rollout")

        return (
            self._observations,
            self._actions,
            self._logprobs,
            self._rewards,
            self._dones,
            self._next_observation,
        )

    def clear(self):
        self._num_added = 0

    @property
    def size(self) -> int:
        """Number of transitions in the buffer"""
        return self._num_added


if __name__ == "__main__":
    env = gym.make("HopperBulletEnv-v0")
    env = ClipActionsWrapper(env)
    env = gym.wrappers.RecordVideo(env, "./monitor/", step_trigger=lambda x: x % 25000 == 0)
    env = NormalizedEnv(env)
    wandb.init(
        project="rl-jax", entity="felipemartins", monitor_gym=True, save_code=True
    )
    learning_rate = 3e-4
    seed = 1
    total_timesteps = 2000000
    n_minibatches = 32
    num_steps = 2048
    num_envs = 1
    gamma = 0.99
    gae_lambda = 0.95
    c2 = 0.0  # ent-coef
    c1 = 0.5  # vf-coef
    max_grad_norm = 0.5
    epsilon = 0.2  # clip-coef
    update_epochs = 10
    # target-kl = 0.03
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // n_minibatches)

    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    rng = jax.random.PRNGKey(seed)

    p_model = Policy(env.action_space.shape[0])
    v_model = Value()

    rng, p_key, v_key = jax.random.split(rng, 3)
    p_params = p_model.init(p_key, env.observation_space.sample())
    v_params = v_model.init(v_key, env.observation_space.sample())

    ppo_loss = build_ppo_loss(p_model, v_model, epsilon, c1, c2)
    ppo_loss_grad = jax.jit(jax.grad(ppo_loss, argnums=[0, 1], has_aux=True))
    ppo_loss_grad_vmap = jax.vmap(ppo_loss_grad, in_axes=(None, None, 0, 0, 0, 0, 0, 0))

    optim = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.scale_by_adam(eps=1e-5),
        optax.scale(-learning_rate),
    )

    p_opt_state = optim.init(p_params)
    v_opt_state = optim.init(v_params)

    optim_update_step = optim_update_fcn(optim)

    sample_action = jax.jit(build_sample_action(p_model))

    calculate_gae = get_calculate_gae_fn(v_model, gamma, gae_lambda, num_steps)

    buffer = RolloutBuffer(env, num_steps)

    st = time.time()
    next_obs = env.reset()
    total_steps = 0
    done = False
    ep_rw = 0

    num_updates = total_timesteps // batch_size
    with tqdm.tqdm(total=total_timesteps) as pbar:
        for update_index in range(num_updates):

            buffer.clear()
            ep_rws = []

            # Get Rollout
            for step_index in range(num_steps):
                total_steps += 1

                obs = next_obs

                rng, a_key = jax.random.split(rng, 2)
                a, logprob = sample_action(a_key, p_params, obs)

                next_obs, reward, done, env_info = env.step(np.array(a))
                ep_rw += env_info['real_reward']

                buffer.add(obs, a, logprob, reward, done, next_obs)

                if done:
                    next_obs = env.reset()
                    ep_rws.append(ep_rw)
                    ep_rw = 0

            # Calculate Advantages
            s_j, a_j, lp_j, r_j, adv_j, v_j = calculate_gae(v_params, buffer.get_rollout())

            # Update Networks

            for i in range(update_epochs):
                for start in range(0, batch_size, minibatch_size):
                    end = start + minibatch_size
                    # Normalize advantages
                    adv_mb = adv_j[start:end]
                    adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                    (p_grad, v_grad), info = ppo_loss_grad_vmap(
                        p_params,
                        v_params,
                        s_j[start:end],
                        a_j[start:end],
                        lp_j[start:end],
                        r_j[start:end],
                        v_j[start:end],
                        adv_mb,
                    )

                    p_params, p_opt_state = optim_update_step(p_params, p_grad, p_opt_state)
                    v_params, v_opt_state = optim_update_step(v_params, v_grad, v_opt_state)

            # Logging
            pbar.update(num_steps)

            et = time.time()
            info_mean = jax.tree_map(lambda x: x.mean(axis=0), info)
            # if len(ep_rws):
            #     info_mean.update(
            #         dict(
            #             ep_rw_mean=sum(ep_rws)
            #             / len(ep_rws)
            #             # rw=sum(r_v),
            #             # mean_v=sum(v_v) / steps,
            #             # steps_s=steps / (et - st),
            #             # mean_return=sum(returns) / steps,
            #             # mean_adv=adv_j.mean(),
            #             # logstd_param=p_params["params"]["logstd"],
            #         )
            #     )
            wandb.log(
                dict(
                    global_steps=total_steps,
                    episodic_return=np.mean(ep_rws),
                    charts_learning_rate=learning_rate,
                    losses_value_loss=info_mean["l_vf"],
                    losses_policy_loss=info_mean["l_clip"],
                    losses_entropy=info_mean["S"],
                    losses_approx_kl=info_mean['approx_kl'],
                    # debug_pg_stop_iter=0,
                )
            )
            st = et
