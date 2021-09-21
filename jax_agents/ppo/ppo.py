import time

import distrax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.config import config
import tqdm

import wandb

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

        S = jnp.expand_dims(dist.entropy(), 0)

        l1 = -l_clip
        l2 = c1 * l_vf
        l3 = -c2 * S


        loss = l1 + l2 + l3

        info = dict(l_clip=l1, l_vf=l2, S=l3, loss=loss)

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
        mean = nn.tanh(nn.Dense(self.action_dims, kernel_init=nn.initializers.orthogonal(0.01))(x))
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
        values = get_v_vmap(v_params, np.concatenate([observations, next_observation], axis=0))
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * lastgaelam
            lastgaelam = advantages[t]
            returns[t] = advantages[t] + values[t]

        return  (
            jnp.array(observations),
            jnp.array(actions),
            jnp.array(logprobs),
            jnp.array(returns),
            jnp.array(advantages),
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
    env = gym.wrappers.RecordVideo(gym.make("MountainCarContinuous-v0"), "./monitor/")
    wandb.init(project="rl-jax", entity="felipemartins", monitor_gym=True, save_code=True)
    np.random.seed(0)
    env.seed(0)
    env.action_space.seed(0)
    env.observation_space.seed(0)
    rng = jax.random.PRNGKey(0)
    epsilon = 0.1
    c1 = 0.25
    c2 = 0.01
    gamma = 0.99
    update_epochs = 3
    learning_rate = 3e-4
    max_grad_norm = 0.5
    num_updates = int(1e5)
    num_steps = 512
    gae_lambda = 0.95
    num_envs = 1
    n_minibatches = 32
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // n_minibatches

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
    for update_index in tqdm.tqdm(range(num_updates), smoothing=0.1):

        buffer.clear()
        ep_rws = []

        # Get Rollout
        for step_index in range(num_steps):
            total_steps += 1

            obs = next_obs

            rng, a_key = jax.random.split(rng, 2)
            a, logprob = sample_action(a_key, p_params, obs)
            # Scaling to environment, assumes env action_space is simmetric around 0
            clipped_a = np.clip(
                a * env.action_space.high, env.action_space.low, env.action_space.high
            )

            next_obs, reward, done, info = env.step(np.array(clipped_a))
            ep_rw += reward

            buffer.add(obs, a, logprob, reward, done, next_obs)

            if done:
                next_obs = env.reset()
                ep_rws.append(ep_rw)
                ep_rw = 0

        # Calculate Advantages
        s_j, a_j, lp_j, r_j, adv_j = calculate_gae(v_params, buffer.get_rollout())


        # Update Networks

        for i in range(update_epochs):
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                # Normalize advantages
                adv_mb = adv_j[start:end]
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

                (p_grad, v_grad), info = ppo_loss_grad_vmap(
                    p_params, v_params, s_j[start:end], a_j[start:end], lp_j[start:end], r_j[start:end], adv_mb
                )

                p_params, p_opt_state = optim_update_step(p_params, p_grad, p_opt_state)
                v_params, v_opt_state = optim_update_step(v_params, v_grad, v_opt_state)


        # Logging
        et = time.time()
        info_mean = jax.tree_map(lambda x: x.mean(axis=0), info)
        if len(ep_rws):
            info_mean.update(
                dict(
                    ep_rw_mean=sum(ep_rws)
                    / len(ep_rws)
                    # rw=sum(r_v),
                    # mean_v=sum(v_v) / steps,
                    # steps_s=steps / (et - st),
                    # mean_return=sum(returns) / steps,
                    # mean_adv=adv_j.mean(),
                    # logstd_param=p_params["params"]["logstd"],
                )
            )
        wandb.log(info_mean)
        st = et
