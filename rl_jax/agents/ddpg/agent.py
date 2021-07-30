import gym
import jax
from jax.interpreters.xla import Backend
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from rl_jax.agents.base import Action, Agent
from rl_jax.networks.ddpg import DDPGActor, DDPGCritic
from rl_jax.replay import Batch, Transition
from rl_jax.replay.simple_replay import ReplayBuffer


class DDPG(Agent):
    def __init__(
        self,
        env: gym.Env,
        rng_key: jnp.ndarray,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        critic_weight_decay: float = 1e-2,
        replay_capacity: int = int(1e6),
        min_replay_size: int = 64,
        batch_size: int = 64,
        ou_sigma: float = 0.2,
        ou_theta: float = 0.15,
        gamma: float = 0.99,
        tau: float = 1e-3,
    ):
        # RNG keys
        _, actor_key, critic_key, ou_key = jax.random.split(rng_key, 4)
        self._ou_key = ou_key

        # Hyperparameters
        self._replay_capacity = replay_capacity
        self._min_replay_size = min_replay_size
        self._batch_size = batch_size
        self._ou_sigma = ou_sigma
        self._ou_theta = ou_theta
        self._gamma = gamma
        self._tau = tau

        # Get env sample state and actions
        obs_sample = env.observation_space.sample()
        act_sample = env.action_space.sample()

        # Initialize Buffer and OU noise state
        self._replay = ReplayBuffer(env, self._replay_capacity)
        self._ou_state = jnp.zeros(env.action_space.shape)

        # Neural Networks
        actor_model = DDPGActor(action_dim=env.action_space.shape[0])
        actor_forward = jax.jit(actor_model.apply)
        actor_params = actor_model.init(actor_key, obs_sample)
        actor_tx = optax.adam(actor_lr)
        actor_opt_state = actor_tx.init(actor_params)

        critic_model = DDPGCritic()
        critic_forward = jax.jit(critic_model.apply)
        critic_tx = optax.adamw(critic_lr, weight_decay=critic_weight_decay)
        critic_params = critic_model.init(critic_key, obs_sample, act_sample)
        critic_opt_state = critic_tx.init(critic_params)

        # Train State
        self._actor = TrainState(
            step=0,
            apply_fn=actor_forward,
            params=actor_params,
            tx=actor_tx,
            opt_state=actor_opt_state,
        )
        self._actor_tgt = TrainState(
            step=0,
            apply_fn=actor_forward,
            params=actor_params,
            tx=None,
            opt_state=None,
        )
        self._critic = TrainState(
            step=0,
            apply_fn=critic_forward,
            params=critic_params,
            tx=critic_tx,
            opt_state=critic_opt_state,
        )
        self._critic_tgt = TrainState(
            step=0,
            apply_fn=critic_forward,
            params=critic_params,
            tx=None,
            opt_state=None,
        )

    def update(self):
        if self._replay.size < self._min_replay_size:
            return None

        batch = self._replay.sample(self._batch_size)
        new_states, info = DDPG._update(
            self._actor,
            self._critic,
            self._actor_tgt,
            self._critic_tgt,
            batch,
        )
        self._actor, self._critic = new_states

        self._actor_tgt = DDPG._target_sync(self._actor, self._actor_tgt, self._tau)
        self._critic_tgt = DDPG._target_sync(self._critic, self._critic_tgt, self._tau)

        return info

    def select_action(self, observation: jnp.ndarray, add_noise: bool = True) -> Action:
        if add_noise:
            self._ou_key, key = jax.random.split(self._ou_key)
            action, self._ou_state = DDPG._select_action(
                self._actor,
                observation,
                key,
                self._ou_state,
                self._ou_sigma,
                self._ou_theta,
            )
        else:
            action = self._actor.apply_fn(self._actor.params, observation)
        return action.copy()

    def observe(self, state, action, reward, done, next_state, info):
        if not done or "TimeLimit.truncated" in info:
            discount = self._gamma
        else:
            discount = 0.0

        self._replay.add(
            Transition(
                state=state,
                action=action,
                reward=reward,
                discount=discount,
                next_state=next_state,
            )
        )

    @jax.jit
    def _update(
        actor: TrainState,
        critic: TrainState,
        actor_tgt: TrainState,
        critic_tgt: TrainState,
        batch: Batch,
    ):
        # Set target values
        acts_tp1 = actor_tgt.apply_fn(actor_tgt.params, batch.next_states)
        qs_tp1 = critic_tgt.apply_fn(critic_tgt.params, batch.next_states, acts_tp1)
        targets = batch.rewards + batch.discounts * qs_tp1

        # Define critic loss function
        def crt_loss_fn(critic_params):
            qs = critic.apply_fn(critic_params, batch.states, batch.actions)
            td_loss = qs - targets
            crt_loss = jnp.mean(td_loss ** 2)
            return crt_loss, {
                "agent/critic_loss": crt_loss,
                "agent/mean_batch_td_loss": td_loss.mean(),
                "agent/mean_batch_q": qs.mean(),
            }

        # Calculate critic loss grad and update params
        grad_critic, critic_info = jax.grad(crt_loss_fn, has_aux=True)(critic.params)
        new_critic = critic.apply_gradients(grads=grad_critic)

        # define actor loss
        def act_loss_fn(actor_params):
            acts_new = actor.apply_fn(actor_params, batch.states)
            qs = critic.apply_fn(critic.params, batch.states, acts_new)
            actor_loss = -qs.mean()
            return actor_loss, {"agent/actor_loss": actor_loss}

        # Calculate actor lossgrad and update params
        grad_actor, act_info = jax.grad(act_loss_fn, has_aux=True)(actor.params)
        new_actor = actor.apply_gradients(grads=grad_actor)

        return (new_actor, new_critic), {
            "agent/batch_reward_mean": batch.rewards.mean(),
            **act_info,
            **critic_info,
        }

    @jax.jit
    def _target_sync(
        state: TrainState,
        tgt_state: TrainState,
        tau: float,
    ):
        new_target_params = jax.tree_multimap(
            lambda p, tp: p * tau + tp * (1 - tau), state.params, tgt_state.params
        )

        return tgt_state.replace(params=new_target_params)

    @jax.partial(jax.jit, static_argnums=(4, 5))
    def _select_action(actor, observation, key, ou_state, sigma, theta):
        action = actor.apply_fn(actor.params, observation)
        ou_state = (1.0 - theta) * ou_state + jax.random.normal(
            key, shape=action.shape
        ) * sigma
        return jnp.clip(action + ou_state, -1, 1), ou_state
