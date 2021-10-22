import functools
from typing import Callable, Dict, Tuple, Union
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
from flax.training.train_state import TrainState
from rl_jax.agents.base import Action, Agent
from rl_jax.networks.sac_networks import ActorSAC, DoubleCriticSAC, ValueSAC
from rl_jax.replay import Batch, Transition
from rl_jax.replay.simple_replay import ReplayBuffer


class ActorTrainState(TrainState):
    evaluate: Callable = flax.struct.field(pytree_node=False)
    get_action: Callable = flax.struct.field(pytree_node=False)


def update_q(critic, value_tgt, batch):
    next_v = value_tgt.apply_fn(value_tgt.params, batch.next_states)

    target_q = batch.rewards + batch.discounts * next_v

    def critic_loss_fn(critic_params):
        q1, q2 = critic.apply_fn(critic_params, batch.states, batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "q1": q1.mean(),
            "q2": q2.mean(),
        }

    grads_critc, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads_critc)

    return new_critic, info


def update_actor(rng, actor, critic, batch, temperature):
    def actor_loss_fn(actor_params):
        action, log_prob, _, _, _ = actor.evaluate(actor_params, batch.states, rng)
        q1, q2 = critic.apply_fn(critic.params, batch.states, action)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_prob * temperature - q).mean()
        return actor_loss, {"actor_loss": actor_loss, "entropy": -log_prob.mean()}

    grads_actor, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads_actor)

    return new_actor, info


def update_v(rng, actor, critic, value, batch):
    action, log_prob, _, _, _ = actor.evaluate(actor.params, batch.states, rng)
    q1, q2 = critic.apply_fn(critic.params, batch.states, action)
    target_v = jnp.minimum(q1, q2)

    def value_loss_fn(value_params):
        v = value.apply_fn(value_params, batch.states)
        value_loss = ((v - target_v) ** 2).mean()
        return value_loss, {
            "value_loss": value_loss,
            "v": v.mean(),
        }

    grads_value, info = jax.grad(value_loss_fn, has_aux=True)(value.params)
    new_value = value.apply_gradients(grads=grads_value)

    return new_value, info


def target_sync(
    state: TrainState,
    tgt_state: TrainState,
    tau: float,
) -> TrainState:
    """Soft target network update."""

    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), state.params, tgt_state.params
    )

    return tgt_state.replace(params=new_target_params)


@jax.partial(jax.jit, static_argnums=(6, 7))
def update(rng, actor, critic, value, value_tgt, batch, tau, temperature):

    new_critic, critic_info = update_q(critic, value_tgt, batch)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, batch, temperature)

    rng, key = jax.random.split(rng)
    new_value, value_info = update_v(key, new_actor, new_critic, value, batch)

    new_target_value = target_sync(new_value, value_tgt, tau)

    return (
        rng,
        new_actor,
        new_critic,
        new_value,
        new_target_value,
        {
            **critic_info,
            **value_info,
            **actor_info,
        },
    )


class SAC(Agent):
    """Soft Actor-Critic (SAC) Agent

    tries to replicate the paper experimental details.
    Soft Actor-Critic:
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
    HAARNOJA, Tuomas et al. 2018
    (http://proceedings.mlr.press/v80/haarnoja18b)
    """

    def __init__(
        self,
        env: gym.Env,
        rng_key: jnp.ndarray,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        replay_capacity: int = int(1e6),
        min_replay_size: int = 256,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 5e-3,
        temperature: float = 1.0,
    ):
        # RNG keys
        self.rng, actor_key, critic_key, value_key = jax.random.split(rng_key, 4)

        # Hyperparameters
        self._replay_capacity = replay_capacity
        self._min_replay_size = min_replay_size
        self._batch_size = batch_size
        self._gamma = gamma
        self._tau = tau
        self._temperature = temperature

        # Get env sample state and actions
        obs_sample = env.observation_space.sample()
        act_sample = env.action_space.sample()

        # Initialize Buffer
        self._replay = ReplayBuffer(env, self._replay_capacity)

        # Neural Networks
        actor_model = ActorSAC(actions_dim=env.action_space.shape[0])
        self._actor, self._critic, self._value, self._value_tgt = SAC._init_nns(
            actor_model=actor_model,
            obs_sample=obs_sample,
            act_sample=act_sample,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            value_lr=value_lr,
            actor_key=actor_key,
            critic_key=critic_key,
            value_key=value_key,
        )

        self._actor_get_action = jax.jit(self._actor.get_action)

    def update(self) -> Union[Dict, None]:
        """Update the agent."""

        if self._replay.size < self._min_replay_size:
            return None

        batch = self._replay.sample(self._batch_size)

        (
            self.rng,
            self._actor,
            self._critic,
            self._value,
            self._value_tgt,
            info,
        ) = update(
            self.rng,
            self._actor,
            self._critic,
            self._value,
            self._value_tgt,
            batch,
            self._tau,
            self._temperature,
        )

        return info

    def select_action(self, observation: jnp.ndarray, add_noise: bool = True) -> Action:
        """Select an action."""
        if add_noise:
            action = self._actor_get_action(self._actor.params, observation, self.rng)
        else:
            action, _ = self._actor.apply_fn(self._actor.params, observation, self.rng)
        return action.copy()

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        next_state: np.ndarray,
        info: Dict,
    ) -> None:
        """Add a transition to the replay buffer."""

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

    @staticmethod
    def _init_nns(
        actor_model: flax.linen.Module,
        obs_sample: np.ndarray,
        act_sample: np.ndarray,
        actor_lr: float,
        critic_lr: float,
        value_lr: float,
        actor_key: jnp.ndarray,
        critic_key: jnp.ndarray,
        value_key: jnp.ndarray,
    ) -> Tuple[TrainState, TrainState, TrainState, TrainState]:
        """Initialize Neural Networks"""

        actor_params = actor_model.init(actor_key, obs_sample)
        actor_tx = optax.adam(actor_lr)
        actor_opt_state = actor_tx.init(actor_params)

        critic_model = DoubleCriticSAC()
        critic_tx = optax.adam(critic_lr)
        critic_params = critic_model.init(critic_key, obs_sample, act_sample)
        critic_opt_state = critic_tx.init(critic_params)

        value_model = ValueSAC()
        value_params = value_model.init(value_key, obs_sample)
        value_tx = optax.adam(value_lr)
        value_opt_state = value_tx.init(value_params)

        # Train State
        actor = ActorTrainState(
            step=0,
            apply_fn=actor_model.apply,
            params=actor_params,
            tx=actor_tx,
            opt_state=actor_opt_state,
            evaluate=functools.partial(actor_model.apply, method=actor_model.evaluate),
            get_action=functools.partial(
                actor_model.apply, method=actor_model.get_action
            ),
        )
        critic = TrainState(
            step=0,
            apply_fn=critic_model.apply,
            params=critic_params,
            tx=critic_tx,
            opt_state=critic_opt_state,
        )

        value = TrainState(
            step=0,
            apply_fn=value_model.apply,
            params=value_params,
            tx=value_tx,
            opt_state=value_opt_state,
        )

        value_tgt = TrainState(0, value_model.apply, value_params, None, None)

        return actor, critic, value, value_tgt
