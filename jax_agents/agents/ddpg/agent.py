import distrax
import gym
import jax
import numpy as np
import optax
from jax_agents.agents.ppo.networks import get_optimizer_step_fn
from jax_agents.agents.ddpg.buffer import ReplayBuffer
from jax_agents.agents.ddpg.hyperparameters import HyperparametersDDPG
from jax_agents.agents.ddpg.loss import get_policy_loss_fn, get_q_value_loss_fn
from jax_agents.agents.ddpg.noise import (
    get_gaussian_noise_fn,
    get_ornstein_uhlenbeck_noise_fn,
)
from jax_agents.agents.ddpg.networks import (
    PolicyModule,
    QValueModule,
    DoubleQValueModule,
    target_params_sync_fn,
)


class AgentDDPG:
    def __init__(self, hyperparameters):
        # Tests
        assert isinstance(hyperparameters, HyperparametersDDPG)

        self.rng = jax.random.PRNGKey(hyperparameters.seed)
        self.rng, policy_key, q_value_key = jax.random.split(self.rng, 3)

        self.hp = hyperparameters
        env = gym.make(self.hp.environment_name)
        self.buffer = ReplayBuffer(
            env.observation_space, env.action_space, self.hp.replay_capacity
        )

        # Networks
        self.policy_model = PolicyModule(env.action_space.shape[0])

        self.q_value_model = (
            DoubleQValueModule() if self.hp.double_q else QValueModule()
        )
        self.policy_params = self.policy_model.init(
            policy_key, env.observation_space.sample()
        )
        self.q_value_params = self.q_value_model.init(
            q_value_key, env.observation_space.sample(), env.action_space.sample()
        )
        self.taget_policy_params = self.policy_params
        self.target_q_value_params = self.q_value_params

        # Optimizers
        optimizer = optax.chain(
            optax.scale_by_adam(),
            optax.scale(-self.hp.learning_rate),
        )
        self.policy_optmizer_state = optimizer.init(self.policy_params)
        self.q_value_optimizer_state = optimizer.init(self.q_value_params)

        # Get functions
        self.optimizer_step = get_optimizer_step_fn(optimizer)
        policy_loss = get_policy_loss_fn(
            policy=self.policy_model,
            q_value=self.q_value_model,
            is_double_q=self.hp.double_q,
        )
        q_value_loss = get_q_value_loss_fn(
            policy=self.policy_model,
            q_value=self.q_value_model,
            gamma=self.hp.gamma,
            is_double_q=self.hp.double_q,
        )
        policy_loss_grad = jax.grad(policy_loss, has_aux=True)
        q_value_loss_grad = jax.grad(q_value_loss, has_aux=True)
        self.batch_policy_loss_grad = jax.vmap(
            policy_loss_grad, in_axes=(None, None, 0)
        )
        self.batch_q_value_loss_grad = jax.vmap(
            q_value_loss_grad, in_axes=(None, None, None, 0, 0, 0, 0, 0)
        )
        if self.hp.use_ou_noise:
            noise = get_ornstein_uhlenbeck_noise_fn(
                env.action_space,
                self.hp.ou_noise_sigma,
                self.hp.ou_noise_theta,
                self.hp.ou_noise_dt,
            )
        else:
            noise = get_gaussian_noise_fn(env.action_space, self.hp.normal_noise_sigma)
        self.noise_state = jax.numpy.zeros(
            env.action_space.shape
        )  # TODO: reseting noise and initial?

        # Jitting
        self.batch_policy_loss_grad = jax.jit(self.batch_policy_loss_grad)
        self.batch_q_value_loss_grad = jax.jit(self.batch_q_value_loss_grad)
        self.optimizer_step = jax.jit(self.optimizer_step)
        self.target_params_update = jax.jit(target_params_sync_fn)
        self.policy_fn = jax.jit(self.policy_model.apply, backend="cpu")
        self.add_action_noise = jax.jit(noise, backend="cpu")

    def observe(
        self, observation, action, action_logprob, reward, done, next_observation
    ):
        "Observe an environment transition, adding it to the rollout buffer"
        self.buffer.add(
            observation, action, action_logprob, reward, done, next_observation
        )

    def update(self):
        if self.buffer.size < self.hp.min_replay_size:
            return None

        info = dict()
        (
            b_observations,
            b_actions,
            b_rewards,
            b_dones,
            b_next_observations,
        ) = self.buffer.get_batch(self.hp.batch_size)

        # Update policy
        (b_policy_grads, info_policy,) = self.batch_policy_loss_grad(
            self.policy_params, self.q_value_params, b_observations
        )
        self.policy_params, self.policy_optmizer_state = self.optimizer_step(
            self.policy_params, b_policy_grads, self.policy_optmizer_state
        )

        # Update q value
        (b_q_value_grads, info_q_value,) = self.batch_q_value_loss_grad(
            self.q_value_params,
            self.taget_policy_params,
            self.target_q_value_params,
            b_observations,
            b_actions,
            b_rewards,
            b_dones,
            b_next_observations,
        )
        self.q_value_params, self.q_value_optimizer_state = self.optimizer_step(
            self.q_value_params, b_q_value_grads, self.q_value_optimizer_state
        )

        # Sync target params
        self.taget_policy_params = self.target_params_update(
            self.policy_params, self.taget_policy_params, self.hp.tau
        )
        self.target_q_value_params = self.target_params_update(
            self.q_value_params, self.target_q_value_params, self.hp.tau
        )

        info.update(info_q_value)
        info.update(info_policy)
        return info

    def sample_action(self, observation):
        self.rng, act_key = jax.random.split(self.rng, 2)
        action = self.policy_fn(self.policy_params, observation)
        action, self.noise_state = self.add_action_noise(
            act_key, self.noise_state, action
        )
        return np.array(action), 1.0

    @staticmethod
    def get_hyperparameters():
        return HyperparametersDDPG()
