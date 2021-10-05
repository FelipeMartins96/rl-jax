import gym
import jax
import numpy as np
import optax
import distrax
from jax_agents.agents.ppo.buffer import RolloutBuffer
from jax_agents.agents.ppo.generalized_advantage_estimate import get_calculate_gae_fn
from jax_agents.agents.ppo.hyperparameters import HyperparametersPPO
from jax_agents.agents.ppo.loss import get_ppo_loss_fn
from jax_agents.agents.ppo.networks import (
    PolicyModule,
    ValueModule,
    get_optimizer_step_fn,
)


class AgentPPO:
    def __init__(self, hyperparameters):
        # Tests
        assert isinstance(hyperparameters, HyperparametersPPO)
        self.rng = jax.random.PRNGKey(hyperparameters.seed)
        self.rng, policy_key, value_key = jax.random.split(self.rng, 3)

        self.hp = hyperparameters
        env = gym.make(self.hp.environment_name)
        self.buffer = RolloutBuffer(
            env.observation_space, env.action_space, self.hp.n_rollout_steps
        )

        # Networks
        self.policy_model = PolicyModule(env.action_space.shape[0])
        self.value_model = ValueModule()
        self.policy_params = self.policy_model.init(
            policy_key, env.observation_space.sample()
        )
        self.value_params = self.value_model.init(
            value_key, env.observation_space.sample()
        )

        # Optimizers
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.hp.maximum_gradient_norm),
            optax.scale_by_adam(eps=self.hp.adam_epsilon),
            optax.scale(-self.hp.learning_rate),
        )
        self.policy_optmizer_state = optimizer.init(self.policy_params)
        self.value_optimizer_state = optimizer.init(self.value_params)
        self.optimizer_step = get_optimizer_step_fn(optimizer)

        # Get functions
        self.calculate_gae = get_calculate_gae_fn(
            self.value_model, self.hp.gamma, self.hp.gae_lambda, self.hp.n_rollout_steps
        )
        ppo_loss = get_ppo_loss_fn(
            self.policy_model,
            self.value_model,
            self.hp.clip_coefficient,
            self.hp.value_loss_coefficient,
            self.hp.entropy_loss_coefficient,
        )
        ppo_loss_grad = jax.grad(ppo_loss, argnums=(0, 1), has_aux=True)
        self.batch_ppo_loss_grad = jax.vmap(
            ppo_loss_grad, in_axes=(None, None, 0, 0, 0, 0, 0, 0)
        )

        # Jitting
        self.batch_ppo_loss_grad = jax.jit(self.batch_ppo_loss_grad)
        self.policy_fn = jax.jit(self.policy_model.apply, backend="cpu")
        self.optimizer_step = jax.jit(self.optimizer_step)

    def observe(
        self, observation, action, action_logprob, reward, done, next_observation
    ):
        "Observe an environment transition, adding it to the rollout buffer"
        self.buffer.add(
            observation, action, action_logprob, reward, done, next_observation
        )

    def update(self):
        if self.buffer.size < self.hp.n_rollout_steps:
            return None

        rollout = self.buffer.get_rollout()
        gae_rollout = self.calculate_gae(self.value_params, rollout)

        indexes = np.arange(self.hp.n_rollout_steps)
        minibatch_size = int(self.hp.n_rollout_steps // self.hp.n_mini_batches)

        (
            b_observations,
            b_actions,
            b_logprobs,
            b_returns,
            b_advantages,
            b_values,
        ) = gae_rollout

        for epoch in range(self.hp.update_epochs):
            np.random.shuffle(indexes)
            for mb_start in range(0, self.hp.n_rollout_steps, minibatch_size):
                mb_end = mb_start + minibatch_size
                mb_indexes = indexes[mb_start:mb_end]

                mb_advantages = b_advantages[mb_indexes]
                mb_advantages = (mb_advantages - np.mean(mb_advantages)) / (
                    np.std(mb_advantages) + 1e-8
                )

                (policy_grad, value_grad), info = self.batch_ppo_loss_grad(
                    self.policy_params,
                    self.value_params,
                    b_observations[mb_indexes],
                    b_actions[mb_indexes],
                    b_logprobs[mb_indexes],
                    b_returns[mb_indexes],
                    b_values[mb_indexes],
                    mb_advantages,
                )

                self.policy_params, self.policy_optmizer_state = self.optimizer_step(
                    self.policy_params, policy_grad, self.policy_optmizer_state
                )
                self.value_params, self.value_optimizer_state = self.optimizer_step(
                    self.value_params, value_grad, self.value_optimizer_state
                )

        return info

    def sample_action(self, observation):
        self.rng, act_key = jax.random.split(self.rng, 2)
        mean, sigma = self.policy_fn(self.policy_params, observation)
        distribution = distrax.MultivariateNormalDiag(mean, sigma)
        action = distribution.sample(seed=act_key)
        logprob = distribution.log_prob(action)

        return np.array(action), np.array(logprob)

    @staticmethod
    def get_hyperparameters():
        return HyperparametersPPO()
