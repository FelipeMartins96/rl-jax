from jax_agents.agents.ppo.hyperparameters import HyperparametersPPO
from jax_agents.agents.ppo.buffer import RolloutBuffer

import gym


class AgentPPO:
    def __init__(self, hyperparameters):
        # Tests
        assert isinstance(hyperparameters, HyperparametersPPO)

        self.hp = hyperparameters
        env = gym.make(self.hp.environment_name)
        self.buffer = RolloutBuffer(
            env.observation_space, env.action_space, self.hp.rollout_steps
        )

    def observe(
        self, observation, action, action_logprob, reward, done, next_observation
    ):
        "Observe an environment transition, adding it to the rollout buffer"
        self.buffer.add(
            observation, action, action_logprob, reward, done, next_observation
        )

    def update(self):
        rollout = get_rollout()
        gaes = calculate_gae(rollout)
        batch = shuffle_gaes(gaes)

        for i in range(update_epochs):
            for j in range(n_minibatches):
                grads, info = ppo_grad(minibatch)
                update_grads(grads)
        
        return info

    def sample_action(self):
        pass

    @staticmethod
    def get_hyperparameters():
        return HyperparametersPPO()
