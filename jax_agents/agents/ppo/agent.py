from jax_agents.agents.ppo.hyperparameters import HyperparametersPPO


class AgentPPO:
    def __init__(self, hyperparameters):
        # Tests
        assert isinstance(hyperparameters, HyperparametersPPO)

        self.hp = hyperparameters

    def observe(self):
        pass

    def update(self):
        pass

    def sample_action(self):
        pass

    @staticmethod
    def get_hyperparameters():
        return HyperparametersPPO()
