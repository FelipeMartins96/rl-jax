import flax.linen as nn
import jax.numpy as jnp
import distrax


class PolicyModule(nn.Module):
    "SAC Paper Actor Network"
    actions_dim: int

    def setup(self):
        self.fc1 = nn.Dense(256)
        self.fc2 = nn.Dense(256)
        self.fc_mean = nn.Dense(self.actions_dim)
        self.fc_logstd = nn.Dense(self.actions_dim)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)

        return mean, log_std

    def evaluate(self, x, rng, epsilon=1e-6):
        mean, log_std = self(x)
        std = jnp.exp(log_std)

        normal = distrax.Normal(mean, std)
        z = normal.sample(seed=rng)
        action = nn.tanh(z)
        log_prob = normal.log_prob(z) - jnp.log(1 - jnp.power(action, 2) + epsilon)

        return action, log_prob, z, mean, log_std

    def get_action(self, x, rng):
        mean, log_std = self(x)
        std = jnp.exp(log_std)

        normal = distrax.Normal(mean, std)
        z = normal.sample(seed=rng)
        action = nn.tanh(z)

        return action


class QValueModule(nn.Module):
    "SAC Paper Critic Network"

    def setup(self):
        self.fc1 = nn.Dense(256)
        self.fc2 = nn.Dense(256)
        self.fc3 = nn.Dense(1)

    def __call__(self, s, a):
        x = jnp.concatenate([s, a], axis=-1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)


class DoubleQValueModule(nn.Module):
    "SAC Paper Critic Network"

    def setup(self):
        self.critic_1 = QValueModule()
        self.critic_2 = QValueModule()

    def __call__(self, s, a):
        return self.critic_1(s, a), self.critic_2(s, a)


class ValueModule(nn.Module):
    "SAC Paper Value Network"

    def setup(self):
        self.fc1 = nn.Dense(256)
        self.fc2 = nn.Dense(256)
        self.fc3 = nn.Dense(1)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)
