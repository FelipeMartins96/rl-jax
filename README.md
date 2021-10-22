# JAX/FLAX Reinforcement Learning Agents

I created this repo in an attempt to learn JAX by implementing reinforcement learning algorithms made to work with openAI gym environments, I hope to use those in my Masters research experiments. I am also using this repo to experiment on git management approaches.

# Running experiments
To run the agents make an pip editable install:
```
pip install -e .
```

# Some repositories I use as reference:
- https://github.com/deepmind/bsuite
- https://github.com/deepmind/rlax
- https://github.com/ikostrikov/jaxrl
- https://github.com/deepmind/acme
- https://github.com/google/dopamine
- https://github.com/vwxyzjn/cleanrl

# Implemented Agents
- DDPG
- PPO

# Changelog

0.0.2:
- [#5](https://github.com/FelipeMartins96/rl-jax/pull/5), [#6](https://github.com/FelipeMartins96/rl-jax/pull/6) - [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) Agent, with code level optimizations based on [costa.sh](https://costa.sh/blog-the-32-implementation-details-of-ppo.html) blog post.
- [#8](https://github.com/FelipeMartins96/rl-jax/pull/8), [#9](https://github.com/FelipeMartins96/rl-jax/pull/9) - Changed agents structure.
- Created an experiments folder, and made the agents installable for running and testing experiments on editable installs

0.0.1:
- [#1](https://github.com/FelipeMartins96/rl-jax/pull/1) - Deep Determinist Policy Gradient Agent, with configuration based on the paper [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971);
- [#2](https://github.com/FelipeMartins96/rl-jax/pull/2) - Add ddpg train script with wandb stats and video logging.