import pprint
import dataclasses

import gym
import jax
import wandb
import numpy as np
from tqdm import tqdm

from utils import get_agent_version
from jax_agents.agents import AgentDDPG

import rsoccer_gym

# Get default agent hyperparameters
hp = AgentDDPG.get_hyperparameters()

hp.environment_name = 'VSS-v0'
hp.total_training_steps = 5000000
hp.gamma = 0.95
hp.min_replay_size = 250000

print("Agent Version: -> ", get_agent_version())
print("Agent DDPG Hyper Parameters:")
pprint.pp(dataclasses.asdict(hp))

# Create environment
env = gym.make(hp.environment_name)
env = gym.wrappers.RecordVideo(env, "./monitor/", step_trigger=lambda x: x % 50000 == 0)

# Set random seeds
np.random.seed(hp.seed)
env.seed(hp.seed)
env.action_space.seed(hp.seed)
env.observation_space.seed(hp.seed)

# Create agent
agent = AgentDDPG(hp)

# Init wandb logging
wandb.init(
    project="jax_agents-rsoccer",
    entity="felipemartins",
    monitor_gym=True,
    save_code=True,
    config=dict(
        algorithm=hp.algorithm_name,
        agent_version=get_agent_version(),
        env=hp.environment_name,
    ),
)
ep_info = None
# Pre training loop variables
obs = env.reset()
ep_rw = 0
ep_rws = []

for step in tqdm(range(hp.total_training_steps), smoothing=0):
    action, logprob = agent.sample_action(obs)
    _obs, reward, done, step_info = env.step(action)
    ep_rw += reward
    terminal_state = False if not done or "TimeLimit.truncated" in step_info else True
    agent.observe(obs, action, logprob, reward, terminal_state, _obs)
    update_info = agent.update()

    if update_info and len(ep_rws):
        metrics = {}
        info_mean = jax.tree_map(lambda x: x.mean(axis=0), update_info)
        metrics.update(
            dict(
                global_steps=step,
                losses_value_loss=info_mean["agent/q_value_loss"],
                losses_policy_loss=info_mean["agent/policy_loss"],
            )
        )
        if len(ep_rws):
            metrics.update(dict(episodic_return=np.mean(ep_rws)))
        if ep_info:
            ep_info.pop("TimeLimit.truncated", None)
            metrics.update(ep_info)
            ep_info = None
        wandb.log(metrics)
        ep_rws = []
        step_infos = []

    if done:
        obs = env.reset()
        ep_rws.append(ep_rw)
        ep_rw = 0
        ep_info = step_info
    else:
        obs = _obs
