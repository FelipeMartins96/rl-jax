from jax_agents.agents import AgentPPO
import gym
from tqdm import tqdm
import numpy as np
import jax
import wandb
from utils import get_agent_version
import pprint
import dataclasses

# Get defaul
hp = AgentPPO.get_hyperparameters()

print("Agent Version: -> ", get_agent_version())
print("Agent PPO Hyper Parameters:")
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
agent = AgentPPO(hp)

# Init wandb logging
wandb.init(
    project="jax_agents",
    entity="felipemartins",
    monitor_gym=True,
    save_code=True,
    config=dict(algorithm=hp.algorithm_name, agent_version=get_agent_version(), env=hp.environment_name),
)

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

    if update_info:
        info_mean = jax.tree_map(lambda x: x.mean(axis=0), update_info)
        wandb.log(
            dict(
                global_steps=step,
                episodic_return=np.mean(ep_rws),
                losses_value_loss=info_mean["l_vf"],
                losses_policy_loss=info_mean["l_clip"],
                losses_entropy=info_mean["S"],
                losses_approx_kl=info_mean["approx_kl"],
            )
        )
        ep_rws = []

    if done:
        obs = env.reset()
        ep_rws.append(ep_rw)
        ep_rw = 0
    else:
        obs = _obs
