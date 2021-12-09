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

hp.environment_name = 'VSSGoTo-v3'
hp.total_training_steps = 5100000
hp.gamma = 0.95
hp.batch_size = 256
hp.min_replay_size = 100000
hp.replay_capacity = 5100000
hp.use_ou_noise = True
hp.ou_noise_dt = 25e-3
hp.ou_noise_sigma = 0.2
hp.learning_rate = 1e-4
validation_frequency = 200000

print("Agent Version: -> ", get_agent_version())
print("Agent DDPG Hyper Parameters:")
pprint.pp(dataclasses.asdict(hp))

# Create environment
env = gym.make(hp.environment_name)
val_env = gym.make(hp.environment_name)
val_env = gym.wrappers.RecordVideo(val_env, "./monitor/", step_trigger=lambda x: True)

# Set random seeds
np.random.seed(hp.seed)
env.seed(hp.seed)
env.action_space.seed(hp.seed)
env.observation_space.seed(hp.seed)
val_env.seed(hp.seed)
val_env.action_space.seed(hp.seed)
val_env.observation_space.seed(hp.seed)

# Create agent
agent = AgentDDPG(hp)

# Init wandb logging
wandb.init(
    project="rl-jax-vssgoto",
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
ep_steps = []
ep_step = 0
done = False

for step in tqdm(range(hp.total_training_steps), smoothing=0):
    action, logprob = agent.sample_action(obs)
    _obs, reward, done, step_info = env.step(action)
    ep_rw += reward
    ep_step += 1
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
            metrics.update(dict(mean_ep_steps=np.mean(ep_steps)))
        if ep_info:
            ep_info.pop("TimeLimit.truncated", None)
            metrics.update(ep_info)
            ep_info = None
        wandb.log(metrics)
        ep_rws = []
        ep_steps = []
        step_infos = []

    if done:
        obs = env.reset()
        ep_rws.append(ep_rw)
        ep_steps.append(ep_step)
        ep_rw = 0
        ep_step = 0
        ep_info = step_info
    else:
        obs = _obs
    
    # Run validation episode
    if step % validation_frequency == 0:
        val_obs = val_env.reset()
        val_rw = 0
        val_step = 0
        val_done = False
        while not val_done:
            val_action = agent.policy_fn(agent.policy_params, val_obs)
            val_obs, val_reward, val_done, val_step_info = val_env.step(np.array(val_action))
            val_rw += val_reward
            val_step += 1
        val_step_info.update(dict(validation_return=val_rw, validation_steps=val_step))
        print(val_step_info)


while not done:
    action, logprob = agent.sample_action(obs)
    _obs, reward, done, step_info = env.step(action)
    obs = _obs

env.close()