import pprint
import dataclasses

import gym
import jax
import flax.training.checkpoints
import wandb
import numpy as np
from tqdm import tqdm

from utils import get_agent_version
from jax_agents.agents import AgentDDPG

import rsoccer_gym

import argparse

parser = argparse.ArgumentParser()

# get experiment if from arguments
parser.add_argument('-e', "--experiment", type=int, required=True)
args = parser.parse_args()

load_worker = True
exp_name = None
ablation = {}
gamma = 0.95

if args.experiment == 0:
    env_name = 'VSSGoToHRL-v0'
    n_steps = 3100000
    exp_name = "non hrl, no energy, no move, new net layers, v0"
    ablation = {'man_w_energy': 0, 'man_w_move': 0}
elif args.experiment == 1:
    env_name = 'VSSGoToHRL-v1'
    n_steps = 6100000
    exp_name = "non hrl, no energy, no move, new net layers, v1"
    ablation = {'man_w_energy': 0, 'man_w_move': 0}


# Get manager agent hyperparameters
man_hp = AgentDDPG.get_hyperparameters()

man_hp.environment_name = env_name
env = gym.make(man_hp.environment_name, **ablation)

man_hp.total_training_steps = n_steps
man_hp.gamma = gamma
man_hp.batch_size = 256
man_hp.min_replay_size = 100000
man_hp.replay_capacity = man_hp.total_training_steps
man_hp.learning_rate = 1e-4
man_hp.custom_env_space = True
man_hp.action_space = gym.spaces.Box(
    low=-1, high=1, shape=env.action_space.sample()[1].shape, dtype=np.float32
)
man_hp.observation_space = gym.spaces.Box(
    low=env.observation_space.low,
    high=env.observation_space.high,
    shape=env.observation_space.shape,
    dtype=env.observation_space.dtype,
)
validation_frequency = 150000

print("Agent Version: -> ", get_agent_version())
print("Agent DDPG Hyper Parameters:")
pprint.pp(dataclasses.asdict(man_hp))

# Create environment
val_env = gym.make(man_hp.environment_name)
val_env = gym.wrappers.RecordVideo(val_env, "./monitor/", step_trigger=lambda x: True)

# Set random seeds
np.random.seed(man_hp.seed)
env.seed(man_hp.seed)
env.action_space.seed(man_hp.seed)
env.observation_space.seed(man_hp.seed)
val_env.seed(man_hp.seed)
val_env.action_space.seed(man_hp.seed)
val_env.observation_space.seed(man_hp.seed)

# Create agents
man_agent = AgentDDPG(man_hp)

# Init wandb logging
wandb.init(
    project="rl-jax-hierarchical-vss",
    entity="felipemartins",
    monitor_gym=True,
    save_code=True,
    name=exp_name,
    config=dict(
        algorithm=man_hp.algorithm_name,
        agent_version=get_agent_version(),
        env=man_hp.environment_name,
    ),
)
ep_info = None
# Pre training loop variables
obs = env.reset()
man_ep_rw = 0
man_ep_rws = []
ep_steps = []
ep_step = 0
done = False
fake_targets = env.action_space.high[0]

for step in tqdm(range(man_hp.total_training_steps), smoothing=0):
    # Get manager action (Target point)
    man_action, logprob = man_agent.sample_action(obs)

    # Join actions and step environment
    a = np.stack([fake_targets, man_action])
    _obs, rewards, done, step_info = env.step(a)
    man_ep_rw += rewards[0]
    ep_step += 1

    terminal_state = False if not done or "TimeLimit.truncated" in step_info else True

    man_agent.observe(obs, man_action, logprob, rewards[0], terminal_state, _obs)

    man_update_info = man_agent.update()

    if man_update_info and len(man_ep_rws):
        metrics = {}

        man_info_mean = jax.tree_map(lambda x: x.mean(axis=0), man_update_info)
        metrics.update(
            dict(
                global_steps=step,
                manager_losses_value_loss=man_info_mean["agent/q_value_loss"],
                manager_losses_policy_loss=man_info_mean["agent/policy_loss"],
            )
        )

        if len(man_ep_rws):
            metrics.update(dict(manager_episodic_return=np.mean(man_ep_rws)))
            metrics.update(dict(mean_ep_steps=np.mean(ep_steps)))
        if ep_info:
            ep_info.pop("TimeLimit.truncated", None)
            metrics.update(ep_info)
            ep_info = None
        wandb.log(metrics)
        man_ep_rws = []
        wor_ep_rws = []
        ep_steps = []

    if done:
        obs = env.reset()
        man_ep_rws.append(man_ep_rw)
        ep_steps.append(ep_step)
        man_ep_rw = 0
        ep_step = 0
        ep_info = step_info
    else:
        obs = _obs

    # Run validation episode
    if step % validation_frequency == 0:
        val_obs = val_env.reset()
        val_man_ep_rw = 0
        val_wor_ep_rw = 0
        val_step = 0
        val_done = False
        while not val_done:
            # Get manager action (Target point)
            val_man_action = man_agent.policy_fn(man_agent.policy_params, val_obs)

            a = np.stack([fake_targets, val_man_action])
            val_obs, val_rewards, val_done, val_step_info = val_env.step(a)
            val_man_ep_rw += val_rewards[0]
            val_step += 1

        val_step_info.update(
            dict(validation_manager_return=val_man_ep_rw, validation_steps=val_step)
        )
        print(val_step_info)


while not done:
    _obs, reward, done, step_info = env.step(env.action_space.sample()[1])
    obs = _obs

env.close()
