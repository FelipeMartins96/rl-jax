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

load_worker = True

# Get manager agent hyperparameters
man_hp = AgentDDPG.get_hyperparameters()

man_hp.environment_name = "VSSGoToHRL-v0"
env = gym.make(man_hp.environment_name)

man_hp.total_training_steps = 3100000
man_hp.gamma = 0.95
man_hp.batch_size = 256
man_hp.min_replay_size = 100000
man_hp.replay_capacity = man_hp.total_training_steps
man_hp.learning_rate = 1e-4
man_hp.custom_env_space = True
man_hp.action_space = gym.spaces.Box(
    low=-1, high=1, shape=env.action_space.sample()[0].shape, dtype=np.float32
)
man_hp.observation_space = gym.spaces.Box(
    low=env.observation_space.low,
    high=env.observation_space.high,
    shape=env.observation_space.shape,
    dtype=env.observation_space.dtype,
)
validation_frequency = 150000

# Get manager agent hyperparameters
wor_hp = AgentDDPG.get_hyperparameters()

wor_hp.environment_name = "VSSGoToHRL-v0"
wor_hp.total_training_steps = man_hp.total_training_steps
wor_hp.gamma = 0.95
wor_hp.batch_size = 256
wor_hp.min_replay_size = 100000
wor_hp.replay_capacity = man_hp.replay_capacity
wor_hp.use_ou_noise = True
wor_hp.ou_noise_dt = 25e-3
wor_hp.ou_noise_sigma = 0.2
wor_hp.learning_rate = 1e-4
wor_hp.custom_env_space = True
wor_hp.action_space = gym.spaces.Box(
    low=-1, high=1, shape=env.action_space.sample()[1].shape, dtype=np.float32
)
wor_hp.observation_space = gym.spaces.Box(
    low=env.observation_space.low[0],
    high=env.observation_space.high[0],
    shape=np.concatenate(
        [env.observation_space.sample(), man_hp.action_space.sample()]
    ).shape,
    dtype=env.observation_space.dtype,
)

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
wor_agent = AgentDDPG(wor_hp)
if load_worker:
    wor_agent.policy_params = flax.training.checkpoints.restore_checkpoint(
        'checkpoints/', 
        wor_agent.policy_params, 
        prefix='goto_worker_policy'
    )

# Init wandb logging
wandb.init(
    project="rl-jax-hierarchical-vss",
    entity="felipemartins",
    monitor_gym=True,
    save_code=True,
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
wor_ep_rw = 0
wor_ep_rws = []
ep_steps = []
ep_step = 0
done = False

for step in tqdm(range(man_hp.total_training_steps), smoothing=0):
    # Get manager action (Target point)
    man_action, logprob = man_agent.sample_action(obs)
    # Get Worker action (wheel speeds)
    wor_obs = np.concatenate([obs, man_action])
    wor_action, logprob = wor_agent.sample_action(wor_obs)
    
    # Join actions and step environment
    action = np.stack([man_action, wor_action])
    _obs, rewards, done, step_info = env.step(action)
    man_ep_rw += rewards[0]
    wor_ep_rw += rewards[1]
    ep_step += 1

    terminal_state = False if not done or "TimeLimit.truncated" in step_info else True
    
    man_agent.observe(obs, man_action, logprob, rewards[0], terminal_state, _obs)

    if load_worker:
        wor_obs2 = np.concatenate([_obs, man_action])
        wor_agent.observe(wor_obs, wor_action, logprob, rewards[1], False, wor_obs2)
    
    man_update_info = man_agent.update()

    if load_worker:
        wor_update_info = wor_agent.update()

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

        if load_worker:
            wor_info_mean = jax.tree_map(lambda x: x.mean(axis=0), wor_update_info)
            metrics.update(
                worker_losses_value_loss=wor_info_mean["agent/q_value_loss"],
                worker_losses_policy_loss=wor_info_mean["agent/policy_loss"],
            )
        
        if len(man_ep_rws):
            metrics.update(dict(manager_episodic_return=np.mean(man_ep_rws)))
            metrics.update(dict(worker_episodic_return=np.mean(wor_ep_rws)))
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
        wor_ep_rws.append(wor_ep_rw)
        ep_steps.append(ep_step)
        man_ep_rw = 0
        wor_ep_rw = 0
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
            val_man_action= man_agent.policy_fn(man_agent.policy_params, val_obs)
            # Get Worker action (wheel speeds)
            val_wor_obs = np.concatenate([val_obs, val_man_action])
            val_wor_action= wor_agent.policy_fn(wor_agent.policy_params, val_wor_obs)
            
            # Join actions and step environment
            val_action = np.stack([val_man_action, val_wor_action])
            val_obs, val_rewards, val_done, val_step_info = val_env.step(val_action)
            val_man_ep_rw += val_rewards[0]
            val_wor_ep_rw += val_rewards[1]
            val_step += 1

        val_step_info.update(dict(validation_manager_return=val_man_ep_rw, validation_worker_return=val_wor_ep_rw, validation_steps=val_step))
        print(val_step_info)


while not done:
    _obs, reward, done, step_info = env.step(env.action_space.sample)
    obs = _obs

env.close()
