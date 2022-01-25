from jax_agents.agents.ddpg.hyperparameters import HyperparametersDDPG
from jax_agents.agents import AgentDDPG
from tqdm import tqdm
from copy import deepcopy
import gym
import rsoccer_gym
import flax.training.checkpoints
import wandb
import jax


def info_to_log(info):
    return {
        'manager/goal': info['manager_weighted_rw'][0],
        'manager/ball_grad': info['manager_weighted_rw'][1],
        'manager/move': info['manager_weighted_rw'][2],
        'manager/collision': info['manager_weighted_rw'][3],
        'manager/energy': info['manager_weighted_rw'][4],
        'worker/dist': info['workers_weighted_rw'][0][0],
        'worker/energy': info['workers_weighted_rw'][0][1],
    }


# Experiment Hyperparameters
exp_name = 'Teste-sarco'
load_worker = True

hp = HyperparametersDDPG(
    environment_name='VSSHRL-v0',
    total_training_steps=3100000,
    replay_capacity=3100000,
    min_replay_size=100000,
    batch_size=256,
    gamma=0.95,
    learning_rate=1e-4,
    seed=0,
)

env = gym.make(hp.environment_name)
env = gym.wrappers.RecordVideo(env, './videos/', step_trigger=lambda x: x % 50000 == 0)
env.set_key(jax.random.PRNGKey(hp.seed))

wandb.init(project='hrl-refactor', entity='felipemartins', name=exp_name, monitor_gym=True)

# Manager Hyperparameters
m_hp = deepcopy(hp)
m_hp.seed = hp.seed + 1
m_hp.custom_env_space = True
m_hp.observation_space, m_hp.action_space = env.get_spaces_m()

# Worker Hyperparameters
w_hp = deepcopy(hp)
m_hp.seed = hp.seed + 2
w_hp.custom_env_space = True
w_hp.observation_space, w_hp.action_space = env.get_spaces_w()

# Create Agents
m_agent = AgentDDPG(m_hp)
w_agent = AgentDDPG(w_hp)
if load_worker:
    w_agent.policy_params = flax.training.checkpoints.restore_checkpoint(
        'checkpoints/', w_agent.policy_params, prefix='goto_worker_policy'
    )

m_obs = env.reset()
rewards = 0
ep_steps = 0
done = False
for step in tqdm(range(hp.total_training_steps), smoothing=0.01):
    m_action, _ = m_agent.sample_action(m_obs)
    w_obs = env.set_action_m(m_action)
    w_action = w_agent.policy_fn(w_agent.policy_params, w_obs)  # TODO: Implement policies
    _obs, reward, done, info = env.step(w_action)

    terminal_state = False if not done or "TimeLimit.truncated" in info else True

    m_agent.observe(m_obs, m_action, None, reward.manager, terminal_state, _obs.manager)
    m_agent.update()

    rewards += reward.manager
    ep_steps += 1
    if done:
        m_obs = env.reset()
        log = info_to_log(info)
        log.update({'ep_reward': rewards, 'ep_steps': ep_steps})
        wandb.log(log, step=step)
        rewards, ep_steps = 0, 0
