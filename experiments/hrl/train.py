from argparse import ArgumentParser
from copy import deepcopy

import flax.training.checkpoints
import gym
import jax
import rsoccer_gym
from jax_agents.agents import AgentDDPG
from jax_agents.agents.ddpg.hyperparameters import HyperparametersDDPG
from tqdm import tqdm
import numpy as np

import wandb
from utils import info_to_log, run_validation_ep


def main(args):
    wandb.init(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        monitor_gym=args.wandb_monitor_gym,
        config=args,
    )

    hp = HyperparametersDDPG(
        environment_name=args.env_name,
        total_training_steps=args.training_total_training_steps
        + args.training_replay_min_size,
        replay_capacity=args.training_total_training_steps
        + args.training_replay_min_size,
        min_replay_size=args.training_replay_min_size,
        batch_size=args.training_batch_size,
        gamma=args.training_gamma,
        learning_rate=args.training_learning_rate,
        custom_env_space=True,
        seed=args.seed,  # TODO: remove every non jax random from agent
    )

    env = gym.make(
        args.env_name,
        n_robots_blue=args.env_n_robots_blue,
        n_robots_yellow=args.env_n_robots_yellow,
    )
    if args.training_val_frequency:
        val_env = gym.wrappers.RecordVideo(
            gym.make(
                args.env_name,
                n_robots_blue=args.env_n_robots_blue,
                n_robots_yellow=args.env_n_robots_yellow,
            ),
            './monitor/',
            episode_trigger=lambda x: True,
        )
    key = jax.random.PRNGKey(hp.seed)
    if args.env_opponent_policy == 'off':
        opponent_policies = [
            lambda: np.array([0.0, 0.0]) for _ in range(args.env_n_robots_yellow)
        ]
    env.set_key(key)
    val_env.set_key(key)

    # Manager Hyperparameters
    m_hp = deepcopy(hp)
    m_hp.seed = hp.seed + 1  # TODO: Change so seed received is a PRNG key
    m_hp.observation_space, m_hp.action_space = env.get_spaces_m()

    # Worker Hyperparameters
    w_hp = deepcopy(hp)
    w_hp.seed = hp.seed + 2
    w_hp.observation_space, w_hp.action_space = env.get_spaces_w()

    # Create Agents
    m_agent = AgentDDPG(m_hp)
    w_agent = AgentDDPG(w_hp)

    if args.training_load_worker:
        w_agent.policy_params = flax.training.checkpoints.restore_checkpoint(
            'checkpoints/', w_agent.policy_params, prefix='goto_worker_policy'
        )

    m_obs = env.reset()
    rewards, ep_steps, done, q_losses, pi_losses = 0, 0, False, [], []
    for step in tqdm(range(hp.total_training_steps), smoothing=0.01):
        if args.training_val_frequency and step % args.training_val_frequency == 0:
            run_validation_ep(m_agent, w_agent, val_env, opponent_policies)
        m_action, _ = m_agent.sample_action(m_obs)
        w_obs = env.set_action_m(m_action)
        w_action = w_agent.policy_fn(w_agent.policy_params, w_obs)
        step_action = np.concatenate([w_action] + [[p()] for p in opponent_policies], axis=0)
        _obs, reward, done, info = env.step(step_action)

        terminal_state = False if not done or "TimeLimit.truncated" in info else True

        m_agent.observe(
            m_obs, m_action, None, reward.manager, terminal_state, _obs.manager
        )
        update_info = m_agent.update()
        if update_info:
            q_losses.append(update_info['agent/q_value_loss'].mean())
            pi_losses.append(update_info['agent/policy_loss'].mean())

        rewards += reward.manager
        ep_steps += 1
        m_obs = _obs.manager
        if done:
            m_obs = env.reset()
            log = info_to_log(info)
            log.update(
                {
                    'ep_reward': rewards,
                    'ep_steps': ep_steps,
                    'q_loss': np.mean(q_losses),
                    'pi_loss': np.mean(pi_losses),
                }
            )
            wandb.log(log, step=step)
            rewards, ep_steps, q_losses, pi_losses = 0, 0, [], []


if __name__ == '__main__':
    parser = ArgumentParser(fromfile_prefix_chars='@')
    # RANDOM
    parser.add_argument('--seed', type=int, default=0)

    # WANDB
    parser.add_argument('--wandb-mode', type=str, default='disabled')
    parser.add_argument('--wandb-project', type=str, default='rsoccer-hrl')
    parser.add_argument('--wandb-entity', type=str, default='felipemartins')
    parser.add_argument('--wandb-name', type=str)
    parser.add_argument('--wandb-monitor-gym', type=bool, default=True)

    # ENVIRONMENT
    parser.add_argument('--env-name', type=str, default='VSSHRL-v0')
    parser.add_argument('--env-n-robots-blue', type=int, default=1)
    parser.add_argument('--env-n-robots-yellow', type=int, default=0)
    parser.add_argument('--env-opponent-policy', type=str, default='off')

    # TRAINING
    parser.add_argument('--training-total-training-steps', type=int, default=3000000)
    parser.add_argument('--training-replay-min-size', type=int, default=100000)
    parser.add_argument('--training-batch-size', type=int, default=256)
    parser.add_argument('--training-gamma', type=float, default=0.95)
    parser.add_argument('--training-learning-rate', type=float, default=1e-4)
    parser.add_argument('--training-val-frequency', type=int, default=100000)
    parser.add_argument('--training-load-worker', type=bool, default=True)

    args = parser.parse_args()
    main(args)
