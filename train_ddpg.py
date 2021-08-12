import gym
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from flax.training.checkpoints import save_checkpoint

from rl_jax.agents import DDPG

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "Pendulum-v0", "Environment name.")
flags.DEFINE_string("run_name", default=None, help="wandb run name.", required=False)
flags.DEFINE_integer("training_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_seed", 199, "Evaluation eps seed.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("replay_capacity", int(1e6), "replay capacty.")
flags.DEFINE_integer("min_replay_size", None, "Minimal replay size.")
flags.DEFINE_float("gamma", 0.99, "Discount.")
flags.DEFINE_float("ou_sigma", 0.20, "OU noise sigma.")
flags.DEFINE_bool("log", False, "Log experiment to wandb")


def main(_):
    FLAGS.min_replay_size = FLAGS.min_replay_size or FLAGS.batch_size

    # Initialize environment and random key
    env = gym.wrappers.Monitor(gym.make(FLAGS.env_name), "./monitor/", force=True)
    rkey = jax.random.PRNGKey(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    env.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)

    config = {
        "env_name": FLAGS.env_name,
        "run_name": "-".join([FLAGS.run_name] + [FLAGS.env_name.split("-")[-1]])
        if FLAGS.run_name
        else None,
        "training_steps": FLAGS.training_steps,
        "seed": FLAGS.seed,
        "evaluation_seed": FLAGS.eval_seed,
        "agent": str(DDPG),
        "replay_capacity": FLAGS.replay_capacity,
        "min_replay_size": FLAGS.min_replay_size,
        "batch_size": FLAGS.batch_size,
        "gamma": FLAGS.gamma,
        "ou_sigma": FLAGS.ou_sigma,
    }

    wandb_mode = None if FLAGS.log else "disabled"
    wandb.init(
        monitor_gym=True,
        project="rl-jax",
        config=config,
        mode=wandb_mode,
    )
    if FLAGS.run_name:
        # wandb.run.name = "-".join([FLAGS.run_name] + [wandb.run.name.split("-")[-1]])
        wandb.run.name = "-".join(
            [FLAGS.run_name] + [FLAGS.env_name.split("-")[-1]] + [str(FLAGS.seed)]
        )
    else:
        wandb.run.name = FLAGS.env_name + "-" + str(FLAGS.seed)
    wandb.config.update({"wandb_run_name": wandb.run.name})

    # Initialize DQN agent and hyperparameters
    rkey, subkey = jax.random.split(rkey)
    # min_replay =
    agent = DDPG(
        env,
        subkey,
        replay_capacity=FLAGS.replay_capacity,
        min_replay_size=FLAGS.min_replay_size,
        batch_size=FLAGS.batch_size,
        gamma=FLAGS.gamma,
        ou_sigma=FLAGS.ou_sigma,
    )

    # Training loop
    s = env.reset()
    rw = 0.0
    ep_steps = 0
    n_episodes = 0
    for step in tqdm.tqdm(range(1, FLAGS.training_steps + 1), smoothing=0.1):
        metrics = {}
        a = agent.select_action(s, add_noise=True)
        s_, r, done, info = env.step(a)
        ep_steps += 1
        rw += r

        agent.observe(s, a, r, done, s_, info)
        grad_info = agent.update()
        if grad_info:
            metrics.update(grad_info)

        s = s_
        if done:
            n_episodes += 1
            metrics.update(info)
            s = env.reset()
            metrics.update(
                {
                    "episode_rw": rw,
                    "episode_steps": ep_steps,
                    "episodes_count": n_episodes,
                }
            )
            rw = 0.0
            ep_steps = 0

        wandb.log(metrics)

    save_dir = "checkpoints/" + wandb.run.name
    save_checkpoint(save_dir, agent._actor, step=FLAGS.training_steps)
    print(wandb.save(save_dir + "/*", policy="now"))

    env.close()
    env = gym.wrappers.Monitor(
        gym.make(FLAGS.env_name),
        "./monitor/",
        resume=True,
        video_callable=lambda x: True,
    )

    eval_seed = FLAGS.eval_seed
    np.random.seed(eval_seed)
    env.seed(eval_seed)
    env.action_space.seed(eval_seed)
    env.observation_space.seed(eval_seed)

    # Play model
    eval_rws = []
    eval_steps = []
    for ep in range(5):
        rkey, subkey = jax.random.split(rkey)
        s = env.reset()
        done = False
        rw = 0.0
        step = 0
        while not done:
            env.render()
            a = agent.select_action(s, add_noise=False)
            s, r, done, _ = env.step(a)
            step += 1
            rw += r
        wandb.log({"evaluate_rw": rw, "evaluate_ep": ep, "evaluate_ep_steps": step})
        eval_rws.append(rw)
        eval_steps.append(step)
        print(f"Episode {ep} Reward: {rw}")

    wandb.summary["evaluate/seed"] = eval_seed
    wandb.summary["evaluate/mean_rw"] = np.mean(eval_rws)
    wandb.summary["evaluate/std_rw"] = np.std(eval_rws)
    wandb.summary["evaluate/mean_steps"] = np.mean(eval_steps)
    wandb.summary["evaluate/std_steps"] = np.std(eval_steps)

    env.close()


if __name__ == "__main__":
    app.run(main)
