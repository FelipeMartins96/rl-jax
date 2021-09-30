from jax_agents.agents import AgentPPO
import gym

hp = AgentPPO.get_hyperparameters()
env = gym.make(hp.environment_name)

agent = AgentPPO(hp)

obs = env.reset()
for i in range(hp.n_rollout_steps):
    action = env.action_space.sample()
    _obs, reward, done, _ = env.step(action)
    
    agent.observe(obs, action, 0.0, reward, done, _obs)

    if done:
        obs = env.reset()
    else:
        obs = _obs

info = agent.update()
print(info)