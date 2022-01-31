import numpy as np

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

def run_validation_ep(m_agent, w_agent, env, opponent_policies):
    m_obs = env.reset()
    done = False
    while not done:
        m_action = m_agent.policy_fn(m_agent.policy_params, m_obs)
        w_obs = env.set_action_m(m_action)
        w_action = w_agent.policy_fn(w_agent.policy_params, w_obs)
        step_action = np.stack([w_action] + [[p()] for p in opponent_policies], axis=0)
        _, _, done, _ = env.step(step_action)