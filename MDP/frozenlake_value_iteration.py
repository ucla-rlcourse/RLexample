"""
Solving FrozenLake environment using Value-Iteration.

Updated 17 Aug 2020

updated by Bolei from the feedback of ghost0832, Jan 3, 2025
"""
import gymnasium as gym
import numpy as np


def run_episode(env, policy, gamma=1.0):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.

    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.

    returns:
    total reward: real value of the total reward received by agent under policy.
    """
    obs, _ = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        obs, reward, terminated, truncated, _ = env.step(int(policy[obs]))
        done = np.logical_or(terminated, truncated)  # here use the logical or, one can use terminal
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward


def evaluate_policy(env, policy, gamma=1.0, n=100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
        run_episode(env, policy, gamma=gamma)
        for _ in range(n)]
    return np.mean(scores)


def extract_policy(v, gamma=1.0):
    """ Extract the policy given a value-function """
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = np.zeros(env.action_space.n)
        for a in range(env.action_space.n):
            for next_sr in env.P[s][a]:
                # next_sr is a tuple of (probability, next state, reward, done)
                p, s_, r, _ = next_sr
                q_sa[a] += (p * (r + gamma * v[s_]))
        policy[s] = np.argmax(q_sa)
    return policy


def value_iteration(env, gamma=1.0):
    """ Value-iteration algorithm """
    v = np.zeros(env.observation_space.n)  # initialize value-function
    max_iterations = 100000
    eps = 1e-20
    for i in range(max_iterations):
        prev_v = np.copy(v)
        for s in range(env.observation_space.n):
            q_sa = [sum([p * (r + gamma * prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in
                    range(env.action_space.n)]
            v[s] = max(q_sa)
        if (np.sum(np.fabs(prev_v - v)) <= eps):
            print('Value-iteration converged at iteration# %d.' % (i + 1))
            break
    return v


if __name__ == '__main__':
    render = True
    env_name = 'FrozenLake-v1'  # 'FrozenLake8x8-v0'
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    env = env.unwrapped
    gamma = 1.0
    optimal_v = value_iteration(env, gamma)
    policy = extract_policy(optimal_v, gamma)
    policy_score = evaluate_policy(env, policy, gamma, n=1000)
    print('Policy average score = ', policy_score)
