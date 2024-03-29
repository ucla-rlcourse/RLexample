"""
Model-free Control for OpenAI FrozenLake env (https://gym.openai.com/envs/FrozenLake-v0/)

Bolei Zhou, RL course example

"""
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

render = False
render_mode = 'human' if render else None
no_slippery = True
render_last = True  # whether to visualize the last episode in testing

# -- hyperparameters--
num_epis_train = 10000
num_iter = 100
learning_rate = 0.01
discount = 0.8
eps = 0.3

if no_slippery == True:
    # the simplified frozen lake without slippery (so the transition is deterministic)
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gymnasium.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name': '4x4', 'is_slippery': False},
        max_episode_steps=1000,
        reward_threshold=0.78,  # optimum = .8196
    )
    env = gym.make('FrozenLakeNotSlippery-v0', render_mode=render_mode)
else:
    # the standard slippery frozen lake
    env = gym.make('FrozenLake-v0', render_mode=render_mode)

q_learning_table = np.zeros([env.observation_space.n, env.action_space.n])

# -- training the agent ----
for epis in range(num_epis_train):
    state, _ = env.reset()
    for iter in range(num_iter):
        if np.random.uniform(0, 1) < eps:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax(q_learning_table[state, :])
        state_new, reward, terminated, truncated, _ = env.step(action)
        done = np.logical_or(terminated, truncated)
        q_learning_table[state, action] = q_learning_table[state, action] + learning_rate * (
                    reward + discount * np.max(q_learning_table[state_new, :]) - q_learning_table[state, action])
        state = state_new
        if done: break

print(np.argmax(q_learning_table, axis=1))
print(np.around(q_learning_table, 6))

if no_slippery == True:
    print('---Frozenlake without slippery move-----')
else:
    print('---Standard frozenlake------------------')

# visualize no uncertainty
num_episode = 500
rewards = 0
for epi in range(num_episode):
    s, _ = env.reset()
    for _ in range(100):
        action = np.argmax(q_learning_table[s, :])
        state_new, reward_episode, terminated, truncated, _ = env.step(action)
        done_episode = np.logical_or(terminated, truncated)
        # if epi == num_episode -1 and render_last:
        #     env.render()
        s = state_new
        if done_episode:
            if reward_episode == 1:
                rewards += 1
            break

print('---Success rate=%.3f' % (rewards * 1.0 / num_episode))
print('-------------------------------')
