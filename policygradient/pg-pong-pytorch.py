# Pytorch implementation of PG Pong
# [Reference]
# 1. Karpathy pg-pong.py: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# 2. PyTorch official example: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import argparse
import os
from itertools import count

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

is_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch policy gradient example at openai-gym pong')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
                    help='decay rate for RMSprop (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=20, metavar='G',
                    help='Every how many episodes to da a param update')
parser.add_argument('--seed', type=int, default=87, metavar='N',
                    help='random seed (default: 87)')
parser.add_argument('--test', action='store_true',
                    help='whether to test the trained model or keep training')

args = parser.parse_args()

torch.manual_seed(args.seed)

D = 80 * 80
test = args.test
if test:
    render = True
else:
    render = False

if render:
    env = gym.make('Pong-v0', render_mode='human')
else:
    env = gym.make('Pong-v0')


def preprocess_image(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


class Policy(nn.Module):
    def __init__(self, num_actions=2):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.affine2 = nn.Linear(200, num_actions)  # action 1: static, action 2: move up, action 3: move down
        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []
        rand_var = torch.tensor([0.5, 0.5])
        if is_cuda:
            rand_var = rand_var.cuda()
        self.random = Categorical(rand_var)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        if is_cuda:
            x = x.cuda()
        probs = self.forward(x)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append(m.log_prob(action))
        return action


# built policy network
policy = Policy()
if is_cuda:
    policy.cuda()

# check & load pretrain model
if os.path.isfile('pg_params.pkl'):
    print('Load Policy Network parameters ...')
    policy.load_state_dict(torch.load('pg_params.pkl'))

# construct a optimal function
optimizer = optim.RMSprop(policy.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)


def finish_episode():
    policy_loss = []
    discounted_return = []  # Storing the discounted return for each step, flattened for all episodes.
    for episode_rewards in policy.rewards[::-1]:
        step_return = 0
        for step_reward in episode_rewards[::-1]:
            step_return = step_reward + args.gamma * step_return
            discounted_return.insert(0, step_return)
    # turn rewards to pytorch tensor and standardize
    discounted_return = torch.Tensor(discounted_return)
    discounted_return = (discounted_return - discounted_return.mean()) / (discounted_return.std() + 1e-6)

    for log_prob, step_return in zip(policy.saved_log_probs, discounted_return):
        policy_loss.append(- log_prob * step_return)
    policy_loss = torch.stack(policy_loss).sum()
    if is_cuda:
        policy_loss.cuda()
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    # clean rewards and saved_actions
    policy.rewards.clear()
    policy.saved_log_probs.clear()


# Main loop
running_reward = None
reward_sum = 0
for i_episode in count(1):
    state, _ = env.reset(seed=args.seed + i_episode)
    policy.rewards.append([])
    prev_x = None
    for t in range(10000):
        if render:
            env.render()
        cur_x = preprocess_image(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        action = policy.select_action(x)
        action_env = action + 2
        state, reward, terminated, truncated, _ = env.step(action_env)
        done = np.logical_or(terminated, truncated)
        reward_sum += reward

        policy.rewards[-1].append(reward)
        if done:
            # tracking log
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print(
                'REINFORCE ep %03d done. reward: %f. reward running mean: %f' % (i_episode, reward_sum, running_reward))
            reward_sum = 0
            break

    # use policy gradient update model weights
    if i_episode % args.batch_size == 0:
        finish_episode()

    # Save model in every 50 episode
    if i_episode % 50 == 0:
        print('ep %d: model saving...' % (i_episode))
        torch.save(policy.state_dict(), 'pg_params.pkl')
