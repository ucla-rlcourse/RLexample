# Pytorch implementation of PG Pong
# [Reference]
# 1. Karpathy pg-pong.py: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
# 2. PyTorch official example: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import os
import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch policy gradient example at openai-gym pong')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
                    help='decay rate for RMSprop (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=10, metavar='G',
                    help='Every how many episodes to da a param update')
parser.add_argument('--seed', type=int, default=87, metavar='N',
                    help='random seed (default: 87)')
args = parser.parse_args()


env = gym.make('Pong-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


def prepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0 ] = 1
    return I.astype(np.float).ravel()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.affine2 = nn.Linear(200, 3) # action 1: static, action 2: move up, action 3: move down

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


# built policy network
policy = Policy()


# check & load pretrain model
if os.path.isfile('pg_params.pkl'):
    print('Load Policy Network parametets ...')
    policy.load_state_dict(torch.load('pg_params.pkl'))


# construct a optimal function
optimizer = optim.RMSprop(policy.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    m = Categorical(probs)
    action = m.sample() # 從multinomial分佈中抽樣
    policy.saved_log_probs.append(m.log_prob(action)) # 蒐集log action以利於backward
    return action.data[0]


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    # turn rewards to pytorch tensor and standardize
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    # clean rewards and saved_actions
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# Main loop
running_reward = None
reward_sum = 0
for i_episode in count(1):
    state = env.reset()
    for t in range(10000):
        state = prepro(state)
        action = select_action(state)
        action = action + 1
        state, reward, done, _ = env.step(action)
        reward_sum += reward

        policy.rewards.append(reward)
        if done:
            # tracking log
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            break

        if reward != 0:
            print('ep %d: game finished, reward: %f' % (i_episode, reward) + ('' if reward == -1 else ' !!!!!!!'))

    # use policy gradient update model weights
    if i_episode % args.batch_size == 0:
        print('ep %d: policy network parameters updating...' % (i_episode))
        finish_episode()

    # Save model in every 50 episode
    if i_episode % 50 == 0:
        print('ep %d: model saving...' % (i_episode))
        torch.save(policy.state_dict(), 'pg_params.pkl')



