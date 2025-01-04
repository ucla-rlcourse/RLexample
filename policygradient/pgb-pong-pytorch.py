# Pytorch implementation of PG with baseline
#
# Bolei Zhou, 22 Feb 2019
# PENG Zhenghao, updated 23 October 2021
# PENG Zhenghao, updated Oct 10, 2023
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

parser = argparse.ArgumentParser(description='PyTorch PG with baseline example at openai-gym pong')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=20, metavar='G',
                    help='Every how many episodes to da a param update')
parser.add_argument('--seed', type=int, default=87, metavar='N',
                    help='random seed (default: 87)')
parser.add_argument('--test', action='store_true',
                    help='whether to test the trained model or keep training')
parser.add_argument('--use_value_gradient', action='store_true',
                    help='whether to pass gradient to value network when computing policy loss. Test purpose only!')

args = parser.parse_args()

torch.manual_seed(args.seed)

D = 80 * 80
test = args.test
if test:
    render = True
else:
    render = False

if render:
    env = gym.make("ale_py:ALE/Pong-v5" , render_mode='human')
else:
    env = gym.make("ale_py:ALE/Pong-v5")


def preprocess_image(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float32).ravel()


class PGbaseline(nn.Module):
    def __init__(self, num_actions=2):
        super(PGbaseline, self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.action_head = nn.Linear(200, num_actions)  # action 1: static, action 2: move up, action 3: move down
        self.value_head = nn.Linear(200, 1)

        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []  # A nested list. Each element is a list containing the reward from one episode.

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        if is_cuda:
            x = x.cuda()
        probs, state_value = self.forward(x)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs.append((m.log_prob(action), state_value))
        return action


# built policy network
policy = PGbaseline()
if is_cuda:
    policy.cuda()

# check & load pretrain model
if os.path.isfile('pgb_params.pkl'):
    print('Load PGbaseline Network parameters ...')
    if is_cuda:
        policy.load_state_dict(torch.load('pgb_params.pkl'))
    else:
        policy.load_state_dict(torch.load('pgb_params.pkl', map_location=lambda storage, loc: storage))

# construct a optimal function
optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)


def finish_episode():
    discounted_return = []  # Storing the discounted return for each step, flattened for all episodes.
    step_return = 0.0
    # Zhenghao Note: Using a for loop over a pytorch tensor is extremely inefficient. Matrix operations are preferred.
    for step_reward in policy.rewards[::-1]:
        # Zhenghao Note: This is a trick here. In Pong environment, one episode might have 21 games, where each game
        # will end while giving the agent a reward +1 or -1. We treat each game as an episode. This is proven to be
        # an effective trick to train on Atari game.
        if step_reward != 0.0:
            step_return = 0.0
        step_return = step_reward + args.gamma * step_return
        discounted_return.insert(0, step_return)

    # turn rewards to pytorch tensor and standardize
    discounted_return = torch.Tensor(discounted_return)
    discounted_return = (discounted_return - discounted_return.mean()) / (discounted_return.std() + 1e-6)

    log_probs = torch.concatenate([v[0] for v in policy.saved_log_probs])
    values = torch.concatenate([v[1] for v in policy.saved_log_probs])
    if is_cuda:
        discounted_return = discounted_return.cuda()
        log_probs = log_probs.cuda()
        values = values.cuda()
    advantage = discounted_return - values
    if args.use_value_gradient:
        pass
    else:
        advantage = advantage.detach()
    policy_loss = (-log_probs * advantage).mean()  # policy gradient
    value_loss = (F.smooth_l1_loss(values.reshape(-1), discounted_return.reshape(-1))).mean()  # value function approximation
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # clean rewards and saved_actions
    policy.rewards.clear()
    policy.saved_log_probs.clear()


# Main loop
running_reward = None
reward_sum = 0
for i_episode in count(1):
    state, _ = env.reset(seed=args.seed + i_episode)
    prev_x = None
    for t in range(10000):
        if render:
            env.render()
        cur_x = preprocess_image(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        action = policy.select_action(x)
        action_env = action + 2
        state, rew, terminated, truncated, _ = env.step(action_env)
        done = np.logical_or(terminated, truncated)
        reward_sum += rew

        policy.rewards.append(rew)
        if done:
            # tracking log
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('Policy Gradient with Baseline ep %03d done. reward: %f. reward running mean: %f' % (
                i_episode, reward_sum, running_reward))
            reward_sum = 0
            break

    # use policy gradient update model weights
    if i_episode % args.batch_size == 0 and test == False:
        finish_episode()

    # Save model in every 50 episode
    if (i_episode % 50 == 0) and (test == False):
        print('ep %d: model saving...' % (i_episode))
        torch.save(policy.state_dict(), 'pgb_params.pkl')
