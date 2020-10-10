# Pytorch implementation of Actor Critic
# Bolei Zhou, 10 March 2020
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
import pdb

is_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Policy Graident Actor Critic example at openai-gym pong')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
                    help='decay rate for RMSprop (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=20, metavar='G',
                    help='Every how many episodes to da a param update')
parser.add_argument('--seed', type=int, default=87, metavar='N',
                    help='random seed (default: 87)')
parser.add_argument('--test', action='store_true',
        help='whether to test the trained model or keep training')
parser.add_argument('--max-grad-norm', type=float, default=10)
parser.add_argument('--value-loss-coef', type=float, default=0.5)
args = parser.parse_args()


env = gym.make('Pong-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

D = 80 * 80
test = args.test
if test ==True:
    render = True
else:
    render = False

def prepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0 ] = 1
    return I.astype(np.float).ravel()


class AC(nn.Module):
    def __init__(self, num_actions=2):
        super(AC, self).__init__()
        self.affine1 = nn.Linear(6400, 200)
        self.action_head = nn.Linear(200, num_actions) # action 1: static, action 2: move up, action 3: move down
        self.value_head = nn.Linear(200, 1)

        self.num_actions = num_actions
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


    def select_action(self, x):
        x = Variable(torch.from_numpy(x).float().unsqueeze(0))
        if is_cuda: x = x.cuda()
        probs, state_value = self.forward(x)
        m = Categorical(probs)
        action = m.sample()

        self.saved_log_probs[-1].append((m.log_prob(action), state_value))
        return action

# built policy network
policy = AC()
if is_cuda:
    policy.cuda()

# check & load pretrain model
if os.path.isfile('ac_params.pkl'):
    print('Load Actor Critic Network parametets ...')
    if is_cuda:
        policy.load_state_dict(torch.load('ac_params.pkl'))
    else:
        policy.load_state_dict(torch.load('ac_params.pkl', map_location=lambda storage, loc: storage))


# construct a optimal function
optimizer = optim.RMSprop(policy.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)

def finish_episode():
    R = 0
    policy_loss = []
    value_loss = []
    rewards = []
    for episode_id, episode_reward_list in enumerate(policy.rewards):
        for i, r in enumerate(episode_reward_list):
            if i == len(episode_reward_list) - 1:
                R = torch.scalar_tensor(r)
            else:
                R = r + args.gamma * policy.saved_log_probs[episode_id][i + 1][1]
            rewards.append(R)
    if is_cuda: rewards = rewards.cuda()
    flatten_log_probs = [sample for episode in policy.saved_log_probs for sample in episode]
    assert len(flatten_log_probs) == len(rewards)
    for (log_prob, value), reward in zip(flatten_log_probs, rewards):
        advantage = reward - value # A(s,a) = r + gamma V(s_t+1) - V(s_t)
        policy_loss.append(- log_prob * advantage)         # policy gradient
        value_loss.append(F.smooth_l1_loss(value.reshape(-1), reward.reshape(-1))) # value function approximation
    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    value_loss = torch.stack(value_loss).sum()
    loss = policy_loss + args.value_loss_coef * value_loss
    if is_cuda:
        loss.cuda()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm) # gradient clip
    optimizer.step()

    # clean rewards and saved_actions
    del policy.rewards[:]
    del policy.saved_log_probs[:]


# Main loop
running_reward = None
reward_sum = 0
for i_episode in count(1):
    state = env.reset()
    prev_x = None
    policy.rewards.append([])  # record rewards separately for each episode
    policy.saved_log_probs.append([])
    for t in range(10000):
        if render: env.render()
        cur_x = prepro(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x
        action = policy.select_action(x)
        action_env = action + 2
        state, reward, done, _ = env.step(action_env)
        reward_sum += reward

        policy.rewards[-1].append(reward)
        if done:
            # tracking log
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('Actor Critic ep %03d done. reward: %f. reward running mean: %f' % (i_episode, reward_sum, running_reward))
            reward_sum = 0
            break


    # use policy gradient update model weights
    if i_episode % args.batch_size == 0 and test == False:
        finish_episode()

    # Save model in every 50 episode
    if i_episode % 50 == 0 and test == False:
        print('ep %d: model saving...' % (i_episode))
        torch.save(policy.state_dict(), 'ac_params.pkl')



