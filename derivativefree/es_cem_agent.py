# demonstration of derivative-free methods evolution strategy and cross-entropy method
# by Bolei

import gym
import numpy as np
import argparse

class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]
    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a

class ContinuousActionLinearPolicy(object):
    def __init__(self, theta, n_in, n_out):
        assert len(theta) == (n_in + 1) * n_out
        self.W = theta[0 : n_in * n_out].reshape(n_in, n_out)
        self.b = theta[n_in * n_out : None].reshape(1, n_out)
    def act(self, ob):
        a = ob.dot(self.W) + self.b
        return a

def es(f, th_mean, batch_size, n_iter, eps = 0.1, lr=0.01):
    """
    implementation of the evolution strategy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    """
    for _ in range(n_iter):
        ds = np.array([dth for dth in np.random.randn(batch_size, th_mean.size)]) # perturbations
        ths = np.array([th_mean + eps *dth for dth in ds]) # perturbed parameters
        ys = np.array([f(th) for th in ths]) # here is the reward for each episode
        delta_ths = np.array([ds[i]*ys[i] for i in range(len(ys))])
        th_mean = th_mean + lr * delta_ths.mean(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1

def noisy_evaluation(theta):
    agent = BinaryActionLinearPolicy(theta)
    rew, T = do_rollout(agent, env, 100)
    return rew

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--env', nargs="?", default="CartPole-v0")
    parser.add_argument('--method',nargs="?", default="es")
    parser.add_argument('--n_iter', default=100)
    args = parser.parse_args()

    env = gym.make(args.env)
    env.seed(0)
    np.random.seed(0)

    # Train the agent, and snapshot each stage
    if args.method == 'es':
        f = es
        params = dict(n_iter=args.n_iter, batch_size=10)
    elif args.method == 'cem':
        f = cem
        params = dict(n_iter=args.n_iter, batch_size=10, elite_frac = 0.2)
    else:
        print('no such method:', args.method)

    for (i, iterdata) in enumerate(f(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)):
        print('%s Iteration %2i. Episode mean reward: %7.3f'%(args.method, i, iterdata['y_mean']))
        agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
        do_rollout(agent, env, 100, render=True)


    env.close()
