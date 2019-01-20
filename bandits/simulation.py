from core import *
import matplotlib.pyplot as plt
import numpy as np
from algorithms.randomselect import *

### construct the multi-arm bandit
arm1 = BernoulliArm(0.5)
arm2 = BernoulliArm(0.5) #arm2 = NormalArm(2.0, 1.0)
arm3 = BernoulliArm(0.5)


rewards_single = []

arms = [arm1, arm2, arm3]
n_arms = len(arms)

### construct the algorithm
algo1 = EpsilonGreedy(0.1, [], [])
algo2 = Softmax(1.0, [], [])
algo3 = UCB1([], [])
algo4 = RandomSelect()


algos = [algo1, algo2, algo3, algo4]
names = ['EpsilonGreedy','softmax','ucb1','RandomSelect']

for algo in algos:
    algo.initialize(n_arms)

### start the simulation
horizon = 1000
rewards_algorithms = np.zeros((len(algos), horizon))
arms_algorithms = np.zeros((len(algos), horizon))

for t in range(horizon):
    for i, algo in enumerate(algos):
        chosen_arm = algo.select_arm()
        reward = arms[chosen_arm].draw()
        algo.update(chosen_arm, reward)

        # bookkeeping
        rewards_algorithms[i][t] = reward
        arms_algorithms[i][t] = chosen_arm


### plot the internal estimation of the algorithm
print('--------------------------------------------------')
for i in range(len(algos)):
    print('-------Algorithm:', names[i])
    print('Count:', algos[i].counts)
    print('Value:', algos[i].values)


### plot the average reward until t
average_rewards_algorithms = np.zeros((len(algos), horizon))
for i in range(len(algos)):
    for t in range(1, horizon):
        average_rewards_algorithms[i][t] = np.mean(rewards_algorithms[i][:t])

for i in range(len(algos)):
    plt.plot(average_rewards_algorithms[i])
plt.legend(names)
plt.title('average reward until time t')
plt.show()
