## Solving FrozenLake using Policy Iteration and Value Iteration

## Introduction
The [FrozenLake env](https://gym.openai.com/envs/FrozenLake-v0/) in OpenAI is a very classic example for Markov Decision Process. Please study the dynamics of the task in detail. Then we will implement the value iteration and policy iteration to search the optimal policy. 

* Load the FrozenLake environment and study the dynamics of the environment:
```
    env_name  = 'FrozenLake-v0' # 'FrozenLake8x8-v0'
    env = gym.make(env_name)
    print(env.env.P)
    # it will show 16 states, in which state, there are 4 actions, for each action, there are three possibile states to go with prob=0.333
```

* Run value iteration on FrozenLake
```
python frozenlake_vale_iteration.py
```

* Run policy iteration on FrozenLake
```
python frozenlake_policy_iteration.py
```

* Switch to FrozenLake8x8-v0 for more challenging task.
