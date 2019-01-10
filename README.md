# Some basic examples for reinforcement learning

## Installing Anaconda and OpenAI gym

* Download and install Anaconda [here](https://www.anaconda.com/download)
* Install OpenAI gym
```
pip install gym
pip install gym[atari]
```

## Examples

* Play with the environment
```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

```

* Random play with ```CartPole-v0```

```
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
```

* Example code for random playing (```Pong-ram-v0```,```Acrobot-v1```,```Breakout-v0```)

```
python my_random_agent.py Pong-ram-v0
```

* Very naive learnable agent playing ```CartPole-v0``` or ```Acrobot-v1```

```
python my_learning_agent.py CartPole-v0

```

* Playing Pong on CPU (with a great [blog](http://karpathy.github.io/2016/05/31/rl/)). One pretrained model is ```pong_model_bolei.p```(after training 20,000 episodes), which you can load in by replacing [save_file](https://github.com/metalbubble/RLexample/blob/master/pg-pong.py#L15) in the script. 

```
python pg-pong.py

```

* Random navigation agent in [AI2THOR](https://github.com/allenai/ai2thor)

```
python navigation_agent.py
```

