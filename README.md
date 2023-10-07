# Some basic examples for reinforcement learning

## Installing Anaconda and Gymnasium

* Download and install Anaconda [here](https://www.anaconda.com/download)
* Create conda env for managing dependencies and activate the conda env
```
conda create -n conda_env
conda activate conda_env
```
* Install gymnasium (Dependencies installed by pip will also go to the conda env)
```
pip install gymnasium[all]
```
* Install torch with either conda or pip
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
```
pip install torch torchvision torchaudio
```

## Examples

* Play with the environment and visualize the agent behaviour
```
import gymnasium as gym
render = True # switch if visualize the agent
if render:
    env = gym.make('CartPole-v0', render_mode='human')
else:
    env = gym.make('CartPole-v0')
env.reset(seed=0)
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
env.close()
```

* Random play with ```CartPole-v0```

```
import gymnasium as gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        print(observation)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = np.logical_or(terminated, truncated)
env.close()
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

