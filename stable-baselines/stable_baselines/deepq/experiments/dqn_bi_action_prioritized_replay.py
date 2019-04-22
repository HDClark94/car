import matplotlib.pyplot as plot
#%matplotlib inline
import gym
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
import numpy as np
from stable_baselines.common.plotting import *
import os

dir = os.path.dirname(__file__)
plot_path = os.path.join(dir, 'figures', 'binary_action_prioritized_replay', '')

action_errors = [0, 0.0001, 0.001, 0.01, 0.1, 1]
actionDim = 2
training_steps = 400000

print("running DQN")

# with error
# multiprocess environment
n_cpu = 4
env_string = 'MountainCar-v0'

for std in action_errors:

    # set params for env
    env = gym.make(env_string)
    env.set_obs_error(std)
    env.set_action_dim(actionDim)
    env = DummyVecEnv([lambda: env])

    for i in range(3):
        if len(str(std).split("."))>1:
            std_str = str(std).split(".")[1]
        else:
            std_str = str(std)
        title = "bivel_std=" + std_str + "_i=" + str(i)
        print("Processing std = ", std)

        model = DQN(MlpPolicy, env, verbose=0, action_error_std=std, actiondim=actionDim, prioritized_replay=False)
        model.learn(total_timesteps=training_steps, eval_env_string=env_string)

        # for plotting
        plot_summary(model.ep_logs, plot_path, title)

        del model # remove to demonstrate saving and loading
