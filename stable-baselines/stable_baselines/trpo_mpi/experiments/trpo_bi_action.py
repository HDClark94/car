import matplotlib.pyplot as plot
#%matplotlib inline
import gym
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TRPO
import numpy as np
from stable_baselines.common.plotting import *
import os

dir = os.path.dirname(__file__)
plot_path = os.path.join(dir, 'figures', 'binary_action', '')

action_errors = [0, 0.0001, 0.001, 0.01, 0.1, 1]
training_steps = 4000

print("running TRPO")

# with error
# multiprocess environment
n_cpu = 4
env_string = 'MountainCar-v0'

for std in action_errors:

    # set params for env
    env = gym.make(env_string)
    env.set_obs_error(std)
    env = DummyVecEnv([lambda: env])

    for i in range(3):
        if len(str(std).split("."))>1:
            std_str = str(std).split(".")[1]
        else:
            std_str = str(std)
        title = "bivel_std=" + std_str + "_i=" + str(i)
        print("Processing std = ", std)

        model = TRPO(MlpPolicy, env, verbose=0, action_error_std=std)
        model.learn(total_timesteps=training_steps, eval_env_string=env_string)

        # for plotting
        plot_summary(model.ep_logs, plot_path, title)

        del model # remove to demonstrate saving and loading
