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
plot_path = os.path.join(dir, 'figures', 'continuous_action_prioritized_replay', '')


action_errors = [0, 0.01, 0.1, 1]
training_steps = 400000

print("running DQN")

# with error
# multiprocess environment
n_cpu = 4
env_string = 'MountainCarContinuous-v0'

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
        title = "contvel_std=" + std_str + "_i=" + str(i)
        print("Processing std = ", std)

        model = DQN(MlpPolicy, env, verbose=0, action_error_std=std, prioritized_replay=False)
        model.learn(total_timesteps=training_steps, eval_env_string=env_string)

        # for plotting
        plot_summary_with_fn(model.ep_logs, model.value_log, model.trialtype_log, plot_path, title)
        plot_network_activation(model.layer_log, model.ep_logs, model.trialtype_log, plot_path,
                                title + "_last_trial_layer_")

        del model # remove to demonstrate saving and loading