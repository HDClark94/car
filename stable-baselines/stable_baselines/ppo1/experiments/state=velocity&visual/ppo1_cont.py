import matplotlib.pyplot as plot
#%matplotlib inline
import gym
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1
import numpy as np
from stable_baselines.common.plotting import *
import os
from stable_baselines.common.misc_util import set_global_seeds

dir = os.path.dirname(__file__)
plot_path = os.path.join(dir, 'figures', 'continuous_action', '')

action_errors = [0, 0.001, 0.01, 0.1]
training_steps = 400000
seed = 3
print("running PPO1")

id = 30

# with error
env_string = 'state_velovisual_MountainCarContinuous_Action-v0'

for std in action_errors:

    # set params for env
    env = gym.make(env_string)
    env.set_obs_error(std)
    env.seed(seed)
    set_global_seeds(seed)

    env = DummyVecEnv([lambda: env])

    for i in range(4):
        if len(str(std).split("."))>1:
            std_str = str(std).split(".")[1]
        else:
            std_str = str(std)

        id_string = str(id).rjust(4, "0")
        title = "id=" + id_string + "_std=" + std_str + "_i=" + str(i)
        print("Processing std = ", std)

        model = PPO1(MlpPolicy, env, verbose=0, action_error_std=std)
        model.learn(total_timesteps=training_steps, eval_env_string=env_string, seed=seed)

        # for plotting
        plot_summary_with_fn(model.ep_logs, model.value_log, model.action_log, model.trialtype_log, plot_path, title)
        plot_network_activation(model.layer_log, model.ep_logs, model.trialtype_log, plot_path, title+"_last_trial_layer_")

        del model # remove to demonstrate saving and loading
        id += 1