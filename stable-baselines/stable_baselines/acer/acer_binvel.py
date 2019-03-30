import matplotlib.pyplot as plot
#%matplotlib inline
import gym
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import ACER
import numpy as np
from stable_baselines.common.plotting import *

action_errors = [0, 0.0001, 0.001, 0.01, 0.1]

actionDim = 2
training_steps = 2000

print("running acer")

rewards = []
plot_path = '/home/harry/PycharmProjects/car_rl/car/figures/acer_bivel/'

# with error
# multiprocess environment
n_cpu = 4

for std in action_errors:
    std_rewards = []

    # set params for env
    env = gym.make('MountainCar-v0')
    env.set_obs_error(std)
    env.set_action_dim(actionDim)
    env = SubprocVecEnv([lambda: env for i in range(n_cpu)])

    for i in range(3):
        if len(str(std).split("."))>1:
            std_str = str(std).split(".")[1]
        else:
            std_str = str(std)
        title = "bivel_std=" + std_str + "_i=" + str(i)
        print("Processing std = ", std)

        model = ACER(MlpPolicy, env, verbose=0, action_error_std=std, actiondim=actionDim)
        model.learn(total_timesteps=training_steps)
        model.save("acer_mountain")
        std_rewards.append(model.ep_rews)

        # for plotting
        plot_summary(model.ep_logs, plot_path, title)
        #raster(model.ep_logs, plot_path, title)

        eval_steps = np.array(model.eval_steps)
        del model # remove to demonstrate saving and loading

    rewards.append(std_rewards)

means = np.mean(np.array(rewards), axis=1)
stds = np.std(np.array(rewards), axis=1)

for i in range(len(means)):
    std = action_errors[i]
    plot.errorbar(eval_steps, means[i], yerr=stds[i], label=str(std))

plot.legend()
plot.xlabel('Training steps')
plot.ylabel('Average Episode Returns')
plot.savefig(plot_path + 'ActionErrorAssay')
