import matplotlib.pyplot as plot
#%matplotlib inline
import gym
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import ACER
import numpy as np

action_errors = [0, 0.0001, 0.001, 0.01, 0.1]
training_steps = 200000
evalfreq = 10000
rewards = []

for std in action_errors:
    std_rewards = []
    for i in range(4):
        print(std, "=std")

        # with error
        # multiprocess environment
        n_cpu = 4
        env = SubprocVecEnv([lambda: gym.make('MountainCar-v0') for i in range(n_cpu)])

        model = ACER(MlpPolicy, env, verbose=0, action_error_std=std, actiondim=2, evalfreq=evalfreq)
        model.learn(total_timesteps=training_steps)
        model.save("acer_mountain")

        std_rewards.append(model.greedy_rewards)

        del model # remove to demonstrate saving and loading
    rewards.append(std_rewards)

means = np.mean(np.array(rewards), axis=1)
stds = np.std(np.array(rewards), axis=1)

for i in range(len(means)):
    std = action_errors[i]
    plot.errorbar(np.arange(evalfreq,training_steps+evalfreq, evalfreq), means[i], yerr=stds[i], label=str(std))

plot.title('Action Error Assay')
plot.legend()
plot.xlabel('Training steps')
plot.ylabel('Average Episode Returns')
plot.show()
