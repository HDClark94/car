import numpy as np
import gym

env = gym.make('MountainCarContinuous_Action-v0')

env.set_obs_error(0.1)
env.seed(0)
env.reset()

first = env.step(1)[0][0]
print(first)

env.set_obs_error(0.1)
env.seed(0)
env.reset()

second = env.step(1)[0][0]
print(second)

if first == second:
    print("passed")