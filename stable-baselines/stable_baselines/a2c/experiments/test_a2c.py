import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

env_string = 'MountainCar-v0'

# multiprocess environment
n_cpu = 4
env = SubprocVecEnv([lambda: gym.make('MountainCar-v0') for i in range(n_cpu)])

model = A2C(MlpPolicy, env, verbose=0, action_error_std=0, actiondim=2)
model.learn(total_timesteps=25000, eval_env_string=env_string)

model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading
model = A2C.load("a2c_cartpole")

n_cpu = 1
env = SubprocVecEnv([lambda: gym.make('MountainCar-v0') for i in range(n_cpu)])
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    print(action)
    print(rewards)
