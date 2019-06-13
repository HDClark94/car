import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines import ACER


'''

# For recurrent policies, with PPO2, the number of environments run in parallel
# should be a multiple of nminibatches.
model = ACER('MlpLstmPolicy', 'CartPole-v1', verbose=0)
model.learn(500000)

# Retrieve the env
env = model.get_env()

obs = env.reset()
# Passing state=None to the predict function means
# it is the initial state
state = None

reward_ep = 0
# When using VecEnv, done is a vector
done = [False for _ in range(env.num_envs)]
for _ in range(1000):
    # We need to pass the previous state and a mask for recurrent policies
    # to reset lstm state when a new episode begin
    action, state = model.predict(obs, state=state, mask=done)
    obs, reward, done, _ = env.step(action)
    # Note: with VecEnv, env.reset() is automatically called

    reward_ep += reward
    if (done[0] == True):
        print(reward_ep)
        reward_ep = 0

    # Show the env
    # env.render()   

'''


env = gym.make('CartPole-v1')
#env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
env = DummyVecEnv([lambda: env])
model = ACER(MlpLstmPolicy, env, verbose=1, action_error_std=0)
model.learn(total_timesteps=500000, eval_env_string='CartPole-v1')


