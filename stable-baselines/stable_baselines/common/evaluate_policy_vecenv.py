import numpy as np
from stable_baselines.common.vec_env import SubprocVecEnv

def evaluate_policy_vecenv(model, env):
    '''
    :param model: stable-baselines policy, valid for environment to be evaluated
    :param env:   open ai gym environment
    :return:      logs for episode including action, reward, observation, done and state per time step
                  logs for episode reward per episode
    '''

    obs, done = env.reset(), False
    episode_rew = 0
    episode_log = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, state = env.step(action)

        # store log for greedy
        episode_log.append([action[0], rew[0], obs[0], float(done[0]), state[0]])
        episode_rew += rew
        #env.render()

    #print(episode_rew, "=episode reward")

    #env.close()
    return episode_log, episode_rew