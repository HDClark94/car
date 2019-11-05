import numpy as np
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
'''
def evaluate_policy_vecenv(model, env):

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

'''

def evaluate_policy_vecenv(model, env, seed=None):

    if model.policy == MlpPolicy:

        obs, done = env.reset(), False
        episode_rew = 0
        episode_log = []
        layer_log = []
        action_log = []
        value_log = []
        trial_type = env.envs[0].env.trialtype

        while not done:
            #env.render()
            action, _, layers_list, value = model.predict(obs, deterministic=True)
            obs, rew, done, state = env.step(action)

            # store log for greedy
            episode_log.append([action[0], rew[0], obs[0], float(done[0]), state[0]])
            layer_log.append(layers_list)
            action_log.append(action)
            value_log.append(value)
            episode_rew += rew

        #print(episode_rew, "=episode reward")

        env.close()
        return episode_log, episode_rew, layer_log, action_log, value_log, trial_type


    elif model.policy == MlpLstmPolicy:

        obs, done = env.reset(), False

        episode_rew = 0
        episode_log = []
        layer_log = []
        action_log = []
        value_log = []

        #trial_type = env.get_attr('trialtype')[0]
        trial_type = env.envs[0].env.trialtype

        _state = None

        while not done:
            #env.render()
            action, _state, layers_list, value = model.predict(obs, state=_state, mask=[done], deterministic=True)
            obs, rew, done, state = env.step(action)

            # store log for greedy
            episode_log.append([action[0], rew[0], obs[0], float(done[0]), state[0]])
            layer_log.append(layers_list)
            action_log.append(action)
            value_log.append(value)
            episode_rew += rew

            done = done[0] # were only interested in the first vectorized environment

        #print(episode_rew, "=episode reward")

        env.close()
        return episode_log, episode_rew, layer_log, action_log, value_log, trial_type