import numpy as np

def evaluate_policy(model, env, seed=None):
    '''
    :param model: stable-baselines policy, valid for environment to be evaluated
    :param env:   open ai gym environment
    :return:      logs for episode including action, reward, observation, done and state per time step
                  logs for episode reward per episode
    '''

    env.set_obs_error(model.action_error_std)
    env.seed(seed)

    obs, done = env.reset(), False
    episode_rew = 0
    episode_log = []
    layer_log = []
    action_log = []
    value_log = []
    trial_type = env.trialtype

    while not done:
        # env.render()
        action, _, layers_list, value = model.predict(obs, deterministic=True)
        obs, rew, done, state = env.step(action)

        # store log for greedy
        episode_log.append([action, rew, obs, float(done), state])
        layer_log.append(layers_list)
        action_log.append(action)
        value_log.append(value)
        episode_rew += rew

    #print(episode_rew, "=episode reward")

    env.close()
    return episode_log, episode_rew, layer_log, action_log, value_log, trial_type