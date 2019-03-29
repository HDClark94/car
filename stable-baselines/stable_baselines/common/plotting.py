import time
import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
from stable_baselines import logger
from stable_baselines.a2c.utils import batch_to_seq, seq_to_batch, Scheduler, find_trainable_variables, EpisodeStats, \
    get_by_index, check_shape, avg_norm, gradient_add, q_explained_variance, total_episode_reward_logger
from stable_baselines.acer.buffer import Buffer
from stable_baselines.common import ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import LstmPolicy, ActorCriticPolicy


def raster(behaviour, save_path, title):
    # behaviour is var_log with vector entries [action, rew, obs, float(done), tn]

    no_trials = len(behaviour) # picks trial number of last trial (number of trials)
    all_trial_stopping = []    # initialise empty list

    # changes structure so behaviour is organised by trial
    for trial in behaviour:
        trial = np.array(trial)

        v = [i[1] for i in trial[:, 4]]  # vector of states per time step in trial (velocity)
        idx = np.array(np.array(v) == 0.0)  # vector of boolean per time step showing v = 0
        pos = np.array([i[0] for i in trial[:, 4]])  # vector of positions for which v = 0
        all_trial_stopping.append(pos[idx])  # appendable list of positions for which v = 0

    # Draw a spike raster plot
    plt.eventplot(all_trial_stopping, linelengths=1, linewidths=5, color='k')

    #plt.title(title)
    plt.xlabel('Track Position')
    plt.ylabel('Trial')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [0, 0, no_trials, no_trials]
    plt.fill(x, y, color="k", alpha=0.2)

    plt.ylim([0, no_trials])
    plt.xlim([-0.6, 1])  # track limits

    plt.savefig(save_path+title)
    plt.close()

