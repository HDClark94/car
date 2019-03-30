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

def plot_summary(behaviour, save_path, title):

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
    raster(behaviour, ax1)
    accum_reward(behaviour, ax2)
    speed_of_last(behaviour, ax3)
    average_ep_reward(behaviour, ax4)
    #plt.show()

    f.savefig(save_path + title)

def average_ep_reward(behaviour, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Trial', ylabel='Average Reward')

    no_trials = len(behaviour)
    ax.set_xlim([1, no_trials])

    ep_rews=[]

    for trial in behaviour:
        trial = np.array(trial)

        rews = [i for i in trial[:, 1]]  # vector of reward in trial
        ep_rews.append(np.sum(rews))


    return ax.plot(np.arange(1,no_trials+1), ep_rews,  color='k')


def accum_reward(behaviour, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Trial', ylabel= 'Cummulative Rewards')

    no_trials = len(behaviour)
    ax.set_xlim([1, no_trials])

    i = 0
    accum_rew = []
    for trial in behaviour:
        trial = np.array(trial)

        rews = [i for i in trial[:, 1]]  # vector of reward in trial
        if i==0:
            accum_rew.append(np.sum(rews))
        else:
            accum_rew.append(np.sum(rews)+accum_rew[-1])
        i+=1

    return ax.plot(np.arange(1,no_trials+1), accum_rew,  color='k')


def speed_of_last(behaviour, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Track Position', ylabel='Trial')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [0, 0, 0.2, 0.2]
    ax.fill(x, y, color="k", alpha=0.2)
    ax.set_xlim([-0.6, 1])  # track limits
    ax.set_ylim([0, 0.2])

    last_trial = np.array(behaviour[-1])

    v = [i[1] for i in last_trial[:, 4]]  # vector of velocities
    pos = np.array([i[0] for i in last_trial[:, 4]])  # vector of positions

    return ax.plot(pos, v, color='k')

def raster(behaviour, ax=None):

    # behaviour is var_log with vector entries [action, rew, obs, float(done), tn]
    no_trials = len(behaviour)
    all_trial_stopping = []    # initialise empty list

    # changes structure so behaviour is organised by trial
    for trial in behaviour:
        trial = np.array(trial)

        v = [i[1] for i in trial[:, 4]]  # vector of states per time step in trial (velocity)
        idx = np.array(np.array(v) == 0.0)  # vector of boolean per time step showing v = 0
        pos = np.array([i[0] for i in trial[:, 4]])  # vector of positions
        all_trial_stopping.append(pos[idx])  # appendable list of positions for which v = 0

    #plt.title(title)
    ax.set(xlabel='Track Position', ylabel='Trial')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [1, 1, no_trials, no_trials]
    ax.fill(x, y, color="k", alpha=0.2)

    ax.set_ylim([1, no_trials])
    ax.set_xlim([-0.6, 1])  # track limits

    # Draw a spike raster plot
    return ax.eventplot(all_trial_stopping, linelengths=1, linewidths=5, color='k')

