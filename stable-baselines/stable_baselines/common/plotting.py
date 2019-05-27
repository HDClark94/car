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

def plot_network_activation(layer_behaviour, behaviour, save_path, title):
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    fig_l2, ax_l2 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    fig_l3, ax_l3 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    fig_l4, ax_l4 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone

    count = 0
    for i in range(8):
        for j in range(8):
            activations = last_trial_layers[:, :, :, count]

            ax_l1[i, j].scatter(pos, activations[:, 0])
            ax_l2[i, j].scatter(pos, activations[:, 1])
            ax_l3[i, j].scatter(pos, activations[:, 2])
            ax_l4[i, j].scatter(pos, activations[:, 3])
            count += 1

            ax_l1[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l2[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l3[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l4[i, j].set(xlabel='Track Position', ylabel= "Unit activation")

            ax_l1[i, j].set_xlim([-0.6, 1])  # track limits
            ax_l2[i, j].set_xlim([-0.6, 1])  # track limits
            ax_l3[i, j].set_xlim([-0.6, 1])  # track limits
            ax_l4[i, j].set_xlim([-0.6, 1])  # track limits

            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l1[i, j].fill(x, y, color="k", alpha=0.2)
            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l2[i, j].fill(x, y, color="k", alpha=0.2)
            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l3[i, j].fill(x, y, color="k", alpha=0.2)
            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l4[i, j].fill(x, y, color="k", alpha=0.2)

            ax_l1[i, j].set_ylim([-1.1, 1.1])  # track limits
            ax_l2[i, j].set_ylim([-1.1, 1.1])  # track limits
            ax_l3[i, j].set_ylim([-1.1, 1.1])  # track limits
            ax_l4[i, j].set_ylim([-1.1, 1.1])  # track limits

    fig_l1.tight_layout()
    fig_l2.tight_layout()
    fig_l3.tight_layout()
    fig_l4.tight_layout()

    fig_l1.savefig(save_path + title + "l1_pn")
    fig_l2.savefig(save_path + title + "l1_vn")
    fig_l3.savefig(save_path + title + "l2_pn")
    fig_l4.savefig(save_path + title + "l2_vn")

    fig_l1.clf()
    fig_l2.clf()
    fig_l3.clf()
    fig_l4.clf()



def plot_summary_with_fn(behaviour, actions, values, save_path, title):

    # TODO add plots for actions and values of last trial

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    raster(behaviour, ax1)
    #accum_reward(behaviour, ax2)
    speed_of_last(behaviour, ax3)
    average_ep_reward(behaviour, ax2)
    #actions_of_last(behaviour, actions, ax5)
    value_fn_of_last(behaviour, values, ax4)
    f.tight_layout()
    #plt.show()

    f.savefig(save_path + title)

    f.clf()


def actions_of_last(behaviour, actions, ax=None):
    ax.set(xlabel="Position", ylabel="Action Selected")
    ax.set_xlim([-0.6, 1])  # track limits

    last_trial = np.array(behaviour[-1])
    last_trial_actions = np.array(actions[-1])

    pos = [i[0] for i in last_trial[:, 4]]  # vector of positions

    return ax.plot(pos, last_trial_actions, color='k')


def value_fn_of_last(behaviour, values, ax=None):
    ax.set(xlabel="Position", ylabel="Value")
    ax.set_xlim([-0.6, 1])  # track limits

    last_trial = np.array(behaviour[-1])
    last_trial_values = np.array(values[-1])

    pos = [i[0] for i in last_trial[:, 4]]  # vector of positions

    ymin = min(last_trial_values) - 0.1*max(last_trial_values)
    ymax = max(last_trial_values) + 0.1*max(last_trial_values)
    ax.set_ylim([ymin, ymax])

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [ymin, ymin, ymax, ymax]
    ax.fill(x, y, color="k", alpha=0.2)

    return ax.plot(pos, last_trial_values, color='k')

def average_ep_reward(behaviour, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Trial', ylabel='Episode Reward')

    no_trials = len(behaviour)
    ax.set_xlim([1, no_trials])

    ep_rews=[]

    for trial in behaviour:
        trial = np.array(trial)

        rews = [i for i in trial[:, 1]]  # vector of reward in trial
        ep_rews.append(np.sum(rews))

    n = 10
    # plots every n episode rewards to avoid over busy plot
    ep_rews = np.array(ep_rews)[0::n]
    every_n = np.arange(1, no_trials + 1)[0::n]

    return ax.plot(every_n, ep_rews,  color='k')


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
    ax.set(xlabel='Track Position', ylabel='Speed')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [0, 0, 0.25, 0.25]
    ax.fill(x, y, color="k", alpha=0.2)
    ax.set_xlim([-0.6, 1])  # track limits
    ax.set_ylim([0, 0.25])

    last_trial = np.array(behaviour[-1])

    v = [i[1] for i in last_trial[:, 4]]  # vector of velocities
    pos = [i[0] for i in last_trial[:, 4]] # vector of positions

    return ax.plot(pos, v, color='k')

def raster(behaviour, ax=None):

    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    no_trials = len(behaviour)
    all_trial_stopping = []    # initialise empty list

    # changes structure so behaviour is organised by trial
    for trial in behaviour:
        trial = np.array(trial)

        v = [i[1] for i in trial[:, 4]]  # vector of states per time step in trial (velocity)
        idx = np.array(np.round(np.array(v), decimals=3) == 0.0)  # vector of boolean per time step showing v = 0
        pos = np.array([i[0] for i in trial[:, 4]])  # vector of positions
        all_trial_stopping.append(pos[idx])  # appendable list of positions for which v = 0
        #print(pos[idx], "=pos[idx]")

    #plt.title(title)
    ax.set(xlabel='Track Position', ylabel='Trial')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [1, 1, no_trials, no_trials]
    ax.fill(x, y, color="k", alpha=0.2)

    ax.set_ylim([1, no_trials])
    ax.set_xlim([-0.6, 1])  # track limits

    #print(np.shape(all_trial_stopping), "shape of all_trial_stopping")
    #print("LAST = ", all_trial_stopping[-1], "  last ", behaviour[-1])

    # Draw a spike raster plot
    return ax.eventplot(all_trial_stopping, linewidths=5, color='k')


