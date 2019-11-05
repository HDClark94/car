import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
#from stable_baselines.common.plotting import *
import os

def plot_network_activation_dqn(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    fig_l2, ax_l2 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone

    count = 0
    for i in range(8):
        for j in range(8):
            activations = last_trial_layers[:, :, :, count]

            ax_l1[i, j].scatter(pos, activations[:, 0])
            ax_l2[i, j].scatter(pos, activations[:, 1])
            count += 1

            ax_l1[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l2[i, j].set(xlabel='Track Position', ylabel= "Unit activation")

            ax_l1[i, j].set_xlim([-0.6, 1])  # track limits
            ax_l2[i, j].set_xlim([-0.6, 1])  # track limits

            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l1[i, j].fill(x, y, color="k", alpha=0.2)
            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l2[i, j].fill(x, y, color="k", alpha=0.2)

            ax_l1[i, j].set_ylim([-1.1, 1.1])  # track limits
            ax_l2[i, j].set_ylim([-1.1, 1.1])  # track limits

    fig_l1.tight_layout()
    fig_l2.tight_layout()

    fig_l1.savefig(save_path + title + "l1_qnet")
    fig_l2.savefig(save_path + title + "l2_qnet")

    fig_l1.clf()
    fig_l2.clf()

def plot_network_activation(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")
    fig_l2, ax_l2 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")
    fig_l3, ax_l3 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")
    fig_l4, ax_l4 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")

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

            if j==0:
                ax_l1[i, j].set(ylabel= "Unit activation")
                ax_l2[i, j].set(ylabel= "Unit activation")
                ax_l3[i, j].set(ylabel= "Unit activation")
                ax_l4[i, j].set(ylabel= "Unit activation")
            if i==7:
                ax_l1[i, j].set(xlabel = 'Track Position')
                ax_l2[i, j].set(xlabel = 'Track Position')
                ax_l3[i, j].set(xlabel = 'Track Position')
                ax_l4[i, j].set(xlabel = 'Track Position')

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

def plot_rasta_test(ep_log, save_path, title):
    # for looking at behaviour of agent within learning env not evaluation env
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    info_ = []
    dones_ = []

    for i in ep_log:
        info_.append(i[0])
        dones_.append(i[1])

    info = [val for sublist in info_ for val in sublist]
    dones = [val for sublist in dones_ for val in sublist]

    ep_log = []

    counter = 0
    ep = []
    for i in dones:
        ep.append(info[counter])
        counter+=1

        if i == True:
            ep_log.append(ep)
            ep = []

    raster_test(ep_log, ax1)
    f.tight_layout()
    f.savefig(save_path + title)
    f.clf()

def raster_test(behaviour, ax=None):

    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    no_trials = len(behaviour)
    all_trial_stopping = []    # initialise empty list

    # changes structure so behaviour is organised by trial
    for trial in behaviour:
        trial = np.array(trial)

        v = trial[:, 1] # vector of states per time step in trial (velocity)
        pos = trial[:, 0] # vector of positions

        idx = np.array(np.round(np.array(v), decimals=3) == 0.0)  # vector of boolean per time step showing v = 0

        all_trial_stopping.append(pos[idx])  # appendable list of positions for which v = 0
        #print(pos[idx], "=pos[idx]")

    #plt.title(title)
    ax.set(xlabel='Track Position', ylabel='Trial')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [1, 1, no_trials, no_trials]
    ax.fill(x, y, color="k", alpha=0.2)

    ax.set_ylim([1, no_trials])
    ax.set_xlim([-0.6, 1])  # track limits

    # Draw a spike raster plot
    return ax.eventplot(all_trial_stopping, linewidths=5)

def plot_summary(behaviour, trialtype_log, save_path, title):

    # TODO add plots for actions and values of last trial

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    raster(behaviour, trialtype_log, ax1)
    #accum_reward(behaviour, ax2)
    speed_of_last(behaviour, trialtype_log, ax3)
    average_ep_reward(behaviour, ax2)
    #actions_of_last(behaviour, actions, ax5)
    #value_fn_of_last(behaviour, values, ax4)
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
    ax.set_xlim([-0.15, 1.15])  # track limits

    last_trial = np.array(behaviour[-1])
    last_trial_values = np.array(values[-1])

    pos = [i[0] for i in last_trial[:, 4]]  # vector of positions

    ymin = min(last_trial_values) - 0.1*max(last_trial_values)
    ymax = max(last_trial_values) + 0.1*max(last_trial_values)
    ax.set_ylim([ymin, ymax])

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [ymin, ymin, ymax, ymax]

    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    style_vr_plot(ax, xbar=ymin, ybar=-0.15)

    return ax.plot(pos, last_trial_values, color='k')

def average_ep_reward(behaviour, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Trial', ylabel='Episode Reward')

    no_trials = len(behaviour)
    ax.set_xlim([1, no_trials])
    #ax.set_xlim([2700, 2800])

    max_cum_reward = 59
    min_cum_reward = -200
    ax.axhline(max_cum_reward, color='k', linestyle='dashed', linewidth=1)
    ax.axhline(min_cum_reward, color='k', linestyle='dashed', linewidth=1)

    style_vr_plot(ax, xbar=0, ybar=-0.15)

    no_trials = len(behaviour)
    ax.set_ylim([-210, 70])

    ep_rews=[]

    for trial in behaviour:
        trial = np.array(trial)

        rews = [i for i in trial[:, 1]]  # vector of reward in trial
        ep_rews.append(np.sum(rews))

    n = 10
    n = 11
    # plots every n episode rewards to avoid over busy plot
    ep_rews = np.array(ep_rews)[0::n]
    every_n = np.arange(1, no_trials + 1)[0::n]
    #every_n = np.arange(2700,2801)[0::n]

    return ax.plot(every_n, ep_rews,  color='k')


def accum_reward(behaviour, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Trial', ylabel= 'Cummulative Rewards')

    max_cum_reward = 59
    min_cum_reward = -200
    ax.axhline(max_cum_reward, color='k', linestyle='dashed', linewidth=1)
    ax.axhline(min_cum_reward, color='k', linestyle='dashed', linewidth=1)

    no_trials = len(behaviour)
    ax.set_xlim([0,no_trials])
    ax.set_xlim([750, 850])
    ax.set_ylim([-200, 61])

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

def last_trials_of_trialtype(behaviour, trialtype_log, trialtype_wanted, last_n=5):
    trialtype_log = np.array(trialtype_log)
    idx = np.where(trialtype_log == trialtype_wanted)[-1][-last_n:]  # picks last n elements of array that meets trialtype condition, returns index

    tmp = []
    for i in idx:
        tmp.append(behaviour[i])
    behaviour = tmp

    return behaviour

def speed_of_last(behaviour, trialtype_log, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Track Position', ylabel='Speed')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [0, 0, 0.25, 0.25]
    y = [0, 0, 0.05, 0.05]

    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    style_vr_plot(ax, xbar=0, ybar=-0.15)

    ax.set_xlim([-0.15, 1.15])  # track limits
    ax.set_ylim([0, 0.05])

    last_trials_b = last_trials_of_trialtype(behaviour, trialtype_log, "beaconed")
    last_trials_nb = last_trials_of_trialtype(behaviour, trialtype_log, "non_beaconed")
    last_trials_p = last_trials_of_trialtype(behaviour, trialtype_log, "probe")

    if len(last_trials_b)>0:
        for trial in last_trials_b:
            v = [i[1] for i in np.array(trial)[:, 4]]  # vector of velocities
            pos = [i[0] for i in np.array(trial)[:, 4]] # vector of positions
            ax.plot(pos, v, color='k')

    if len(last_trials_nb) > 0:
        for trial in last_trials_nb:
            v = [i[1] for i in np.array(trial)[:, 4]]  # vector of velocities
            pos = [i[0] for i in np.array(trial)[:, 4]]  # vector of positions
            ax.plot(pos, v, color='k')

    if len(last_trials_p) > 0:
        for trial in last_trials_p:
            v = [i[1] for i in np.array(trial)[:, 4]]  # vector of velocities
            pos = [i[0] for i in np.array(trial)[:, 4]]  # vector of positions
            ax.plot(pos, v, color='k')

def raster(behaviour, trialtype_log, ax=None):

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

    #x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    #y = [1, 1, no_trials, no_trials]
    #ax.fill(x, y, color="k", alpha=0.2)

    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    ax.set_ylim([1, no_trials])
    #ax.set_ylim([2700, 2800])
    ax.set_xlim([-0.15, 1.15])  # track limits

    #print(np.shape(all_trial_stopping), "shape of all_trial_stopping")
    #print("LAST = ", all_trial_stopping[-1], "  last ", behaviour[-1])


    colors = []
    for trialtype in trialtype_log:
        if trialtype == "beaconed":
            colors.append("k")
        elif trialtype == "non_beaconed":
            colors.append("r")
        elif trialtype == "probe":
            colors.append("b")
        else:
            print("trial type does not match string beaconed probe or non_beaconed")

    style_vr_plot(ax, xbar=0, ybar=-0.15)

    for i in range(no_trials):
        stops = np.unique(all_trial_stopping[i])
        stops = stops[stops<=1.0]

        if trialtype_log[i] == "beaconed":
            #i = i + 2700
            ax.plot(stops, np.ones(len(stops))*i, 'o', color='0.5', markersize=2)
        elif trialtype_log[i] == "non_beaconed":
            #i = i + 2700
            ax.plot(stops, np.ones(len(stops))*i, 'o', color='red', markersize=2)

    #ax.eventplot(all_trial_stopping, linewidths=5, color=colors)

    # Draw a spike raster plot
    return ax
    #return ax.eventplot(all_trial_stopping, linewidths=5, color=colors)

def style_vr_plot(ax, xbar, ybar):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=True,
        labelleft=True,
        labelbottom=True)  # labels along the bottom edge are off

    #ax.set_aspect('equal')

    #ax.axvline(ybar, linewidth=2.5, color='black') # bold line on the y axis
    #ax.axhline(xbar, linewidth=2.5, color='black') # bold line on the x axis

def plot_activation_gradients(behaviour, layer_behaviour, trialtype_log, save_path, title):
    f, ax1 = plt.subplots()

    #f.savefig(save_path + title, dpi=1000)
    #f.clf()

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)
    layer_behaviour = np.array(layer_behaviour)

    beaconed_beh = behaviour[trialtype_log == "beaconed"]
    non_beaconed_beh = behaviour[trialtype_log == "non_beaconed"]

    beaconed_layer_beh = layer_behaviour[trialtype_log == "beaconed"]
    non_beaconed_layer_beh = layer_behaviour[trialtype_log == "non_beaconed"]

    bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
    b_binned = [[] for i in range(len(bins))]
    nb_binned = [[] for i in range(len(bins))]

    count = 0

    b_hb_gradients = []
    b_ib_gradients = []
    nb_hb_gradients = []
    nb_ib_gradients = []


    for i in range(16):
        for j in range(16):

            bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
            b_binned = [[] for k in range(len(bins))]
            nb_binned = [[] for k in range(len(bins))]

            counter = 0
            for trial, trial_layer in zip(beaconed_beh, beaconed_layer_beh):
                trial = np.array(trial)
                trial_layer = np.array(trial_layer)
                trial_unit = trial_layer[:, :, :, count].flatten()

                v = np.array([k[1] for k in np.array(trial)[:, 4]])  # vector of velocities
                pos = np.round(np.array([k[0] for k in trial[:, 4]]), decimals=3)  # vector of

                for m in range(len(pos)):
                    pos_tmp = pos[m]
                    trial_unit_tmp = trial_unit[m]
                    idx = bins.index(pos_tmp)
                    b_binned[idx].append(trial_unit_tmp)

            b_means = []
            b_stds = []
            for k in range(len(b_binned)):
                if len(b_binned[k]) > 0:
                    b_means.append(np.mean(b_binned[k]))
                    b_stds.append(np.std(b_binned[k]))
                else:
                    b_means.append(np.nan)
                    b_stds.append(np.nan)

            for trial, trial_layer in zip(non_beaconed_beh, non_beaconed_layer_beh):
                trial = np.array(trial)
                trial_layer = np.array(trial_layer)
                trial_unit = trial_layer[:, :, :, count].flatten()

                v = np.array([k[1] for k in np.array(trial)[:, 4]])  # vector of velocities
                pos = np.round(np.array([k[0] for k in trial[:, 4]]), decimals=3)  # vector of

                for k in range(len(pos)):
                    pos_tmp = pos[k]
                    trial_unit_tmp = trial_unit[k]
                    idx = bins.index(pos_tmp)
                    nb_binned[idx].append(trial_unit_tmp)

            nb_means = []
            nb_stds = []
            for k in range(len(nb_binned)):
                if len(nb_binned[k]) > 0:
                    nb_means.append(np.mean(nb_binned[k]))
                    nb_stds.append(np.std(nb_binned[k]))
                else:
                    nb_means.append(np.nan)
                    nb_stds.append(np.nan)

            bins = np.array(bins[0:40])
            b_means = np.array(b_means[0:40])
            b_stds = np.array(b_stds[0:40])
            nb_means = np.array(nb_means[0:40])
            nb_stds = np.array(nb_stds[0:40])


            # now calculate in bound and homebound gradients

            ib_bins = bins[0:16]
            hb_bins = bins[24:40]

            b_ib_means = b_means[0:16]
            b_hb_means = b_means[24:40]

            nb_ib_means = nb_means[0:16]
            nb_hb_means = nb_means[24:40]

            b_ib_gradient = (b_ib_means[-1]-b_ib_means[1])/(ib_bins[-1]-ib_bins[0])
            b_hb_gradient = (b_hb_means[-1]-b_hb_means[1])/(hb_bins[-1]-hb_bins[0])

            nb_ib_gradient = (nb_ib_means[-1]-nb_ib_means[1])/(ib_bins[-1]-ib_bins[0])
            nb_hb_gradient = (nb_hb_means[-1]-nb_hb_means[1])/(hb_bins[-1]-hb_bins[0])

            b_hb_gradients.append(b_hb_gradient)
            b_ib_gradients.append(b_ib_gradient)
            nb_hb_gradients.append(nb_hb_gradient)
            nb_ib_gradients.append(nb_ib_gradient)

            count += 1

    b_hb_gradients = np.array(b_hb_gradients)
    b_ib_gradients = np.array(b_ib_gradients)
    nb_hb_gradients = np.array(nb_hb_gradients)
    nb_ib_gradients = np.array(nb_ib_gradients)

    ax1.scatter(b_hb_gradients, b_ib_gradients, marker="x", color="k", label="Beaconed")
    ax1.scatter(nb_hb_gradients, nb_ib_gradients, marker="x", color="r", label="Non-beaconed")

    ax1.set_xlabel("Homebound Gradient")
    ax1.set_ylabel("Inbound Gradient")
    ax1.legend()

    f.tight_layout()
    f.savefig(save_path + title + "l1_pn", dpi=700)
    f.clf()



def plot_summary_with_fn(behaviour, values, trialtype_log, save_path, title):

    # TODO add plots for actions and values of last trial

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    raster(behaviour, trialtype_log, ax1)
    #accum_reward(behaviour, ax2)
    #speed_of_last(behaviour, trialtype_log, ax3)
    average_ep_reward(behaviour, ax2)
    #actions_of_last(behaviour, actions, ax5)
    #value_fn_of_last(behaviour, values, ax4)
    average_value_function(behaviour, trialtype_log, values, ax4)
    average_speed(behaviour, trialtype_log, ax3)

    f.tight_layout()
    #plt.show()

    f.savefig(save_path + title, dpi=1000)

    f.clf()

def average_speed(behaviour, trialtype_log, ax=None):
    ax.set(xlabel="Track Position", ylabel="Average Velocity")
    ax.set_xlim([-0.15, 1.15])  # track limits

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)

    beaconed_beh = behaviour[trialtype_log == "beaconed"]
    non_beaconed_beh = behaviour[trialtype_log == "non_beaconed"]

    bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
    b_binned = [[] for i in range(len(bins))]
    nb_binned = [[] for i in range(len(bins))]

    for trial in beaconed_beh:
        trial = np.array(trial)
        v = np.array([i[1] for i in np.array(trial)[:, 4]])  # vector of velocities
        pos = np.round(np.array([i[0] for i in trial[:, 4]]), decimals=3)  # vector of

        for j in range(len(pos)):
            pos_tmp = pos[j]
            v_tmp = v[j]
            idx = bins.index(pos_tmp)
            b_binned[idx].append(v_tmp)

    b_means = []
    b_stds = []
    for i in range(len(b_binned)):
        if len(b_binned[i])>0:
            b_means.append(np.mean(b_binned[i]))
            b_stds.append(np.std(b_binned[i]))
        else:
            b_means.append(np.nan)
            b_stds.append(np.nan)

    for trial in non_beaconed_beh:
        trial = np.array(trial)
        v = np.array([i[1] for i in np.array(trial)[:, 4]])  # vector of velocities
        pos = np.round(np.array([i[0] for i in trial[:, 4]]), decimals=3)  # vector of

        for j in range(len(pos)):
            pos_tmp = pos[j]
            v_tmp = v[j]
            idx = bins.index(pos_tmp)
            nb_binned[idx].append(v_tmp)

    nb_means = []
    nb_stds = []
    for i in range(len(nb_binned)):
        if len(nb_binned[i])>0:
            nb_means.append(np.mean(nb_binned[i]))
            nb_stds.append(np.std(nb_binned[i]))
        else:
            nb_means.append(np.nan)
            nb_stds.append(np.nan)


    bins = np.array(bins[0:40])
    b_means = np.array(b_means[0:40])
    b_stds = np.array(b_stds[0:40])
    nb_means = np.array(nb_means[0:40])
    nb_stds = np.array(nb_stds[0:40])

    print("now make plot")
    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    style_vr_plot(ax, xbar=-20, ybar=-0.15)

    ax.plot(bins, b_means, color="k")
    ax.fill_between(bins, b_means - b_stds, b_means + b_stds, facecolor="k", alpha=0.3)
    ax.plot(bins, nb_means, color="r")
    ax.fill_between(bins, nb_means - nb_stds, nb_means + nb_stds, facecolor="r", alpha=0.3)

    return ax

def first_stop_trialblocks(behaviour, trialtype_log, plot_path, title):

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)

    beaconed_beh = behaviour[trialtype_log == "beaconed"]
    non_beaconed_beh = behaviour[trialtype_log == "non_beaconed"]

    counter = 0
    b_avg_first_stops = []
    b_std_first_stops = []
    nb_avg_first_stops = []
    nb_std_first_stops = []

    b_first_stops = []
    block = []
    counter = 0
    for trial in beaconed_beh:
        already_stopped = False
        trial = np.array(trial)

        v = np.array([i[1] for i in np.array(trial)[:, 4]])  # vector of velocities
        pos = np.array([i[0] for i in trial[:, 4]])  # vector of positions

        for j in range(len(pos)):
            if (np.round(v[j], decimals=3) == 0) and already_stopped==False:
                first_stop = pos[j]
                already_stopped = True

        block.append(first_stop)

        if (counter!=0) and (counter%80==0):
            b_first_stops.append(block)
            block=[]
        counter+=1

    nb_first_stops = []
    block = []
    counter = 0
    for trial in non_beaconed_beh:
        already_stopped = False
        trial = np.array(trial)

        v = np.array([i[1] for i in np.array(trial)[:, 4]])  # vector of velocities
        pos = np.array([i[0] for i in trial[:, 4]])  # vector of positions

        for j in range(len(pos)):
            if (np.round(v[j], decimals=3) == 0) and already_stopped == False:
                first_stop = pos[j]
                already_stopped = True
        block.append(first_stop)

        if (counter != 0) and (counter % 20 == 0):
            nb_first_stops.append(block)
            block = []
        counter += 1

    b_means = []
    b_stds = []
    for block in b_first_stops:
        b_means.append(np.mean(np.array(block)))
        b_stds.append(np.std(np.array(block)))

    nb_means = []
    nb_stds = []
    for block in nb_first_stops:
        nb_means.append(np.mean(np.array(block)))
        nb_stds.append(np.std(np.array(block)))

    # now plot

    fig = plt.figure(figsize=(8, 4))  # make figure, this shape (width, height)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom')  # title
    ax.axhspan(0.4, 0.6, facecolor='g', alpha=0.15, linewidth=0)  # green box of reward zone
    ax.axhspan(0, 0.1, facecolor='k', alpha=0.15, linewidth=0)  # black box (4 is used for appearance)
    #ax.axhline(4, linewidth=1, ls='--', color='black')  # mark black box border
    #ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
    #ax.axhline(2, linewidth=3, color='black')  # bold line on the x axis

    ax.plot(np.arange(0,len(b_means)), b_means, 'o', color='0.3', label='Non reward zone score', linewidth=2, markersize=6,
            markeredgecolor='black')
    ax.errorbar(np.arange(0, len(b_means)), b_means, b_stds, fmt='o', color='0.3', capsize=1.5, markersize=2, elinewidth=1.5)
    ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7,
                   labelsize=15)  # tick parameters: pad is tick label relative to
    ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=15)
    ax.set_xlim(0, len(b_means))
    ax.set_ylim(0, 0.6)
    adjust_spines(ax, ['left', 'bottom'])  # remove right and top axis
    plt.locator_params(nbins=3, axis='y')  # set tick number on y axis
    plt.locator_params(nbins=10, axis='x')  # set tick number on x axis
    ax = plt.ylabel('First Stop location', fontsize=12, labelpad=18)
    ax = plt.xlabel('Training Blocks (100 Trials)', fontsize=12, labelpad=14)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom')
    ax.axhspan(0.4, 0.6, facecolor='g', alpha=0.15,
               linewidth=0)  # green box spanning the rewardzone - to mark reward zone
    ax.axhspan(0, 0.1, facecolor='k', alpha=0.15, linewidth=0)  # black box
    ax.axhline(4, linewidth=1, ls='--', color='black')  # mark black box border
    #ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
    #ax.axhline(2, linewidth=3, color='black')  # bold line on the x axis

    ax.plot(np.arange(0,len(nb_means)), nb_means, 'o', color='0.3', label='Non reward zone score', linewidth=2, markersize=6,
            markeredgecolor='black')
    ax.errorbar(np.arange(0,len(nb_means)), nb_means, nb_stds, fmt='o', color='0.3', capsize=1.5, markersize=2, elinewidth=1.5)
    ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=15)
    ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=15)
    ax.set_xlim(0, len(nb_means))
    ax.set_ylim(0, 0.6)
    adjust_spines(ax, ['left', 'bottom'])  # remove right and top axis
    plt.locator_params(nbins=3, axis='y')  # set tick number on y axis
    plt.locator_params(nbins=10, axis='x')  # set tick number on x axis
    ax = plt.xlabel('Training Blocks (100 Trials)', fontsize=12, labelpad=14)

    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.25, left=0.15, right=0.82, top=0.85)

    save_path = plot_path+title
    fig.savefig(save_path, dpi=200)
    plt.close()


def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0)) # outward by 10 points
        #spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])



def average_value_function(behaviour, trialtype_log, values, ax=None):
    ax.set(xlabel="Track Position", ylabel="Expected Reward")
    ax.set_xlim([-0.15, 1.15])  # track limits

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)
    values = np.array(values)

    beaconed_beh = behaviour[trialtype_log=="beaconed"]
    non_beaconed_beh = behaviour[trialtype_log=="non_beaconed"]

    beaconed_vals = values[trialtype_log=="beaconed"]
    non_beaconed_vals = values[trialtype_log=="non_beaconed"]

    all_beaconed_pos = []
    all_beaconed_vals = []

    counter = 0
    for trial in beaconed_beh:
        trial = np.array(trial)
        vals = np.array(beaconed_vals[counter])
        pos = np.array([i[0] for i in trial[:, 4]]) # vector of positions
        unique_pos, unique_ind = np.unique(pos, return_index=True)
        pos = pos[unique_ind]
        all_beaconed_pos.append(pos)
        all_beaconed_vals.append(vals[unique_ind])
        counter+=1

    max_l = max(len(l) for l in all_beaconed_pos)

    b_max_pos_arr = np.array([0])
    for trial_pos in all_beaconed_pos:
        if len(trial_pos)>len(b_max_pos_arr):
            b_max_pos_arr = trial_pos

    tmp = np.empty((len(all_beaconed_vals),max_l))
    tmp[:] = np.nan
    counter = 0
    for vals in all_beaconed_vals:
        tmp[counter][0:len(vals)] = vals[:,0]
        counter+=1
    # TODO: tmp array isn't perfectly alligned for examples where agent doesnt move immedicately

    b_mean = np.nanmean(tmp, axis=0)
    b_std = np.nanstd(tmp, axis=0)

    all_non_beaconed_pos = []
    all_non_beaconed_vals = []

    counter = 0
    for trial in non_beaconed_beh:
        trial = np.array(trial)
        vals = np.array(non_beaconed_vals[counter])
        pos = np.array([i[0] for i in trial[:, 4]])  # vector of positions
        unique_pos, unique_ind = np.unique(pos, return_index=True)
        pos = pos[unique_ind]
        all_non_beaconed_pos.append(pos)
        all_non_beaconed_vals.append(vals[unique_ind])
        counter += 1

    max_l = max(len(l) for l in all_non_beaconed_pos)

    nb_max_pos_arr = np.array([0])
    for trial_pos in all_non_beaconed_pos:
        if len(trial_pos) > len(nb_max_pos_arr):
            nb_max_pos_arr = trial_pos

    tmp = np.empty((len(all_non_beaconed_vals), max_l))
    tmp[:] = np.nan
    counter = 0
    for vals in all_non_beaconed_vals:
        tmp[counter][0:len(vals)] = vals[:, 0]
        counter += 1
    # TODO: tmp array isn't perfectly alligned for examples where agent doesnt move immedicately

    nb_mean = np.nanmean(tmp, axis=0)
    nb_std = np.nanstd(tmp, axis=0)

    print("now make plot")
    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    style_vr_plot(ax, xbar=-20, ybar=-0.15)

    ax.plot(b_max_pos_arr[0:40], b_mean[0:40], color="k")
    ax.fill_between(b_max_pos_arr[0:40], b_mean[0:40]-b_std[0:40], b_mean[0:40]+b_std[0:40], facecolor="k", alpha= 0.3)
    ax.plot(nb_max_pos_arr[0:40], nb_mean[0:40], color="r")
    ax.fill_between(nb_max_pos_arr[0:40], nb_mean[0:40]-nb_std[0:40], nb_mean[0:40]+nb_std[0:40], facecolor="r", alpha=0.3)

    return ax

def plot_network_activation_rnn(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    last_trialtype = trialtype_log[-1]

    # search for last non beaconed and beaconed
    last_beaconed_log = None
    last_non_beaconed_log = None

    while (last_beaconed_log == None):
        for i in range(1, 11):
            if trialtype_log[-i] == "beaconed":
                last_beaconed_log = np.array(behaviour[-i])
                last_b_layers = np.array(layer_behaviour[-i])
                b_pos = [j[0] for j in last_beaconed_log[:, 4]]

    while (last_non_beaconed_log == None):
        for i in range(1, 11):
            if trialtype_log[-i] == "non_beaconed":
                last_non_beaconed_log = np.array(behaviour[-i])
                last_nb_layers = np.array(layer_behaviour[-i])
                nb_pos = [j[0] for j in last_non_beaconed_log[:, 4]]

    print(last_trialtype, "=last_trialtype")
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(16, 16, sharex=True, sharey=True, figsize=(20, 20))

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [-1.2, -1.2, 1.2, 1.2]

    count = 0
    for i in range(16):
        for j in range(16):
            activations = last_trial_layers[:, :, :, count]



            b_activations = last_b_layers[:, :, :, count]
            nb_activations = last_nb_layers[:, :, :, count]

            ax_l1[i, j].plot(b_pos, b_activations[:, 0], '-', color='Black')
            ax_l1[i, j].plot(nb_pos, nb_activations[:, 0], '-', color='Red')

            #ax_l1[i, j].plot(pos, activations[:, 0], '-', color='Black')

            if j==0:
                ax_l1[i, j].set(ylabel= "Unit activation")
            if i==15:
                ax_l1[i, j].set(xlabel = 'Track Position')

            #ax_l1[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l1[i, j].set_xlim([-0.15, 1.15])  # track limits
            ax_l1[i, j].axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
            ax_l1[i, j].axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
            ax_l1[i, j].axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

            ax_l1[i, j].fill(x, y, color="k", alpha=0.2)
            ax_l1[i, j].set_ylim([-1.1, 1.1])  # track limits

            count += 1
    fig_l1.tight_layout()
    fig_l1.savefig(save_path + title + "l1_pn", dpi=700)
    fig_l1.clf()

def plot_average_network_activation_rnn(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)
    layer_behaviour = np.array(layer_behaviour)

    beaconed_beh = behaviour[trialtype_log == "beaconed"]
    non_beaconed_beh = behaviour[trialtype_log == "non_beaconed"]

    beaconed_layer_beh = layer_behaviour[trialtype_log == "beaconed"]
    non_beaconed_layer_beh = layer_behaviour[trialtype_log == "non_beaconed"]

    bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
    b_binned = [[] for i in range(len(bins))]
    nb_binned = [[] for i in range(len(bins))]

    fig_l1, ax_l1 = plt.subplots(16, 16, sharex=True, sharey=True, figsize=(20, 20))

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [-1.2, -1.2, 1.2, 1.2]

    count = 0
    for i in range(16):
        for j in range(16):

            bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
            b_binned = [[] for k in range(len(bins))]
            nb_binned = [[] for k in range(len(bins))]

            counter = 0
            for trial, trial_layer in zip(beaconed_beh, beaconed_layer_beh):
                trial = np.array(trial)
                trial_layer = np.array(trial_layer)
                trial_unit = trial_layer[:,:,:,count].flatten()

                v = np.array([k[1] for k in np.array(trial)[:, 4]])  # vector of velocities
                pos = np.round(np.array([k[0] for k in trial[:, 4]]), decimals=3)  # vector of

                for m in range(len(pos)):
                    pos_tmp = pos[m]
                    trial_unit_tmp = trial_unit[m]
                    idx = bins.index(pos_tmp)
                    b_binned[idx].append(trial_unit_tmp)

            b_means = []
            b_stds = []
            for k in range(len(b_binned)):
                if len(b_binned[k]) > 0:
                    b_means.append(np.mean(b_binned[k]))
                    b_stds.append(np.std(b_binned[k]))
                else:
                    b_means.append(np.nan)
                    b_stds.append(np.nan)

            for trial, trial_layer in zip(non_beaconed_beh, non_beaconed_layer_beh):
                trial = np.array(trial)
                trial_layer = np.array(trial_layer)
                trial_unit = trial_layer[:, :, :, count].flatten()

                v = np.array([k[1] for k in np.array(trial)[:, 4]])  # vector of velocities
                pos = np.round(np.array([k[0] for k in trial[:, 4]]), decimals=3)  # vector of

                for k in range(len(pos)):
                    pos_tmp = pos[k]
                    trial_unit_tmp = trial_unit[k]
                    idx = bins.index(pos_tmp)
                    nb_binned[idx].append(trial_unit_tmp)

            nb_means = []
            nb_stds = []
            for k in range(len(nb_binned)):
                if len(nb_binned[k]) > 0:
                    nb_means.append(np.mean(nb_binned[k]))
                    nb_stds.append(np.std(nb_binned[k]))
                else:
                    nb_means.append(np.nan)
                    nb_stds.append(np.nan)

            bins = np.array(bins[0:40])
            b_means = np.array(b_means[0:40])
            b_stds = np.array(b_stds[0:40])
            nb_means = np.array(nb_means[0:40])
            nb_stds = np.array(nb_stds[0:40])

            ax_l1[i, j].plot(bins, b_means, color="k")
            ax_l1[i, j].fill_between(bins, b_means - b_stds, b_means + b_stds, facecolor="k", alpha=0.3)
            ax_l1[i, j].plot(bins, nb_means, color="r")
            ax_l1[i, j].fill_between(bins, nb_means - nb_stds, nb_means + nb_stds, facecolor="r", alpha=0.3)

            if j == 0:
                ax_l1[i, j].set(ylabel="Unit activation")
            if i == 15:
                ax_l1[i, j].set(xlabel='Track Position')

            # ax_l1[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l1[i, j].set_xlim([-0.15, 1.15])  # track limits
            ax_l1[i, j].axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
            ax_l1[i, j].axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
            ax_l1[i, j].axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

            ax_l1[i, j].fill(x, y, color="k", alpha=0.2)
            ax_l1[i, j].set_ylim([-1.1, 1.1])  # track limits

            count += 1
    fig_l1.tight_layout()
    fig_l1.savefig(save_path + title + "l1_pn", dpi=900)
    fig_l1.clf()

dir = os.path.dirname(__file__)
plot_path = os.path.join(dir, 'figures', 'binary_action', '')

ep_logs =       np.load('/home/harry/car/stable-baselines/stable_baselines/ppo2/experiments/grid_cell/figures/binary_action/id=0131_std=0_i=2ep_logs.npy', allow_pickle=True).tolist()
value_log =     np.load('/home/harry/car/stable-baselines/stable_baselines/ppo2/experiments/grid_cell/figures/binary_action/id=0131_std=0_i=2value_log.npy', allow_pickle=True).tolist()
trialtype_log = np.load('/home/harry/car/stable-baselines/stable_baselines/ppo2/experiments/grid_cell/figures/binary_action/id=0131_std=0_i=2trialtype_log.npy', allow_pickle=True).tolist()
layer_log =     np.load('/home/harry/car/stable-baselines/stable_baselines/ppo2/experiments/grid_cell/figures/binary_action/id=0131_std=0_i=2layer_log.npy', allow_pickle=True).tolist()
title = "for_figure"

first_stop_trialblocks(ep_logs,trialtype_log, plot_path, "for_figurefirststop")

plot_summary_with_fn(ep_logs, value_log, trialtype_log, plot_path, "for_figure_")

plot_summary_with_fn(ep_logs[0:100], value_log[0:100], trialtype_log[0:100], plot_path, "for_figure_first100")
#plot_summary_with_fn(ep_logs[750:850], value_log[750:850], trialtype_log[750:850], plot_path, "for_figure_first750_850")
#plot_summary_with_fn(ep_logs[2700:2800], value_log[2700:2800], trialtype_log[2700:2800], plot_path, "for_figure_last100")

#plot_average_network_activation_rnn(layer_log[0:100], ep_logs[0:100], trialtype_log[0:100], plot_path, title + "_last_trial_layer_first100")
#plot_average_network_activation_rnn(layer_log[2700:2800], ep_logs[2700:2800], trialtype_log[2700:2800], plot_path, title + "_last_trial_layer_last100")
#plot_average_network_activation_rnn(layer_log[750:850], ep_logs[750:850], trialtype_log[750:850], plot_path, title + "_last_trial_layer_last750_850")

#plot_activation_gradients(ep_logs[0:100], layer_log[0:100], trialtype_log[0:100], plot_path, "activation_gradients_first100")
#plot_activation_gradients(ep_logs[2700:2800], layer_log[2700:2800], trialtype_log[2700:2800], plot_path, "activation_gradients_last100")
#plot_activation_gradients(ep_logs[750:850], layer_log[750:850], trialtype_log[750:850], plot_path, "activation_gradients_last750-850")

print("hello there")


'''
this is the whole plotting file that is new




import numpy as np
import matplotlib.pyplot as plt
import os
'''
import time
import gym

import tensorflow as tf
from gym.spaces import Discrete, Box

from stable_baselines import logger
from stable_baselines.a2c.utils import batch_to_seq, seq_to_batch, Scheduler, find_trainable_variables, EpisodeStats, \
    get_by_index, check_shape, avg_norm, gradient_add, q_explained_variance, total_episode_reward_logger
from stable_baselines.acer.buffer import Buffer
from stable_baselines.common import ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import LstmPolicy, ActorCriticPolicy
'''

def plot_network_activation_dqn(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    fig_l2, ax_l2 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone

    count = 0
    for i in range(8):
        for j in range(8):
            activations = last_trial_layers[:, :, :, count]

            ax_l1[i, j].scatter(pos, activations[:, 0])
            ax_l2[i, j].scatter(pos, activations[:, 1])
            count += 1

            ax_l1[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l2[i, j].set(xlabel='Track Position', ylabel= "Unit activation")

            ax_l1[i, j].set_xlim([-0.6, 1])  # track limits
            ax_l2[i, j].set_xlim([-0.6, 1])  # track limits

            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l1[i, j].fill(x, y, color="k", alpha=0.2)
            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l2[i, j].fill(x, y, color="k", alpha=0.2)

            ax_l1[i, j].set_ylim([-1.1, 1.1])  # track limits
            ax_l2[i, j].set_ylim([-1.1, 1.1])  # track limits

    fig_l1.tight_layout()
    fig_l2.tight_layout()

    fig_l1.savefig(save_path + title + "l1_qnet")
    fig_l2.savefig(save_path + title + "l2_qnet")

    fig_l1.clf()
    fig_l2.clf()

def plot_network_activation(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")
    fig_l2, ax_l2 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")
    fig_l3, ax_l3 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")
    fig_l4, ax_l4 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")

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

            if j==0:
                ax_l1[i, j].set(ylabel= "Unit activation")
                ax_l2[i, j].set(ylabel= "Unit activation")
                ax_l3[i, j].set(ylabel= "Unit activation")
                ax_l4[i, j].set(ylabel= "Unit activation")
            if i==7:
                ax_l1[i, j].set(xlabel = 'Track Position')
                ax_l2[i, j].set(xlabel = 'Track Position')
                ax_l3[i, j].set(xlabel = 'Track Position')
                ax_l4[i, j].set(xlabel = 'Track Position')

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

def plot_rasta_test(ep_log, save_path, title):
    # for looking at behaviour of agent within learning env not evaluation env
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    info_ = []
    dones_ = []

    for i in ep_log:
        info_.append(i[0])
        dones_.append(i[1])

    info = [val for sublist in info_ for val in sublist]
    dones = [val for sublist in dones_ for val in sublist]

    ep_log = []

    counter = 0
    ep = []
    for i in dones:
        ep.append(info[counter])
        counter+=1

        if i == True:
            ep_log.append(ep)
            ep = []

    raster_test(ep_log, ax1)
    f.tight_layout()
    f.savefig(save_path + title)
    f.clf()

def raster_test(behaviour, ax=None):

    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    no_trials = len(behaviour)
    all_trial_stopping = []    # initialise empty list

    # changes structure so behaviour is organised by trial
    for trial in behaviour:
        trial = np.array(trial)

        v = trial[:, 1] # vector of states per time step in trial (velocity)
        pos = trial[:, 0] # vector of positions

        idx = np.array(np.round(np.array(v), decimals=3) == 0.0)  # vector of boolean per time step showing v = 0

        all_trial_stopping.append(pos[idx])  # appendable list of positions for which v = 0
        #print(pos[idx], "=pos[idx]")

    #plt.title(title)
    ax.set(xlabel='Track Position', ylabel='Trial')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [1, 1, no_trials, no_trials]
    ax.fill(x, y, color="k", alpha=0.2)

    ax.set_ylim([1, no_trials])
    ax.set_xlim([-0.6, 1])  # track limits

    # Draw a spike raster plot
    return ax.eventplot(all_trial_stopping, linewidths=5)

def plot_summary(behaviour, trialtype_log, save_path, title):

    # TODO add plots for actions and values of last trial

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    raster(behaviour, trialtype_log, ax1)
    #accum_reward(behaviour, ax2)
    speed_of_last(behaviour, trialtype_log, ax3)
    average_ep_reward(behaviour, ax2)
    #actions_of_last(behaviour, actions, ax5)
    #value_fn_of_last(behaviour, values, ax4)
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
    ax.set_xlim([-0.15, 1.15])  # track limits

    last_trial = np.array(behaviour[-1])
    last_trial_values = np.array(values[-1])

    pos = [i[0] for i in last_trial[:, 4]]  # vector of positions

    ymin = min(last_trial_values) - 0.1*max(last_trial_values)
    ymax = max(last_trial_values) + 0.1*max(last_trial_values)
    ax.set_ylim([ymin, ymax])

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [ymin, ymin, ymax, ymax]

    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    style_vr_plot(ax, xbar=ymin, ybar=-0.15)

    return ax.plot(pos, last_trial_values, color='k')

def average_ep_reward(behaviour, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Trial', ylabel='Episode Reward')

    no_trials = len(behaviour)
    ax.set_xlim([1, no_trials])
    #ax.set_xlim([2700, 2800])

    max_cum_reward = 59
    min_cum_reward = -200
    ax.axhline(max_cum_reward, color='k', linestyle='dashed', linewidth=1)
    ax.axhline(min_cum_reward, color='k', linestyle='dashed', linewidth=1)

    style_vr_plot(ax, xbar=0, ybar=-0.15)

    no_trials = len(behaviour)
    ax.set_ylim([-210, 70])

    ep_rews=[]

    for trial in behaviour:
        trial = np.array(trial)

        rews = [i for i in trial[:, 1]]  # vector of reward in trial
        ep_rews.append(np.sum(rews))

    n = 10
    n = 11
    # plots every n episode rewards to avoid over busy plot
    ep_rews = np.array(ep_rews)[0::n]
    every_n = np.arange(1, no_trials + 1)[0::n]
    #every_n = np.arange(2700,2801)[0::n]

    return ax.plot(every_n, ep_rews,  color='k')


def accum_reward(behaviour, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Trial', ylabel= 'Cummulative Rewards')

    max_cum_reward = 59
    min_cum_reward = -200
    ax.axhline(max_cum_reward, color='k', linestyle='dashed', linewidth=1)
    ax.axhline(min_cum_reward, color='k', linestyle='dashed', linewidth=1)

    no_trials = len(behaviour)
    ax.set_xlim([0,no_trials])
    ax.set_xlim([750, 850])
    ax.set_ylim([-200, 61])

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

def last_trials_of_trialtype(behaviour, trialtype_log, trialtype_wanted, last_n=5):
    trialtype_log = np.array(trialtype_log)
    idx = np.where(trialtype_log == trialtype_wanted)[-1][-last_n:]  # picks last n elements of array that meets trialtype condition, returns index

    tmp = []
    for i in idx:
        tmp.append(behaviour[i])
    behaviour = tmp

    return behaviour

def speed_of_last(behaviour, trialtype_log, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Track Position', ylabel='Speed')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [0, 0, 0.25, 0.25]
    y = [0, 0, 0.05, 0.05]

    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    style_vr_plot(ax, xbar=0, ybar=-0.15)

    ax.set_xlim([-0.15, 1.15])  # track limits
    ax.set_ylim([0, 0.05])

    last_trials_b = last_trials_of_trialtype(behaviour, trialtype_log, "beaconed")
    last_trials_nb = last_trials_of_trialtype(behaviour, trialtype_log, "non_beaconed")
    last_trials_p = last_trials_of_trialtype(behaviour, trialtype_log, "probe")

    if len(last_trials_b)>0:
        for trial in last_trials_b:
            v = [i[1] for i in np.array(trial)[:, 4]]  # vector of velocities
            pos = [i[0] for i in np.array(trial)[:, 4]] # vector of positions
            ax.plot(pos, v, color='k')

    if len(last_trials_nb) > 0:
        for trial in last_trials_nb:
            v = [i[1] for i in np.array(trial)[:, 4]]  # vector of velocities
            pos = [i[0] for i in np.array(trial)[:, 4]]  # vector of positions
            ax.plot(pos, v, color='k')

    if len(last_trials_p) > 0:
        for trial in last_trials_p:
            v = [i[1] for i in np.array(trial)[:, 4]]  # vector of velocities
            pos = [i[0] for i in np.array(trial)[:, 4]]  # vector of positions
            ax.plot(pos, v, color='k')

def raster(behaviour, trialtype_log, ax=None):

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

    #x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    #y = [1, 1, no_trials, no_trials]
    #ax.fill(x, y, color="k", alpha=0.2)

    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    ax.set_ylim([1, no_trials])
    #ax.set_ylim([2700, 2800])
    ax.set_xlim([-0.15, 1.15])  # track limits

    #print(np.shape(all_trial_stopping), "shape of all_trial_stopping")
    #print("LAST = ", all_trial_stopping[-1], "  last ", behaviour[-1])


    colors = []
    for trialtype in trialtype_log:
        if trialtype == "beaconed":
            colors.append("k")
        elif trialtype == "non_beaconed":
            colors.append("r")
        elif trialtype == "probe":
            colors.append("b")
        else:
            print("trial type does not match string beaconed probe or non_beaconed")

    style_vr_plot(ax, xbar=0, ybar=-0.15)

    for i in range(no_trials):
        stops = np.unique(all_trial_stopping[i])
        stops = stops[stops<=1.0]

        if trialtype_log[i] == "beaconed":
            #i = i + 2700
            ax.plot(stops, np.ones(len(stops))*i, 'o', color='0.5', markersize=2)
        elif trialtype_log[i] == "non_beaconed":
            #i = i + 2700
            ax.plot(stops, np.ones(len(stops))*i, 'o', color='red', markersize=2)

    #ax.eventplot(all_trial_stopping, linewidths=5, color=colors)

    # Draw a spike raster plot
    return ax
    #return ax.eventplot(all_trial_stopping, linewidths=5, color=colors)

def style_vr_plot(ax, xbar, ybar):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    plt.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        right=False,
        left=True,
        labelleft=True,
        labelbottom=True)  # labels along the bottom edge are off

    #ax.set_aspect('equal')

    #ax.axvline(ybar, linewidth=2.5, color='black') # bold line on the y axis
    #ax.axhline(xbar, linewidth=2.5, color='black') # bold line on the x axis

def plot_activation_gradients(behaviour, layer_behaviour, trialtype_log, save_path, title):
    f, ax1 = plt.subplots()

    #f.savefig(save_path + title, dpi=1000)
    #f.clf()

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)
    layer_behaviour = np.array(layer_behaviour)

    beaconed_beh = behaviour[trialtype_log == "beaconed"]
    non_beaconed_beh = behaviour[trialtype_log == "non_beaconed"]

    beaconed_layer_beh = layer_behaviour[trialtype_log == "beaconed"]
    non_beaconed_layer_beh = layer_behaviour[trialtype_log == "non_beaconed"]

    bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
    b_binned = [[] for i in range(len(bins))]
    nb_binned = [[] for i in range(len(bins))]

    count = 0

    b_hb_gradients = []
    b_ib_gradients = []
    nb_hb_gradients = []
    nb_ib_gradients = []


    for i in range(16):
        for j in range(16):

            bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
            b_binned = [[] for k in range(len(bins))]
            nb_binned = [[] for k in range(len(bins))]

            counter = 0
            for trial, trial_layer in zip(beaconed_beh, beaconed_layer_beh):
                trial = np.array(trial)
                trial_layer = np.array(trial_layer)
                trial_unit = trial_layer[:, :, :, count].flatten()

                v = np.array([k[1] for k in np.array(trial)[:, 4]])  # vector of velocities
                pos = np.round(np.array([k[0] for k in trial[:, 4]]), decimals=3)  # vector of

                for m in range(len(pos)):
                    pos_tmp = pos[m]
                    trial_unit_tmp = trial_unit[m]
                    idx = bins.index(pos_tmp)
                    b_binned[idx].append(trial_unit_tmp)

            b_means = []
            b_stds = []
            for k in range(len(b_binned)):
                if len(b_binned[k]) > 0:
                    b_means.append(np.mean(b_binned[k]))
                    b_stds.append(np.std(b_binned[k]))
                else:
                    b_means.append(np.nan)
                    b_stds.append(np.nan)

            for trial, trial_layer in zip(non_beaconed_beh, non_beaconed_layer_beh):
                trial = np.array(trial)
                trial_layer = np.array(trial_layer)
                trial_unit = trial_layer[:, :, :, count].flatten()

                v = np.array([k[1] for k in np.array(trial)[:, 4]])  # vector of velocities
                pos = np.round(np.array([k[0] for k in trial[:, 4]]), decimals=3)  # vector of

                for k in range(len(pos)):
                    pos_tmp = pos[k]
                    trial_unit_tmp = trial_unit[k]
                    idx = bins.index(pos_tmp)
                    nb_binned[idx].append(trial_unit_tmp)

            nb_means = []
            nb_stds = []
            for k in range(len(nb_binned)):
                if len(nb_binned[k]) > 0:
                    nb_means.append(np.mean(nb_binned[k]))
                    nb_stds.append(np.std(nb_binned[k]))
                else:
                    nb_means.append(np.nan)
                    nb_stds.append(np.nan)

            bins = np.array(bins[0:40])
            b_means = np.array(b_means[0:40])
            b_stds = np.array(b_stds[0:40])
            nb_means = np.array(nb_means[0:40])
            nb_stds = np.array(nb_stds[0:40])


            # now calculate in bound and homebound gradients

            ib_bins = bins[0:16]
            hb_bins = bins[24:40]

            b_ib_means = b_means[0:16]
            b_hb_means = b_means[24:40]

            nb_ib_means = nb_means[0:16]
            nb_hb_means = nb_means[24:40]

            b_ib_gradient = (b_ib_means[-1]-b_ib_means[1])/(ib_bins[-1]-ib_bins[0])
            b_hb_gradient = (b_hb_means[-1]-b_hb_means[1])/(hb_bins[-1]-hb_bins[0])

            nb_ib_gradient = (nb_ib_means[-1]-nb_ib_means[1])/(ib_bins[-1]-ib_bins[0])
            nb_hb_gradient = (nb_hb_means[-1]-nb_hb_means[1])/(hb_bins[-1]-hb_bins[0])

            b_hb_gradients.append(b_hb_gradient)
            b_ib_gradients.append(b_ib_gradient)
            nb_hb_gradients.append(nb_hb_gradient)
            nb_ib_gradients.append(nb_ib_gradient)

            count += 1

    b_hb_gradients = np.array(b_hb_gradients)
    b_ib_gradients = np.array(b_ib_gradients)
    nb_hb_gradients = np.array(nb_hb_gradients)
    nb_ib_gradients = np.array(nb_ib_gradients)

    ax1.scatter(b_hb_gradients, b_ib_gradients, marker="x", color="k", label="Beaconed")
    ax1.scatter(nb_hb_gradients, nb_ib_gradients, marker="x", color="r", label="Non-beaconed")

    ax1.set_xlabel("Homebound Gradient")
    ax1.set_ylabel("Inbound Gradient")
    ax1.legend()

    f.tight_layout()
    f.savefig(save_path + title + "l1_pn", dpi=300)
    f.clf()



def plot_summary_with_fn(behaviour, values, trialtype_log, save_path, title):

    # TODO add plots for actions and values of last trial

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    raster(behaviour, trialtype_log, ax1)
    #accum_reward(behaviour, ax2)
    #speed_of_last(behaviour, trialtype_log, ax3)
    average_ep_reward(behaviour, ax2)
    #actions_of_last(behaviour, actions, ax5)
    #value_fn_of_last(behaviour, values, ax4)
    average_value_function(behaviour, trialtype_log, values, ax4)
    average_speed(behaviour, trialtype_log, ax3)

    f.tight_layout()
    #plt.show()

    f.savefig(save_path + title, dpi=400)

    f.clf()

def average_speed(behaviour, trialtype_log, ax=None):
    ax.set(xlabel="Track Position", ylabel="Average Velocity")
    ax.set_xlim([-0.15, 1.15])  # track limits

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)

    beaconed_beh = behaviour[trialtype_log == "beaconed"]
    non_beaconed_beh = behaviour[trialtype_log == "non_beaconed"]

    bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
    b_binned = [[] for i in range(len(bins))]
    nb_binned = [[] for i in range(len(bins))]

    for trial in beaconed_beh:
        trial = np.array(trial)
        v = np.array([i[1] for i in np.array(trial)[:, 4]])  # vector of velocities
        pos = np.round(np.array([i[0] for i in trial[:, 4]]), decimals=3)  # vector of

        for j in range(len(pos)):
            pos_tmp = pos[j]
            v_tmp = v[j]
            idx = bins.index(pos_tmp)
            b_binned[idx].append(v_tmp)

    b_means = []
    b_stds = []
    for i in range(len(b_binned)):
        if len(b_binned[i])>0:
            b_means.append(np.mean(b_binned[i]))
            b_stds.append(np.std(b_binned[i]))
        else:
            b_means.append(np.nan)
            b_stds.append(np.nan)

    for trial in non_beaconed_beh:
        trial = np.array(trial)
        v = np.array([i[1] for i in np.array(trial)[:, 4]])  # vector of velocities
        pos = np.round(np.array([i[0] for i in trial[:, 4]]), decimals=3)  # vector of

        for j in range(len(pos)):
            pos_tmp = pos[j]
            v_tmp = v[j]
            idx = bins.index(pos_tmp)
            nb_binned[idx].append(v_tmp)

    nb_means = []
    nb_stds = []
    for i in range(len(nb_binned)):
        if len(nb_binned[i])>0:
            nb_means.append(np.mean(nb_binned[i]))
            nb_stds.append(np.std(nb_binned[i]))
        else:
            nb_means.append(np.nan)
            nb_stds.append(np.nan)


    bins = np.array(bins[0:40])
    b_means = np.array(b_means[0:40])
    b_stds = np.array(b_stds[0:40])
    nb_means = np.array(nb_means[0:40])
    nb_stds = np.array(nb_stds[0:40])

    print("now make plot")
    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    style_vr_plot(ax, xbar=-20, ybar=-0.15)

    ax.plot(bins, b_means, color="k")
    ax.fill_between(bins, b_means - b_stds, b_means + b_stds, facecolor="k", alpha=0.3)
    ax.plot(bins, nb_means, color="r")
    ax.fill_between(bins, nb_means - nb_stds, nb_means + nb_stds, facecolor="r", alpha=0.3)

    return ax

def first_stop_trialblocks(behaviour, trialtype_log, plot_path, title):

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)

    beaconed_beh = behaviour[trialtype_log == "beaconed"]
    non_beaconed_beh = behaviour[trialtype_log == "non_beaconed"]

    counter = 0
    b_avg_first_stops = []
    b_std_first_stops = []
    nb_avg_first_stops = []
    nb_std_first_stops = []

    b_first_stops = []
    block = []
    counter = 0
    for trial in beaconed_beh:
        already_stopped = False
        trial = np.array(trial)

        v = np.array([i[1] for i in np.array(trial)[:, 4]])  # vector of velocities
        pos = np.array([i[0] for i in trial[:, 4]])  # vector of positions

        for j in range(len(pos)):
            if (np.round(v[j], decimals=3) == 0) and already_stopped==False:
                first_stop = pos[j]
                already_stopped = True

        block.append(first_stop)

        if (counter!=0) and (counter%80==0):
            b_first_stops.append(block)
            block=[]
        counter+=1

    nb_first_stops = []
    block = []
    counter = 0
    for trial in non_beaconed_beh:
        already_stopped = False
        trial = np.array(trial)

        v = np.array([i[1] for i in np.array(trial)[:, 4]])  # vector of velocities
        pos = np.array([i[0] for i in trial[:, 4]])  # vector of positions

        for j in range(len(pos)):
            if (np.round(v[j], decimals=3) == 0) and already_stopped == False:
                first_stop = pos[j]
                already_stopped = True
        block.append(first_stop)

        if (counter != 0) and (counter % 20 == 0):
            nb_first_stops.append(block)
            block = []
        counter += 1

    b_means = []
    b_stds = []
    for block in b_first_stops:
        b_means.append(np.mean(np.array(block)))
        b_stds.append(np.std(np.array(block)))

    nb_means = []
    nb_stds = []
    for block in nb_first_stops:
        nb_means.append(np.mean(np.array(block)))
        nb_stds.append(np.std(np.array(block)))

    # now plot

    fig = plt.figure(figsize=(8, 4))  # make figure, this shape (width, height)
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Beaconed', fontsize=20, verticalalignment='bottom')  # title
    ax.axhspan(0.4, 0.6, facecolor='g', alpha=0.15, linewidth=0)  # green box of reward zone
    ax.axhspan(0, 0.1, facecolor='k', alpha=0.15, linewidth=0)  # black box (4 is used for appearance)
    #ax.axhline(4, linewidth=1, ls='--', color='black')  # mark black box border
    #ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
    #ax.axhline(2, linewidth=3, color='black')  # bold line on the x axis

    ax.plot(np.arange(0,len(b_means)), b_means, 'o', color='0.3', label='Non reward zone score', linewidth=2, markersize=6,
            markeredgecolor='black')
    ax.errorbar(np.arange(0, len(b_means)), b_means, b_stds, fmt='o', color='0.3', capsize=1.5, markersize=2, elinewidth=1.5)
    ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7,
                   labelsize=15)  # tick parameters: pad is tick label relative to
    ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=15)
    ax.set_xlim(0, len(b_means))
    ax.set_ylim(0, 0.6)
    adjust_spines(ax, ['left', 'bottom'])  # remove right and top axis
    plt.locator_params(nbins=3, axis='y')  # set tick number on y axis
    plt.locator_params(nbins=10, axis='x')  # set tick number on x axis
    ax = plt.ylabel('First Stop location', fontsize=12, labelpad=18)
    ax = plt.xlabel('Training Blocks (100 Trials)', fontsize=12, labelpad=14)

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Non-Beaconed', fontsize=20, verticalalignment='bottom')
    ax.axhspan(0.4, 0.6, facecolor='g', alpha=0.15,
               linewidth=0)  # green box spanning the rewardzone - to mark reward zone
    ax.axhspan(0, 0.1, facecolor='k', alpha=0.15, linewidth=0)  # black box
    ax.axhline(4, linewidth=1, ls='--', color='black')  # mark black box border
    #ax.axvline(0, linewidth=3, color='black')  # bold line on the y axis
    #ax.axhline(2, linewidth=3, color='black')  # bold line on the x axis

    ax.plot(np.arange(0,len(nb_means)), nb_means, 'o', color='0.3', label='Non reward zone score', linewidth=2, markersize=6,
            markeredgecolor='black')
    ax.errorbar(np.arange(0,len(nb_means)), nb_means, nb_stds, fmt='o', color='0.3', capsize=1.5, markersize=2, elinewidth=1.5)
    ax.tick_params(axis='x', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=15)
    ax.tick_params(axis='y', pad=10, top='off', right='off', direction='out', width=2, length=7, labelsize=15)
    ax.set_xlim(0, len(nb_means))
    ax.set_ylim(0, 0.6)
    adjust_spines(ax, ['left', 'bottom'])  # remove right and top axis
    plt.locator_params(nbins=3, axis='y')  # set tick number on y axis
    plt.locator_params(nbins=10, axis='x')  # set tick number on x axis
    ax = plt.xlabel('Training Blocks (100 Trials)', fontsize=12, labelpad=14)

    plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.25, left=0.15, right=0.82, top=0.85)

    save_path = plot_path+title
    fig.savefig(save_path, dpi=200)
    plt.close()


def adjust_spines(ax,spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward',0)) # outward by 10 points
        #spine.set_smart_bounds(True)
        else:
            spine.set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])



def average_value_function(behaviour, trialtype_log, values, ax=None):
    ax.set(xlabel="Track Position", ylabel="Expected Reward")
    ax.set_xlim([-0.15, 1.15])  # track limits

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)
    values = np.array(values)

    beaconed_beh = behaviour[trialtype_log=="beaconed"]
    non_beaconed_beh = behaviour[trialtype_log=="non_beaconed"]

    beaconed_vals = values[trialtype_log=="beaconed"]
    non_beaconed_vals = values[trialtype_log=="non_beaconed"]

    all_beaconed_pos = []
    all_beaconed_vals = []

    counter = 0
    for trial in beaconed_beh:
        trial = np.array(trial)
        vals = np.array(beaconed_vals[counter])
        pos = np.array([i[0] for i in trial[:, 4]]) # vector of positions
        unique_pos, unique_ind = np.unique(pos, return_index=True)
        pos = pos[unique_ind]
        all_beaconed_pos.append(pos)
        all_beaconed_vals.append(vals[unique_ind])
        counter+=1

    max_l = max(len(l) for l in all_beaconed_pos)

    b_max_pos_arr = np.array([0])
    for trial_pos in all_beaconed_pos:
        if len(trial_pos)>len(b_max_pos_arr):
            b_max_pos_arr = trial_pos

    tmp = np.empty((len(all_beaconed_vals),max_l))
    tmp[:] = np.nan
    counter = 0
    for vals in all_beaconed_vals:
        tmp[counter][0:len(vals)] = vals[:,0]
        counter+=1
    # TODO: tmp array isn't perfectly alligned for examples where agent doesnt move immedicately

    b_mean = np.nanmean(tmp, axis=0)
    b_std = np.nanstd(tmp, axis=0)

    all_non_beaconed_pos = []
    all_non_beaconed_vals = []

    counter = 0
    for trial in non_beaconed_beh:
        trial = np.array(trial)
        vals = np.array(non_beaconed_vals[counter])
        pos = np.array([i[0] for i in trial[:, 4]])  # vector of positions
        unique_pos, unique_ind = np.unique(pos, return_index=True)
        pos = pos[unique_ind]
        all_non_beaconed_pos.append(pos)
        all_non_beaconed_vals.append(vals[unique_ind])
        counter += 1

    max_l = max(len(l) for l in all_non_beaconed_pos)

    nb_max_pos_arr = np.array([0])
    for trial_pos in all_non_beaconed_pos:
        if len(trial_pos) > len(nb_max_pos_arr):
            nb_max_pos_arr = trial_pos

    tmp = np.empty((len(all_non_beaconed_vals), max_l))
    tmp[:] = np.nan
    counter = 0
    for vals in all_non_beaconed_vals:
        tmp[counter][0:len(vals)] = vals[:, 0]
        counter += 1
    # TODO: tmp array isn't perfectly alligned for examples where agent doesnt move immedicately

    nb_mean = np.nanmean(tmp, axis=0)
    nb_std = np.nanstd(tmp, axis=0)

    print("now make plot")
    ax.axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
    ax.axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
    ax.axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

    style_vr_plot(ax, xbar=-20, ybar=-0.15)

    ax.plot(b_max_pos_arr[0:40], b_mean[0:40], color="k")
    ax.fill_between(b_max_pos_arr[0:40], b_mean[0:40]-b_std[0:40], b_mean[0:40]+b_std[0:40], facecolor="k", alpha= 0.3)
    ax.plot(nb_max_pos_arr[0:40], nb_mean[0:40], color="r")
    ax.fill_between(nb_max_pos_arr[0:40], nb_mean[0:40]-nb_std[0:40], nb_mean[0:40]+nb_std[0:40], facecolor="r", alpha=0.3)

    return ax

def plot_network_activation_rnn(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    last_trialtype = trialtype_log[-1]

    # search for last non beaconed and beaconed
    last_beaconed_log = None
    last_non_beaconed_log = None

    while (last_beaconed_log == None):
        for i in range(1, 11):
            if trialtype_log[-i] == "beaconed":
                last_beaconed_log = np.array(behaviour[-i])
                last_b_layers = np.array(layer_behaviour[-i])
                b_pos = [j[0] for j in last_beaconed_log[:, 4]]

    while (last_non_beaconed_log == None):
        for i in range(1, 11):
            if trialtype_log[-i] == "non_beaconed":
                last_non_beaconed_log = np.array(behaviour[-i])
                last_nb_layers = np.array(layer_behaviour[-i])
                nb_pos = [j[0] for j in last_non_beaconed_log[:, 4]]

    print(last_trialtype, "=last_trialtype")
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(16, 16, sharex=True, sharey=True, figsize=(20, 20))

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [-1.2, -1.2, 1.2, 1.2]

    count = 0
    for i in range(16):
        for j in range(16):
            activations = last_trial_layers[:, :, :, count]



            b_activations = last_b_layers[:, :, :, count]
            nb_activations = last_nb_layers[:, :, :, count]

            ax_l1[i, j].plot(b_pos, b_activations[:, 0], '-', color='Black')
            ax_l1[i, j].plot(nb_pos, nb_activations[:, 0], '-', color='Red')

            #ax_l1[i, j].plot(pos, activations[:, 0], '-', color='Black')

            if j==0:
                ax_l1[i, j].set(ylabel= "Unit activation")
            if i==15:
                ax_l1[i, j].set(xlabel = 'Track Position')

            #ax_l1[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l1[i, j].set_xlim([-0.15, 1.15])  # track limits
            ax_l1[i, j].axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
            ax_l1[i, j].axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
            ax_l1[i, j].axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

            ax_l1[i, j].fill(x, y, color="k", alpha=0.2)
            ax_l1[i, j].set_ylim([-1.1, 1.1])  # track limits

            count += 1
    fig_l1.tight_layout()
    fig_l1.savefig(save_path + title + "l1_pn", dpi=300)
    fig_l1.clf()

def plot_average_network_activation_rnn(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    behaviour = np.array(behaviour)
    trialtype_log = np.array(trialtype_log)
    layer_behaviour = np.array(layer_behaviour)

    beaconed_beh = behaviour[trialtype_log == "beaconed"]
    non_beaconed_beh = behaviour[trialtype_log == "non_beaconed"]

    beaconed_layer_beh = layer_behaviour[trialtype_log == "beaconed"]
    non_beaconed_layer_beh = layer_behaviour[trialtype_log == "non_beaconed"]

    bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
    b_binned = [[] for i in range(len(bins))]
    nb_binned = [[] for i in range(len(bins))]

    fig_l1, ax_l1 = plt.subplots(16, 16, sharex=True, sharey=True, figsize=(20, 20))

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [-1.2, -1.2, 1.2, 1.2]

    count = 0
    for i in range(16):
        for j in range(16):

            bins = list((np.round(np.arange(0, 1.225, 0.025), decimals=3)))
            b_binned = [[] for k in range(len(bins))]
            nb_binned = [[] for k in range(len(bins))]

            counter = 0
            for trial, trial_layer in zip(beaconed_beh, beaconed_layer_beh):
                trial = np.array(trial)
                trial_layer = np.array(trial_layer)
                trial_unit = trial_layer[:,:,:,count].flatten()

                v = np.array([k[1] for k in np.array(trial)[:, 4]])  # vector of velocities
                pos = np.round(np.array([k[0] for k in trial[:, 4]]), decimals=3)  # vector of

                for m in range(len(pos)):
                    pos_tmp = pos[m]
                    trial_unit_tmp = trial_unit[m]
                    idx = bins.index(pos_tmp)
                    b_binned[idx].append(trial_unit_tmp)

            b_means = []
            b_stds = []
            for k in range(len(b_binned)):
                if len(b_binned[k]) > 0:
                    b_means.append(np.mean(b_binned[k]))
                    b_stds.append(np.std(b_binned[k]))
                else:
                    b_means.append(np.nan)
                    b_stds.append(np.nan)

            for trial, trial_layer in zip(non_beaconed_beh, non_beaconed_layer_beh):
                trial = np.array(trial)
                trial_layer = np.array(trial_layer)
                trial_unit = trial_layer[:, :, :, count].flatten()

                v = np.array([k[1] for k in np.array(trial)[:, 4]])  # vector of velocities
                pos = np.round(np.array([k[0] for k in trial[:, 4]]), decimals=3)  # vector of

                for k in range(len(pos)):
                    pos_tmp = pos[k]
                    trial_unit_tmp = trial_unit[k]
                    idx = bins.index(pos_tmp)
                    nb_binned[idx].append(trial_unit_tmp)

            nb_means = []
            nb_stds = []
            for k in range(len(nb_binned)):
                if len(nb_binned[k]) > 0:
                    nb_means.append(np.mean(nb_binned[k]))
                    nb_stds.append(np.std(nb_binned[k]))
                else:
                    nb_means.append(np.nan)
                    nb_stds.append(np.nan)

            bins = np.array(bins[0:40])
            b_means = np.array(b_means[0:40])
            b_stds = np.array(b_stds[0:40])
            nb_means = np.array(nb_means[0:40])
            nb_stds = np.array(nb_stds[0:40])

            ax_l1[i, j].plot(bins, b_means, color="k")
            ax_l1[i, j].fill_between(bins, b_means - b_stds, b_means + b_stds, facecolor="k", alpha=0.3)
            ax_l1[i, j].plot(bins, nb_means, color="r")
            ax_l1[i, j].fill_between(bins, nb_means - nb_stds, nb_means + nb_stds, facecolor="r", alpha=0.3)

            if j == 0:
                ax_l1[i, j].set(ylabel="Unit activation")
            if i == 15:
                ax_l1[i, j].set(xlabel='Track Position')

            # ax_l1[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l1[i, j].set_xlim([-0.15, 1.15])  # track limits
            ax_l1[i, j].axvspan(0.4, 0.6, facecolor='DarkGreen', alpha=.25, linewidth=0)
            ax_l1[i, j].axvspan(-0.2, 0, facecolor='k', linewidth=0, alpha=.25)  # black box
            ax_l1[i, j].axvspan(1, 1.2, facecolor='k', linewidth=0, alpha=.25)  # black box

            ax_l1[i, j].fill(x, y, color="k", alpha=0.2)
            ax_l1[i, j].set_ylim([-1.1, 1.1])  # track limits

            count += 1
    fig_l1.tight_layout()
    fig_l1.savefig(save_path + title + "l1_pn", dpi=300)
    fig_l1.clf()

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    dir = os.path.dirname(__file__)
    plot_path = os.path.join(dir, 'figures', 'binary_action', '')

    ep_logs = np.load(
        '/home/harry/car/stable-baselines/stable_baselines/ppo2/experiments/grid_cell/figures/binary_action/id=0131_std=0_i=2ep_logs.npy',
        allow_pickle=True).tolist()
    value_log = np.load(
        '/home/harry/car/stable-baselines/stable_baselines/ppo2/experiments/grid_cell/figures/binary_action/id=0131_std=0_i=2value_log.npy',
        allow_pickle=True).tolist()
    trialtype_log = np.load(
        '/home/harry/car/stable-baselines/stable_baselines/ppo2/experiments/grid_cell/figures/binary_action/id=0131_std=0_i=2trialtype_log.npy',
        allow_pickle=True).tolist()
    layer_log = np.load(
        '/home/harry/car/stable-baselines/stable_baselines/ppo2/experiments/grid_cell/figures/binary_action/id=0131_std=0_i=2layer_log.npy',
        allow_pickle=True).tolist()
    title = "for_figure"

    first_stop_trialblocks(ep_logs, trialtype_log, plot_path, "for_figurefirststop")

    plot_summary_with_fn(ep_logs, value_log, trialtype_log, plot_path, "for_figure_")

    plot_summary_with_fn(ep_logs[0:100], value_log[0:100], trialtype_log[0:100], plot_path, "for_figure_first100")
    # plot_summary_with_fn(ep_logs[750:850], value_log[750:850], trialtype_log[750:850], plot_path, "for_figure_first750_850")
    # plot_summary_with_fn(ep_logs[2700:2800], value_log[2700:2800], trialtype_log[2700:2800], plot_path, "for_figure_last100")

    # plot_average_network_activation_rnn(layer_log[0:100], ep_logs[0:100], trialtype_log[0:100], plot_path, title + "_last_trial_layer_first100")
    # plot_average_network_activation_rnn(layer_log[2700:2800], ep_logs[2700:2800], trialtype_log[2700:2800], plot_path, title + "_last_trial_layer_last100")
    # plot_average_network_activation_rnn(layer_log[750:850], ep_logs[750:850], trialtype_log[750:850], plot_path, title + "_last_trial_layer_last750_850")

    # plot_activation_gradients(ep_logs[0:100], layer_log[0:100], trialtype_log[0:100], plot_path, "activation_gradients_first100")
    # plot_activation_gradients(ep_logs[2700:2800], layer_log[2700:2800], trialtype_log[2700:2800], plot_path, "activation_gradients_last100")
    # plot_activation_gradients(ep_logs[750:850], layer_log[750:850], trialtype_log[750:850], plot_path, "activation_gradients_last750-850")

    print("hello there")

if __name__ == '__main__':
    main()


'''























'''
this is the old plotting file

import numpy as np
import matplotlib.pyplot as plt
'''
import time
import gym

import tensorflow as tf
from gym.spaces import Discrete, Box

from stable_baselines import logger
from stable_baselines.a2c.utils import batch_to_seq, seq_to_batch, Scheduler, find_trainable_variables, EpisodeStats, \
    get_by_index, check_shape, avg_norm, gradient_add, q_explained_variance, total_episode_reward_logger
from stable_baselines.acer.buffer import Buffer
from stable_baselines.common import ActorCriticRLModel, tf_util, SetVerbosity, TensorboardWriter
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.policies import LstmPolicy, ActorCriticPolicy
'''

def plot_network_activation_dqn(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    fig_l2, ax_l2 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone

    count = 0
    for i in range(8):
        for j in range(8):
            activations = last_trial_layers[:, :, :, count]

            ax_l1[i, j].scatter(pos, activations[:, 0])
            ax_l2[i, j].scatter(pos, activations[:, 1])
            count += 1

            ax_l1[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l2[i, j].set(xlabel='Track Position', ylabel= "Unit activation")

            ax_l1[i, j].set_xlim([-0.6, 1])  # track limits
            ax_l2[i, j].set_xlim([-0.6, 1])  # track limits

            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l1[i, j].fill(x, y, color="k", alpha=0.2)
            y = [-1.2, -1.2, 1.2, 1.2]
            ax_l2[i, j].fill(x, y, color="k", alpha=0.2)

            ax_l1[i, j].set_ylim([-1.1, 1.1])  # track limits
            ax_l2[i, j].set_ylim([-1.1, 1.1])  # track limits

    fig_l1.tight_layout()
    fig_l2.tight_layout()

    fig_l1.savefig(save_path + title + "l1_qnet")
    fig_l2.savefig(save_path + title + "l2_qnet")

    fig_l1.clf()
    fig_l2.clf()

def plot_network_activation(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")
    fig_l2, ax_l2 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")
    fig_l3, ax_l3 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")
    fig_l4, ax_l4 = plt.subplots(8, 8, sharex=True, sharey=True, figsize=(20, 20))
    plt.xlabel("common X")
    plt.ylabel("common Y")

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

            if j==0:
                ax_l1[i, j].set(ylabel= "Unit activation")
                ax_l2[i, j].set(ylabel= "Unit activation")
                ax_l3[i, j].set(ylabel= "Unit activation")
                ax_l4[i, j].set(ylabel= "Unit activation")
            if i==7:
                ax_l1[i, j].set(xlabel = 'Track Position')
                ax_l2[i, j].set(xlabel = 'Track Position')
                ax_l3[i, j].set(xlabel = 'Track Position')
                ax_l4[i, j].set(xlabel = 'Track Position')

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


def plot_network_activation_rnn(layer_behaviour, behaviour, trialtype_log, save_path, title):
    # TODO plot activations for last example for beaconed, probe and non beaconed
    # currently hardcoded for 64 units and 4 layers (2 per network)

    last_trial_log = np.array(behaviour[-1])
    pos = [i[0] for i in last_trial_log[:, 4]] # vector of positions
    last_trial_layers = np.array(layer_behaviour[-1])

    fig_l1, ax_l1 = plt.subplots(16, 16, sharex=True, sharey=True, figsize=(20, 20))

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [-1.2, -1.2, 1.2, 1.2]

    count = 0
    for i in range(16):
        for j in range(16):
            activations = last_trial_layers[:, :, :, count]

            ax_l1[i, j].scatter(pos, activations[:, 0])

            if j==0:
                ax_l1[i, j].set(ylabel= "Unit activation")
            if i==15:
                ax_l1[i, j].set(xlabel = 'Track Position')

            #ax_l1[i, j].set(xlabel='Track Position', ylabel= "Unit activation")
            ax_l1[i, j].set_xlim([0, 1])  # track limits
            ax_l1[i, j].fill(x, y, color="k", alpha=0.2)
            ax_l1[i, j].set_ylim([-1.1, 1.1])  # track limits

            count += 1
    fig_l1.tight_layout()
    fig_l1.savefig(save_path + title + "l1_pn")
    fig_l1.clf()

def plot_rasta_test(ep_log, save_path, title):
    # for looking at behaviour of agent within learning env not evaluation env
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    info_ = []
    dones_ = []

    for i in ep_log:
        info_.append(i[0])
        dones_.append(i[1])

    info = [val for sublist in info_ for val in sublist]
    dones = [val for sublist in dones_ for val in sublist]

    ep_log = []

    counter = 0
    ep = []
    for i in dones:
        ep.append(info[counter])
        counter+=1

        if i == True:
            ep_log.append(ep)
            ep = []

    raster_test(ep_log, ax1)
    f.tight_layout()
    f.savefig(save_path + title)
    f.clf()

def raster_test(behaviour, ax=None):

    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    no_trials = len(behaviour)
    all_trial_stopping = []    # initialise empty list

    # changes structure so behaviour is organised by trial
    for trial in behaviour:
        trial = np.array(trial)

        v = trial[:, 1] # vector of states per time step in trial (velocity)
        pos = trial[:, 0] # vector of positions

        idx = np.array(np.round(np.array(v), decimals=3) == 0.0)  # vector of boolean per time step showing v = 0

        all_trial_stopping.append(pos[idx])  # appendable list of positions for which v = 0
        #print(pos[idx], "=pos[idx]")

    #plt.title(title)
    ax.set(xlabel='Track Position', ylabel='Trial')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [1, 1, no_trials, no_trials]
    ax.fill(x, y, color="k", alpha=0.2)

    ax.set_ylim([1, no_trials])
    ax.set_xlim([-0.6, 1])  # track limits

    # Draw a spike raster plot
    return ax.eventplot(all_trial_stopping, linewidths=5)



def plot_summary_with_fn(behaviour, values, trialtype_log, save_path, title):

    # TODO add plots for actions and values of last trial

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    raster(behaviour, trialtype_log, ax1)
    #accum_reward(behaviour, ax2)
    speed_of_last(behaviour, trialtype_log, ax3)
    average_ep_reward(behaviour, ax2)
    #actions_of_last(behaviour, actions, ax5)
    value_fn_of_last(behaviour, values, ax4)
    f.tight_layout()
    #plt.show()

    f.savefig(save_path + title)

    f.clf()

def plot_summary(behaviour, trialtype_log, save_path, title):

    # TODO add plots for actions and values of last trial

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    raster(behaviour, trialtype_log, ax1)
    #accum_reward(behaviour, ax2)
    speed_of_last(behaviour, trialtype_log, ax3)
    average_ep_reward(behaviour, ax2)
    #actions_of_last(behaviour, actions, ax5)
    #value_fn_of_last(behaviour, values, ax4)
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

def last_trials_of_trialtype(behaviour, trialtype_log, trialtype_wanted, last_n=5):
    trialtype_log = np.array(trialtype_log)
    idx = np.where(trialtype_log == trialtype_wanted)[-1][-last_n:]  # picks last n elements of array that meets trialtype condition, returns index

    tmp = []
    for i in idx:
        tmp.append(behaviour[i])
    behaviour = tmp

    return behaviour

def speed_of_last(behaviour, trialtype_log, ax=None):
    # [episode[timestep[action, rew, obs, float(done), state]]]   obs = [position velocity]
    # plt.title(title)
    ax.set(xlabel='Track Position', ylabel='Speed')

    x = [0.4, 0.6, 0.6, 0.4]  # setting fill area for reward zone
    y = [0, 0, 0.25, 0.25]
    ax.fill(x, y, color="k", alpha=0.2)
    ax.set_xlim([-0.6, 1])  # track limits
    ax.set_ylim([0, 0.25])

    last_trials_b = last_trials_of_trialtype(behaviour, trialtype_log, "beaconed")
    last_trials_nb = last_trials_of_trialtype(behaviour, trialtype_log, "non_beaconed")
    last_trials_p = last_trials_of_trialtype(behaviour, trialtype_log, "probe")

    if len(last_trials_b)>0:
        for trial in last_trials_b:
            v = [i[1] for i in np.array(trial)[:, 4]]  # vector of velocities
            pos = [i[0] for i in np.array(trial)[:, 4]] # vector of positions
            ax.plot(pos, v, color='k')

    if len(last_trials_nb) > 0:
        for trial in last_trials_nb:
            v = [i[1] for i in np.array(trial)[:, 4]]  # vector of velocities
            pos = [i[0] for i in np.array(trial)[:, 4]]  # vector of positions
            ax.plot(pos, v, color='b')

    if len(last_trials_p) > 0:
        for trial in last_trials_p:
            v = [i[1] for i in np.array(trial)[:, 4]]  # vector of velocities
            pos = [i[0] for i in np.array(trial)[:, 4]]  # vector of positions
            ax.plot(pos, v, color='r')

def raster(behaviour, trialtype_log, ax=None):

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


    colors = []
    for trialtype in trialtype_log:
        if trialtype == "beaconed":
            colors.append("k")
        elif trialtype == "non_beaconed":
            colors.append("b")
        elif trialtype == "probe":
            colors.append("r")
        else:
            print("trial type does not match string beaconed probe or non_beaconed")

    # Draw a spike raster plot
    return ax.eventplot(all_trial_stopping, linewidths=5, color=colors)




'''


