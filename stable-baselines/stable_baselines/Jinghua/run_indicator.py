"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DQNPrioritizedReplay
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os

env = gym.make('Jinghua_Original_MountainCar-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 5000
reward_history = []

sess = tf.Session()

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=2, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())

def train(RL):
    total_steps = 0
    min_step = 200
    min_epi = 0
    steps = []
    episodes = []
    appro_observation=[]

    for i_episode in range(1000):
        observation = env.reset()
        real_position = observation[0]
        episode_step = 0
        exp = 0
        loc_list = list()
        vel_list = list()
        episode_reward = 0

        while True:
            action = RL.choose_action(observation)
            appro_position = observation[0]
            observation_, real_position, reward, done, info = env.step(real_position,action,exp)

            if i_episode%5 == 0 and i_episode>500:
               observation_[0] = 0

            episode_step += 1
            loc_list.append(real_position)
            vel_list.append(observation_[1])

            if done == 1: 
               reward = 10
               exp = 1

            episode_reward += reward
   
            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()


            if done == 2:
              if episode_step <= min_step:
                min_step = episode_step
                min_epi = i_episode
                print('min_step:',min_step)

              print('episode ', i_episode, ' finished')
              steps.append(total_steps)
              episodes.append(i_episode)
              reward_history.append(episode_reward)

              if i_episode >= 990:


               st = list(range(0,episode_step))
               plt.plot(st,loc_list, alpha = 0.9, color = 'b')
               plt.savefig(format(i_episode) + 'location.png')
               plt.cla()

               st = list(range(0,episode_step))
               plt.plot(st,vel_list, alpha = 0.9, color = 'r')
               plt.savefig(format(i_episode) + 'velocity.png')
               plt.cla()


               layer1 = np.zeros(( 20,len(loc_list) ))
               neuron = np.zeros(( 2,len(loc_list) ))
               loc_list = np.array(loc_list)
               print(loc_list)
               vel_list = np.array(vel_list)
               saver = tf.train.Saver()

               #saver.save(sess, '/home/harry/PycharmProjects/car/stable-baselines/stable_baselines/Jinghua/model.ckpt')
               saver.save(sess, '/home/harry/PycharmProjects/car_rl/car/stable-baselines/stable_baselines/Jinghua/model.ckpt')

               #saver.save(sess, "/Home/yang/DQ/model.ckpt")
               #reader = tf.train.NewCheckpointReader("/HSome/yang/DQ/model.ckpt")
               #reader = tf.train.NewCheckpointReader('/home/harry/PycharmProjects/car/stable-baselines/stable_baselines/Jinghua/model.ckpt')
               reader = tf.train.NewCheckpointReader('/home/harry/PycharmProjects/car_rl/car/stable-baselines/stable_baselines/Jinghua/model.ckpt')

               w1 = reader.get_tensor("DQN_with_prioritized_replay/eval_net/l1/w1")
               b1 = reader.get_tensor("DQN_with_prioritized_replay/eval_net/l1/b1")
               w2 = reader.get_tensor("DQN_with_prioritized_replay/eval_net/l2/w2")
               b2 = reader.get_tensor("DQN_with_prioritized_replay/eval_net/l2/b2")

               for neu_num in range(20): 
                 for timestep in range(len(loc_list)):
                   layer1[neu_num][timestep] = loc_list[timestep] * w1[0][neu_num] + vel_list[timestep]*w1[1][neu_num] + b1[0][neu_num]
                   if layer1[neu_num][timestep] < 0:
                      layer1[neu_num][timestep] = 0

               for neu_num in range(2): 
                 for timestep in range(len(loc_list)):
                   for weight_num in range(20):
                     neuron[neu_num][timestep] += layer1[weight_num][timestep] * w2[weight_num][neu_num]
                   neuron[neu_num][timestep] += b2[0][neu_num]
                   if neuron[neu_num][timestep] < 0:
                      neuron[neu_num][timestep] = 0

               for neu_num in range(20):
                 timesteps = range(len(loc_list))
                 plt.plot(loc_list,layer1[neu_num], alpha = 0.9, color = 'b')
                 plt.xlabel('location')
                 plt.ylabel('Q-value '+ format(neu_num) +' output')
                 plt.savefig(format(i_episode) + 'episode' + format(neu_num) + 'neuron.png')
                 plt.cla()
              break

            if episode_step >= 300:
               print('episode ', i_episode, 'unfinished')
               reward_history.append(episode_reward)
               break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))

his_prio = train(RL_prio)

#print accumulative reward
plt.clf()
rh = pd.Series(reward_history)
rhma = rh.rolling(10).mean()
plt.plot(rh,alpha=0.8)
plt.plot(rhma)
plt.xlabel('Episode')
plt.ylabel('reward')
plt.title('accumulative reward')
plt.savefig('accumulative reward.png')
plt.show()

# compare based on first success
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
#plt.show()
