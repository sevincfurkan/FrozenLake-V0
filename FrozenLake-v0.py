# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:55:31 2019

@author: sfurk
"""
#q-table
#Hyperparameter
#Plotting Metrix
#Episode
#Initialize enviroment
#Explot vs Explore
# action process and take reward/ observation
# Q learning function
# Q table update 
# update state
# find wrong dropouts

import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env=gym.make("FrozenLake-v0")

q_table=np.zeros([env.observation_space.n,env.action_space.n])

alpha=0.6
gamma=0.9
epsilon=0.1

epsido=75000
reward_list=[]
for i in range(1,epsido):
    state=env.reset()
    reward_count=0
    while True:
        
        if random.uniform(0,1)<epsilon:
            action=env.action_space.sample()
        else:
            action=np.argmax(q_table[state])
        
        next_state,reward,done,info=env.step(action)
        
        old_value=q_table[state,action]
        
        next_max=np.max(q_table[next_state])
        
        new_value=(1-alpha)*old_value+alpha*(reward+gamma*next_max)
        
        q_table[state,action]=new_value
        state=next_state
        
        reward_count +=reward
        
        if done:
            break
    
    
    if i%10 == 0:
        reward_list.append(reward_count)
        print("Episode: {}, reward {}".format(i,reward_count))

plt.plot(reward_list)


















