import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from tensorboardX import SummaryWriter 

import matplotlib.pyplot as plt

import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

env_id = 'FrozenLakeNotSlippery-v0'
env = gym.make(env_id)

"""
use gym (openAI)
https://blog.techbridge.cc/2017/11/04/openai-gym-intro-and-q-learning/

"""
print(env.observation_space.n)
print(env.action_space.n)

epsilon_start = 1.
epsilon_final = 0.01
epsilon_decay = 3000.

def epsilon_by_frame(frame_idx):
    """
    your design
    """
    epsilon = math.exp(-1/epsilon_decay*frame_idx) 
    if epsilon < epsilon_final: epsilon = epsilon_final
    # epsilon = ??
    return epsilon

def act(Q, state, epsilon):
    print(epsilon)
    if random.random() > epsilon:
        # choose greedy
        action = np.argmax(Q[state])
        if action == np.argmin(Q[state]):
            action = env.action_space.sample()
    else:
        # choose random
        action = env.action_space.sample() 
    return action

Q = np.zeros((16,4))
losses         = []
all_rewards    = []
frames = []
episode_reward = 0
num_frames = 10000
gamma = 0.8
rate = 0.9
count = 0
state = env.reset()
for frame_idx in range(1, num_frames + 1):
    
    if(frame_idx == 1): 
        epsilon = epsilon_by_frame(frame_idx)
        action = act(Q, state, epsilon)

    # interact with environment
    env.render()
    next_state, reward, done, info = env.step(action)
    epsilon = epsilon_by_frame(frame_idx + 1)
    next_action = act(Q, next_state, epsilon)
    
    # update Q table
    # Q[??][??] = ??
    Q[state][action] += rate * (reward + gamma * Q[next_state][next_action] - Q[state][action])
    
    # go to next state
    state = next_state
    action = next_action
    episode_reward += reward
    
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        print('-----------done')
        if (reward == 1): count += 1

env.close()
print (count)
print(Q)
