import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import cv2
import numpy as np
import keyboard
import time

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


prog_start=time.time()

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0
EPS_END = 0
EPS_DECAY = 1000000
TAU = 0.005
LR = 1e-4
K=4


device = torch.device(  "cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env=gym.make('PongNoFrameskip-v4',obs_type='grayscale',render_mode='human')






    
class DQN(nn.Module):

    def __init__(self, input_channels=K, output_size=6):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(32 * 9 * 9, 256)
        self.output_layer = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x=x/255.0
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        
        cv2.waitKey(10)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc(x))
        x = self.output_layer(x)
        return x

steps_done=0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold  and state != None:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            print(policy_net(state))
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)





policy_net = torch.load('pong.pt')


epochs=10000
for epoch in range(epochs):

    framebuffer=deque([],K)
    
    frame,info=env.reset()
    frame=cv2.resize(frame , (84,84))
    for i in range(K): framebuffer.append(frame)
    cur_state=torch.tensor(np.array(framebuffer),dtype=torch.float32,device=device).unsqueeze(0)
    new_state=None
    
    for t in count():
        
        action=select_action(cur_state)
        
        skip_reward=0.0
        done = False
        
        for i in range(K):
            new_frame,reward,terminated,truncated,_=env.step(action.item())
            framebuffer.append(cv2.resize(new_frame , (84,84)))
            
            done=terminated or truncated
            if done: break

        
        if terminated:
            new_state=None
        else:
            new_state=torch.tensor(np.array(framebuffer),dtype=torch.float32,device=device).unsqueeze(0)

        
        cur_state=new_state

        
        if keyboard.is_pressed('end'):
            break
        if done:
            break
            

    if keyboard.is_pressed('end'):
        break        


        

        
        

        
        
        


