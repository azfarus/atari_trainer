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
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 1000000
TAU = 0.005
LR = 5e-5
K=4
EPS=0
reward_scale=1



device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

env=gym.make('PongNoFrameskip-v4',obs_type='grayscale',render_mode='rgb_array')

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, input_channels=4, output_size=6):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(32 * 9 * 9, 256)
        self.output_layer = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = (x ) /255.0
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc(x))
        x = self.output_layer(x)
        return x



steps_done=0
def select_action(state):
    global steps_done , EPS
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    EPS=eps_threshold
    steps_done += 1
    if sample > eps_threshold  and state != None:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations=[]    
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool,)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch*reward_scale

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # print(loss)
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


policy_net = DQN(input_channels=K).to(device)
target_net = DQN(input_channels=K).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(150000)

def printdeets(epoch):

    global steps_done , EPS
    mins= (time.time() - prog_start)/60.0
    print('--------------------------------')
    print(f'Mins Elapsed: {mins}')
    print(f'EPS: {EPS}')
    print(f'Steps Done: {steps_done}')
    print(f'Buffer Len: {len(memory)}')
    print(f'Epochs: {(epoch)}')
    print(f'Step/Min: {steps_done/mins}')
    print(f'Step/Min: {steps_done/epoch}')


tot_reward=0.0
render=False
for epoch in count():

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
        exit=False
        for i in range(K):
            new_frame,reward,terminated,truncated,_=env.step(action.item())
            framebuffer.append(cv2.resize(new_frame , (84,84)))
            skip_reward+=reward
            done=terminated or truncated
            if render: 
                cv2.imshow('frame',framebuffer[K-1])
                cv2.waitKey(5)
            
            

        tot_reward+=skip_reward

        if terminated:
            new_state=None
        else:
            new_state=torch.tensor(np.array(framebuffer),dtype=torch.float32,device=device).unsqueeze(0)

        skip_reward=torch.tensor([skip_reward],dtype=torch.float32,device=device)

        
        memory.push(cur_state,action,new_state,skip_reward)
        cur_state=new_state
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        if keyboard.is_pressed('end') or done:
            break
        
        if keyboard.is_pressed('delete'):
            render = not render
            cv2.waitKey(200)

        if steps_done % 1000 == 0:
            printdeets(epoch+1)
            plot_durations()
        
    if keyboard.is_pressed('end'):
        break        
    episode_durations.append(tot_reward)
    tot_reward=0
    



torch.save(policy_net,'pong.pt')           

        
        

        
        
        


