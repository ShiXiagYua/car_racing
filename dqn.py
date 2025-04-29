import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
import cv2
import time
import matplotlib.pyplot as plt
def process_state(state):
    # state=state//50
    # state=state*50
    gray_img = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    crop_img = gray_img[:84, 6:90]
    crop_img=crop_img[::2,::2]
    crop_img=crop_img[None,...]
    return crop_img

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer)>self.capacity:
            self.buffer.pop(0)
    def sample(self, batch_size):
        state,action,reward,next_state,done=zip(*random.sample(self.buffer,batch_size))
        return np.array(state),np.array(action),np.array(reward),np.array(next_state),np.array(done)
class DQN(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(DQN, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2),#42->21
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=3),#21->7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),#7->3
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=1),#3->1
            nn.ReLU(),
            nn.Flatten(),
        )
        self.net=nn.Sequential(
            self.encoder,
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.target_net=copy.deepcopy(self.net)
        self.loss_func=nn.MSELoss()
        self.gamma=0.98
        self.update_gap=50
        self.steps=0
        self.optimizer=optim.Adam(self.net.parameters(), lr=0.002)
        self.device=device
    def take_action(self,state):
        state=torch.tensor(state,dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values=self.net(state)
        action=torch.argmax(q_values.squeeze(0)).item()
        return action
    def update(self,states,actions,rewards,next_states,dones):
        states=torch.tensor(states,dtype=torch.float).to(self.device)
        next_states=torch.tensor(next_states,dtype=torch.float).to(self.device)
        actions=torch.tensor(actions,dtype=torch.long).unsqueeze(1).to(self.device)
        rewards=torch.tensor(rewards,dtype=torch.float).unsqueeze(1).to(self.device)
        dones=torch.tensor(dones,dtype=torch.float).unsqueeze(1).to(self.device)
        td_targets=rewards+self.gamma*self.target_net(next_states).gather(1,torch.argmax(self.net(next_states),dim=1,keepdim=True))*(1-dones)
        loss=self.loss_func(self.net(states).gather(1,actions),td_targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.steps+=1
        if self.steps%self.update_gap==0:
            self.target_net.load_state_dict(self.net.state_dict())
        return loss.item()
env=gym.make('CarRacing-v2',render_mode="rgb_array",domain_randomize=False,continuous=False)
device=torch.device("cuda:2")
action_stick=5
input_size=42*42
output_size=env.action_space.n
buffer_size=1000000
batch_size=32
min_replay_size=500
explore_rate=1
explore_decay=0.95
min_explore_rate=0.01
agent=DQN(input_size,output_size,device).to(device)
reward_list=[]
buffer=ReplayBuffer(buffer_size)
for i_episode in range(1000):
    state,info=env.reset()
    state=process_state(state)
    episode_reward=0
    episode_steps=0
    terminate=False
    while not terminate:
        if random.random()<explore_rate:
            action=env.action_space.sample()
        else:
            action=agent.take_action(state)
        rewards=0
        for i in range(action_stick):
            next_state,reward,done,trunct,info=env.step(action)
            terminate = done or trunct
            rewards+=reward
            if terminate:
                break
        next_state=process_state(next_state)
        buffer.push(state,action,rewards,next_state,done)
        episode_reward+=rewards
        episode_steps+=action_stick
        state=next_state

        if len(buffer.buffer)>min_replay_size:
            states,actions,rewards,next_states,dones=buffer.sample(batch_size)
            loss=agent.update(states,actions,rewards,next_states,dones)
            # print('Loss: {}'.format(loss))
    explore_rate*=explore_decay
    reward_list.append(episode_reward)
    if explore_rate<min_explore_rate:
        explore_rate=min_explore_rate
    print('Episode: {}, Reward: {}, Epi_Steps: {},Explore Rate: {}'.format(i_episode,episode_reward,episode_steps,explore_rate))
    plt.clf()
    plt.plot(reward_list)
    plt.savefig("plot2.png")


