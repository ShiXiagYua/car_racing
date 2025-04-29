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
    crop_img=np.array(crop_img,dtype=float)/255
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
class Q_Net(nn.Module):
    def __init__(self,action_bounds):
        super(Q_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2),  # 42->21
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=3),  # 21->7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),  # 7->3
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=1),  # 3->1
            nn.ReLU(),
            nn.Flatten(),
        )
        self.action_proj=nn.Linear(len(action_bounds),256)
        self.net = nn.Sequential(
            nn.Linear(256*2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, action):
        state=self.encoder(state)
        action=self.action_proj(action)
        state_action=torch.cat([state,action],dim=-1)
        return self.net(state_action)
class Policy_Net(nn.Module):
    def __init__(self,action_bounds):
        super(Policy_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2),  # 42->21
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=3),  # 21->7
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),  # 7->3
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=1),  # 3->1
            nn.ReLU(),
            nn.Flatten(),
        )
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(action_bounds))
        )
        action_bounds=torch.tensor(np.array(action_bounds),dtype=torch.float)
        self.action_scale=(action_bounds[:,1]-action_bounds[:,0])
        self.action_bias=action_bounds[:,0]
    def forward(self, state):
        state=self.encoder(state)
        action=(torch.tanh(self.net(state))+1.0)*0.5
        # print(action.shape)
        # print(self.action_scale.shape)
        # print(self.action_bias.shape)
        action=action*self.action_scale.to(action.device)+self.action_bias.to(action.device)
        return action

class DDPG(nn.Module):
    def __init__(self,action_bounds ,device):
        super(DDPG, self).__init__()
        self.net=Q_Net(action_bounds)
        self.target_net=copy.deepcopy(self.net)
        self.policy=Policy_Net(action_bounds)
        self.target_policy=copy.deepcopy(self.policy)
        self.action_bounds=torch.tensor(np.array(action_bounds),dtype=torch.float)
        self.loss_func=nn.MSELoss()
        self.explore_rate=0.4
        self.explore_decay=0.99
        self.min_explore_rate=0.01
        self.gamma=0.98
        self.update_gap=50
        self.tau=1
        self.steps=0
        self.optimizer=optim.Adam(self.net.parameters(), lr=0.0005)
        self.policy_optimizer=optim.Adam(self.policy.parameters(), lr=0.00005)
        self.device=device
    def decay_explore_rate(self):
        self.explore_rate*=self.explore_decay
        if self.explore_rate<self.min_explore_rate:
            self.explore_rate=self.min_explore_rate
    def take_action(self,state):
        state=torch.tensor(state,dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action=self.policy(state)
        noise=torch.randn(action.shape).to(self.device)*self.explore_rate
        action=action+noise
        action=torch.clamp(action,self.action_bounds[:,0].to(self.device),self.action_bounds[:,1].to(self.device))
        return np.array(action.detach().cpu()[0])
    def update(self,states,actions,rewards,next_states,dones):
        states=torch.tensor(states,dtype=torch.float).to(self.device)
        next_states=torch.tensor(next_states,dtype=torch.float).to(self.device)
        actions=torch.tensor(actions,dtype=torch.float).to(self.device)
        rewards=torch.tensor(rewards,dtype=torch.float).unsqueeze(1).to(self.device)
        dones=torch.tensor(dones,dtype=torch.float).unsqueeze(1).to(self.device)
        td_targets=rewards+self.gamma*self.target_net(next_states,self.target_policy(next_states))*(1-dones)
        loss=self.loss_func(self.net(states,actions),td_targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        policy_loss=-self.net(states,self.policy(states)).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.steps+=1
        if self.steps%self.update_gap==0:
            for target_param, param in zip(self.target_net.parameters(), self.net.parameters()):
                target_param.data.copy_(target_param.data*(1-self.tau)+param.data*self.tau)
            for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_param.data.copy_(target_param.data*(1-self.tau)+param.data*self.tau)
        return loss.item(),policy_loss.item()
env=gym.make('CarRacing-v2',render_mode="rgb_array",domain_randomize=False,continuous=True)
bounds=np.array(list(zip(env.action_space.low,env.action_space.high)))
device=torch.device("cuda:0")
action_stick=5
updata_skip=1
buffer_size=10000000
batch_size=256
min_replay_size=1000
agent=DDPG(bounds,device).to(device)
reward_list=[]
buffer=ReplayBuffer(buffer_size)
for i_episode in range(1000):
    state,info=env.reset()
    state=process_state(state)
    episode_reward=0
    episode_steps=0
    terminate=False
    updata_times=1
    mean_loss,mean_policy_loss=0.0,0.0
    while not terminate:
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

        if len(buffer.buffer)>min_replay_size and (episode_steps//action_stick)%updata_skip==0:
            states,actions,rewards,next_states,dones=buffer.sample(batch_size)
            loss,policy_loss=agent.update(states,actions,rewards,next_states,dones)
            mean_loss+=loss
            mean_policy_loss+=policy_loss
            updata_times+=1
    # print('Loss: {}, Policy Loss {}'.format(mean_loss/updata_times,mean_policy_loss/updata_times))
    agent.decay_explore_rate()
    reward_list.append(episode_reward)
    print('Episode: {}, Reward: {}, Epi_Steps: {},Explore Rate: {}'.format(i_episode,episode_reward,episode_steps,agent.explore_rate))
    plt.clf()
    plt.plot(reward_list)
    plt.savefig("plot2.png")


