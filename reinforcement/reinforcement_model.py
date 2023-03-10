import sys
sys.path.append("../Proyecto-IA")

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from snake_game import BLOCK_SIZE, SnakeGame, Point, Direction

MAX_EPSILON = 80
MAX_MEMORY = 10000
LR = 0.001

class Linear_QNet(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size).cuda()
        self.linear2 = nn.Linear(hidden_size,output_size).cuda()
        
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='reinforcement/model.pt'):
        torch.save(self.state_dict(),file_name)
    

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimer = optim.Adam(model.parameters(),lr = self.lr)    
        self.criterion = nn.MSELoss()

    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float).cuda()
        next_state = torch.tensor(next_state,dtype=torch.float).cuda()
        action = torch.tensor(action,dtype=torch.long).cuda()
        reward = torch.tensor(reward,dtype=torch.float).cuda()

        if(len(state.shape) == 1): 
            state = torch.unsqueeze(state,0).cuda()
            next_state = torch.unsqueeze(next_state,0).cuda()
            action = torch.unsqueeze(action,0).cuda()
            reward = torch.unsqueeze(reward,0).cuda()
            done = (done, )

        prediction = self.model(state).cuda()
        target = prediction.clone().cuda()

        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i])).cuda()
            target[i][torch.argmax(action).item()] = Q_new 

        self.optimer.zero_grad()
        loss = self.criterion(target,prediction)
        loss.backward()

        self.optimer.step()



class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = []
        self.model = Linear_QNet(11,256,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) 

    def train_long_memory(self):
        if (len(self.memory) > MAX_MEMORY):
            mini_sample = random.sample(self.memory,MAX_MEMORY)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)
    
    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)


    def get_action(self,state):
        self.epsilon = MAX_EPSILON - self.n_game
        final_move = [0,0,0]

        if(random.randint(0,200)<self.epsilon):
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state = torch.tensor(state,dtype=torch.float).cuda()
            prediction = self.model(state).cuda()
            move = torch.argmax(prediction).item()
            final_move[move]=1
        return final_move