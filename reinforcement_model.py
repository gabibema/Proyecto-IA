import sys
sys.path.append("../Proyecto-IA")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from snake_game import BLOCK_SIZE, SnakeGame, Point, Direction

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

        prediction = self.model(state).cuda()
        target = prediction.clone().cuda()

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = []
        self.model = Linear_QNet(11,256,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)


    def get_state(self,game):
        head = game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_u and game.is_collision(point_u))or 
            (dir_d and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_r)),

            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),


            (dir_u and game.is_collision(point_r))or 
            (dir_d and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),


            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x, 
            game.food.x > game.head.x,
            game.food.y < game.head.y, 
            game.food.y > game.head.y
        ]
        return np.array(state,dtype=int)