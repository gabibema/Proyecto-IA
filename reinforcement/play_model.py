import torch
import reinforcement_model
import pygame
from snake_game import SnakeGame, Direction


model = reinforcement_model.Linear_QNet(11,256,3)
model.load_state_dict(torch.load("reinforcement\modelTT.pt"))

font = pygame.font.Font(None,25) 
game = SnakeGame()
while not game.game_over:
    state = game.get_state()
    state = torch.tensor(state,dtype=torch.float).cuda()
    move = model(state)
    move = torch.argmax(move).item()
    final_move = [0,0,0]
    final_move[move]=1 
    game.play_step(final_move)