import pygame
from snake_game import SnakeGame, Direction

font = pygame.font.Font(None,25) 
game = SnakeGame()

while not game.game_over:
    game.play_step()
