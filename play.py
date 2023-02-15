import pygame
from snake_game import SnakeGame, Direction

font = pygame.font.Font(None,25) 
game = SnakeGame()

while not game.game_over:
    action = Direction.UP
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = Direction.UP
            if event.key == pygame.K_DOWN:
                action = Direction.DOWN
            if event.key == pygame.K_LEFT:
                action = Direction.LEFT
            if event.key == pygame.K_RIGHT:
                action = Direction.RIGHT
 
    game.play_step(action)
