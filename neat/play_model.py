import sys
sys.path.append("../Proyecto-IA")
import neat
import pygame
import pickle
from snake_game import SnakeGame
from snake_gameAI import get_state

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     './neat/config/neat-config')
genome = pickle.load(open('./neat/snake-best-network.pt', 'rb'))
network = neat.nn.FeedForwardNetwork.create(genome, config)

game = SnakeGame()
while not game.game_over:
    # Obtengo el estado actual del juego
    state = get_state(game)

    # Dejo que la red decida el proximo output
    output = network.activate(state)
        
    # Obtengo la accion de la red
    action = output.index(max(output))
    
    # ejecuto accion
    move = [0,0,0]
    move[action] = 1
    game.play_step(move)
