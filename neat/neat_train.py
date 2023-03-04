import os
import sys
sys.path.append("../Proyecto-IA")

import neat
import pickle
import snake_gameAI as snake_gameAI

def evaluate_genomes(genomes, config):
    print("evaluating genomes")
    for genome_id, genome in genomes:
        # Create a neural network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Play the game with the neural network
        fitness = snake_gameAI.play_game(net)

        genome.fitness = fitness 

# Set up the NEAT algorithm configuration
config_path = "./neat/config/neat-config.txt"
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

# Create a population and run the NEAT algorithm to evolve the population
pop = neat.population.Population(config)

# Restore the population from the last checkpoint
#checkpoint = neat.Checkpointer.restore_checkpoint('./neat/config/NeatCheckpoints/snake-checkpoint-23')
#pop = checkpoint

pop.add_reporter(neat.StdOutReporter(True))
stats = neat.statistics.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.Checkpointer(5, filename_prefix="./neat/config/NeatCheckpoints/snake-checkpoint-"))
winner = pop.run(evaluate_genomes)

# Save the best neural network to a file|
with open("snake-best-network", "wb") as f:
    pickle.dump(winner, f)
