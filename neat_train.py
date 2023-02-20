import neat
import pickle
import snake_gameAI

# Define the fitness function
def evaluate_genomes(genomes, config):
    print("evaluating genomes")
    for genome_id, genome in genomes:
        #print("genome id: ", genome_id)
        # Create a neural network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Play the game with the neural network
        score= snake_gameAI.play_game(net)

        # Penalize the genome if it repeatedly moves in the same direction
        genome.fitness = score 
        #print("fitness score: ", genome.fitness)

# Set up the NEAT algorithm configuration
config_path = "./config/neat-config.txt"
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

# Create a population and run the NEAT algorithm to evolve the population
pop = neat.population.Population(config)
pop.add_reporter(neat.StdOutReporter(True))
stats = neat.statistics.StatisticsReporter()
pop.add_reporter(stats)
pop.add_reporter(neat.Checkpointer(5, filename_prefix="./config/NeatCheckpoints/snake-checkpoint-"))
winner = pop.run(evaluate_genomes)

# Save the best neural network to a file|
with open("snake-best-network", "wb") as f:
    pickle.dump(winner, f)
