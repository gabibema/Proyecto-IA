import os
import sys
sys.path.append("../Proyecto-IA")
import multiprocessing

import neat
import pickle
import snake_gameAI as snake_gameAI

def evaluate_genomes_singleThread(genomes, config):
    print("evaluating genomes")
    best_score = 0
    for genome_id, genome in genomes:
        # Creo red neuronal para el genoma
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Juego con la red neuronal creada
        fitness, score = snake_gameAI.play_game(net)
        if score > best_score:
            best_score = score
        genome.fitness = fitness 
    print("best score: ", best_score)

def evaluate_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness, score = snake_gameAI.play_game(net)
    genome.fitness = fitness
    return fitness, score

def evaluate_genomes_multiThread(genomes, config):
    print("evaluating genomes")
    best_score = 0
    pool = multiprocessing.Pool(processes=48)
    results = [pool.apply_async(evaluate_genome, args=(genome, config)) for genome_id, genome in genomes]
    scores = [result.get() for result in results]
    pool.close()
    pool.join()
    for i, genome_score in enumerate(scores):
        genome = genomes[i][1]
        fitness, score = genome_score
        if score > best_score:
            best_score = score
        genome.fitness = fitness
    print("best score: ", best_score)

def main():
    # Configuracion de arranque de NEAT
    config_path = "./neat/config/neat-config"
    config = neat.config.Config(neat.DefaultGenome, 
                                neat.DefaultReproduction, 
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, 
                                config_path)

    # Creo poblacion a partir de la configuracion de NEAT
    pop = neat.population.Population(config)

    # Restauro poblacion a partir de checkpoint
    #checkpoint = neat.Checkpointer.restore_checkpoint('./neat/config/NeatCheckpoints/snake-checkpoint-359')
    #pop = checkpoint

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.statistics.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(10, filename_prefix="./neat/config/NeatCheckpoints/snake-checkpoint-"))

    winner = pop.run(evaluate_genomes_singleThread)
    
    # Guardo la red mejor red neuronal
    with open("./neat/snake-best-network", "wb") as bestFile:
        pickle.dump(winner, bestFile)

if __name__ == "__main__":
    main()
