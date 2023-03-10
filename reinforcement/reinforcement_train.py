import sys
sys.path.append("../Proyecto-IA")

from reinforcement_model import Agent
from snake_game import SnakeGame
from reinforcement_plot import plot

mean_scores = []
scores = []

def train():
    total_score = 0
    best_score = 0
    agent = Agent()
    game = SnakeGame()
    
    
    while True:
        state_old = game.get_state()
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()

        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if(score > reward): 
                reward = score
                agent.model.save()
            if(score > best_score):
                best_score = score
            print('Game:',agent.n_game,'Score:',score,'Best Score:',best_score)
            
            scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_game
            mean_scores.append(mean_score)
            plot(scores,mean_scores,best_score)


train()