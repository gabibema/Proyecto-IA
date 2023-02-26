import sys
sys.path.append("../Proyecto-IA")

from reinforcement_model import Agent
from snake_game import SnakeGame

def train():
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame()
    
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old,final_move,reward,state_new,done)
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if(score > reward): 
                reward = score
                agent.model.save()
            print('Game:',agent.n_game,'Score:',score,'Record:',record)
            
            total_score+=score


train()