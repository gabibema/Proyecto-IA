import sys
sys.path.append("../Proyecto-IA")
from snake_game import SnakeGame, Direction
import pygame
import math

def normalize(val, max):
    return abs(val/max)

def distance_to_food(head_x, head_y, food_x, food_y):
    return normalize(int(math.sqrt((head_x - food_x) ** 2 + (head_y - food_y) ** 2) / 10), 360)

def game_duration(start_time):
    return int((pygame.time.get_ticks() - start_time) / 3600)

def moving_towards_food(head_x, head_y, food_x, food_y, direction):
    if direction == Direction.UP:
        return 1 if head_y > food_y else 0 
    elif direction == Direction.DOWN:
        return 1 if head_y < food_y else 0
    elif direction == Direction.LEFT:
        return 1 if head_x > food_x else 0
    elif direction == Direction.RIGHT:
        return 1 if head_x < food_x else 0

def distance_to_walls(element, screen_width, screen_height):
    head_x, head_y = element
    
    north_dist = normalize(head_y, 360)
    east_dist = normalize(screen_width - head_x, 360)
    south_dist = normalize(screen_height - head_y, 360)
    west_dist = normalize(head_x, 360)
    
    return (north_dist, east_dist, south_dist, west_dist)

def polar_distance_to_food(head, food):
    head_x, head_y = head
    food_x, food_y = food

    # Calculate the distance in each direction
    north_dist = normalize(head_y - food_y, 360)
    east_dist = normalize(food_x - head_x, 360)
    south_dist = normalize(food_y - head_y, 360)
    west_dist = normalize(head_x - food_x, 360)

    return (north_dist, east_dist, south_dist, west_dist)

def play_game(network):
    game = SnakeGame()
    prev_game_score = 0
    ai_score = 0
    prev_food_distance = 360

    while not game.game_over:
        # Observe the current state of the game
        head_x, head_y = game.head
        head_x = normalize(head_x, 360)
        head_y = normalize(head_y, 360)
        food_x, food_y = game.food
        food_x = normalize(food_x, 360)
        food_y = normalize(food_y, 360)        
        is_approaching_food = normalize(moving_towards_food(head_x, head_y, food_x, food_y, game.direction), 360)
        food_distance = normalize(distance_to_food(head_x, head_y, food_x, food_y), 360)
        state = [head_x, head_y, food_x, food_y, food_distance, is_approaching_food]
        
        north_dist, east_dist, south_dist, west_dist = distance_to_walls(game.head, game.w, game.h)
        state += [north_dist, east_dist, south_dist, west_dist]
        
        food_north_dist, food_east_dist, food_south_dist, food_west_dist = polar_distance_to_food(game.head, game.food)
        state += [food_north_dist, food_east_dist, food_south_dist, food_west_dist]
        #print(state)

        # Use the network to choose an action
        output = network.activate(state)
        #print("output: ", output)
        action = output.index(max(output))
        #print("action: ", action)

        # Convert the action to a direction
        if action == 0:
            game.direction = Direction.UP
        elif action == 1:
            game.direction = Direction.DOWN
        elif action == 2:
            game.direction = Direction.LEFT
        elif action == 3:
            game.direction = Direction.RIGHT
            
        game.play_step()
        
        if game.score > prev_game_score:
            ai_score += 200    
        prev_game_score = game.score
        
        if food_distance > prev_food_distance:
            ai_score += 5 
        else: 
            ai_score -= 1
        prev_food_distance = food_distance        

        duration = game_duration(game.start_time)

    return ai_score + duration
