import sys
sys.path.append("../Proyecto-IA")
from snake_game import SnakeGame, Direction
import pygame
import math

def normalize(val, max):
    return abs(val/max)

def distance_to_food(head, food):
    head_x, head_y = head
    food_x, food_y = food
    return (math.sqrt((head_x - food_x) ** 2 + (head_y - food_y) ** 2))

def game_duration(start_time):
    return int((pygame.time.get_ticks() - start_time) / 60)

def moving_towards_food(head, food, direction):
    head_x, head_y = head
    food_x, food_y = food
        
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

def is_food(head, food): 
    head_x, head_y = head
    food_x, food_y = food
    
    if head_x == food_x:
        is_food_right = 0
        is_food_left = 0
    elif head_x < food_x:
        is_food_right = 1
        is_food_left = 0
    else:
        is_food_right = 0
        is_food_left = 1

    if head_y == food_y:
        is_food_up = 0
        is_food_down = 0
    elif head_y < food_y:
        is_food_up = 0
        is_food_down = 1
    else: 
        is_food_up = 1
        is_food_down = 0
            
    return (is_food_up, is_food_right, is_food_down, is_food_left)

def corner_distances(head):
    head_x, head_y = head
    
    topLeft_corner_distance = normalize(math.sqrt((head_x - 0) ** 2 + (head_y - 0) ** 2), math.sqrt(2*360*360)) 
    topRight_corner_distance = normalize(math.sqrt((head_x - 360) ** 2 + (head_y - 0) ** 2), math.sqrt(2*360*360))
    bottomLeft_corner_distance = normalize(math.sqrt((head_x - 0) ** 2 + (head_y - 360) ** 2), math.sqrt(2*360*360)) 
    bottomRight_corner_distance = normalize(math.sqrt((head_x - 360) ** 2 + (head_y - 360) ** 2), math.sqrt(2*360*360)) 
    
    return topLeft_corner_distance, topRight_corner_distance, bottomLeft_corner_distance, bottomRight_corner_distance

def print_state(state):
    #print("snake pos (x,y): ", state[0], state[1], " food pos (x,y): ", state[2], state[3])
    #print("approaching food: ", state[4], "food distance: ", state[5])
    #print("n_wall_dist: ", state[6], "e_wall_dist: ", state[7], "s_wall_dist: ", state[8], "w_wall_dist: ", state[9])
    #print("is food up: ", state[10], "is food right: ", state[11], "is food down: ", state[12], "is food left: ", state[13])
    #print("body size: ", state[14])
    print("top left corner distance: ", state[15], " top right corner distance: ", state[16], " bottom left corner distance: ", state[17], " bottom right corner distance: ", state[18])

def play_game(network):
    game = SnakeGame()
    prev_game_score = 0
    ai_score = 0
    prev_food_distance = math.sqrt(2*360*360)
    was_approaching_food = False

    while not game.game_over:
        # Observe the current state of the game
        body_size = len(game.body)
        head_x, head_y = game.head
        head_x = normalize(head_x, 360)
        head_y = normalize(head_y, 360)
        food_x, food_y = game.food
        food_x = normalize(food_x, 360)
        food_y = normalize(food_y, 360)        
        is_approaching_food = moving_towards_food(game.head, game.food, game.direction)
        food_distance = normalize(distance_to_food(game.head, game.food), math.sqrt(2*360*360))
        state = [head_x, head_y, food_x, food_y, is_approaching_food, food_distance]
        
        north_dist, east_dist, south_dist, west_dist = distance_to_walls(game.head, game.w, game.h)
        state += [north_dist, east_dist, south_dist, west_dist]
        
        is_food_up, is_food_right, is_food_down, is_food_left = is_food(game.head, game.food)
        state += [is_food_up, is_food_right, is_food_down, is_food_left]
        
        state += [body_size]
        
        topLeft_corner_distance, topRight_corner_distance, bottomLeft_corner_distance, bottomRight_corner_distance = corner_distances(game.head)
        state += [topLeft_corner_distance, topRight_corner_distance, bottomLeft_corner_distance, bottomRight_corner_distance]
        
        #print_state(state)

        # Use the network to choose an action
        output = network.activate(state)
        action = output.index(max(output))

        # Convert the action to a direction
        if action == 0 and game.direction != Direction.DOWN:
            game.direction = Direction.UP
        elif action == 1 and game.direction != Direction.UP:
            game.direction = Direction.DOWN
        elif action == 2 and game.direction != Direction.RIGHT:
            game.direction = Direction.LEFT
        elif action == 3 and game.direction != Direction.LEFT:
            game.direction = Direction.RIGHT
            
        game.play_step()
        
        if game.score > prev_game_score:
            ai_score += 500    
        prev_game_score = game.score
        
        if food_distance < prev_food_distance:
            ai_score += 1 
        prev_food_distance = food_distance
        
        if not was_approaching_food and is_approaching_food:
            ai_score += 10
        elif was_approaching_food and is_approaching_food:
            ai_score += 2
        elif not was_approaching_food and not is_approaching_food:
            ai_score -= 1
        was_approaching_food = is_approaching_food
        
        duration = game_duration(game.start_time)
        #print("current score: ", ai_score)

    return ai_score + duration * 2
