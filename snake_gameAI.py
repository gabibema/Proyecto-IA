from snake_game import SnakeGame, Direction
import pygame
import math

def distance_to_food(head_x, head_y, food_x, food_y):
    return int(math.sqrt((head_x - food_x) ** 2 + (head_y - food_y) ** 2) / 20)

def game_duration(start_time):
    return int((pygame.time.get_ticks() - start_time) / 3600)

def moving_towards_food(head_x, head_y, food_x, food_y, direction):
    if direction == Direction.UP:
        return head_y > food_y
    elif direction == Direction.DOWN:
        return head_y < food_y
    elif direction == Direction.LEFT:
        return head_x > food_x
    elif direction == Direction.RIGHT:
        return head_x < food_x

def one_hot_direction(direction):
    if direction == Direction.UP:
        return 3 
    elif direction == Direction.DOWN:
        return 4
    elif direction == Direction.LEFT:
        return 2
    elif direction == Direction.RIGHT:
        return 1

def play_game(network):
    game = SnakeGame()
    ai_score = 0
    last_distance_to_food = None
    last_direction = Direction.RIGHT
    was_approaching_food = False

    while not game.game_over:
        # Observe the current state of the game
        head_x, head_y = game.head
        food_x, food_y = game.food
        body = game.body
        duration = game_duration(game.start_time)
        direction = one_hot_direction(game.direction)
        is_approaching_food = moving_towards_food(head_x, head_y, food_x, food_y, game.direction)
        distance = distance_to_food(head_x, head_y, food_x, food_y)
        
        # add observed state
        state = [head_x, head_y, food_x, food_y, direction, distance, is_approaching_food]
        for segment in body:
            state += [segment[0], segment[1]]

        # Pad the state with zeros if the snake is shorter than 4 segments
        while len(state) < 24:
            state += [0, 0]

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

        ai_score = int((game.score + 1) * (duration ** 2.1) + (game.score ^ 2))

        
    return ai_score 
