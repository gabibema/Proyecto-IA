import sys
sys.path.append("../Proyecto-IA")
from snake_game import SnakeGame, Direction
import pygame
import math

# Tamaño elementos
BLOCK_SIZE = 20

# Utilizado para normalizar los estados
def normalize(val, max):
    return abs(val/max)

# distancia euclidiana de la cabeza de la serpiente
# a la comida
def distance_to_food(head, food):
    head_x, head_y = head
    food_x, food_y = food
    return (math.sqrt((head_x - food_x) ** 2 + (head_y - food_y) ** 2))

# duracion actual de la partida
def game_duration(start_time):
    return int((pygame.time.get_ticks() - start_time) / 1000)

# bool que indica si la serpiente se esta acercando hacia la comida
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

# distancia a los muros norte, sur, este, oeste
def distance_to_walls(element, screen_width, screen_height):
    head_x, head_y = element
    
    north_dist = normalize(head_y, screen_height)
    east_dist = normalize(screen_width - head_x, screen_width)
    south_dist = normalize(screen_height - head_y, screen_height)
    west_dist = normalize(head_x, screen_height)
    
    return (north_dist, east_dist, south_dist, west_dist)

# hot encoding para acercamiento hacia comida
# para direccion arriba, abajo, izquierda y derecha
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

# distancia euclidia a las cuatro esquinas
def corner_distances(head, screen_width, screen_height):
    head_x, head_y = head
    
    topLeft_corner_distance = normalize(math.sqrt((head_x - 0) ** 2 + (head_y - 0) ** 2), math.sqrt(2*screen_height*screen_width)) 
    topRight_corner_distance = normalize(math.sqrt((head_x - screen_width) ** 2 + (head_y - 0) ** 2), math.sqrt(2*screen_height*screen_width))
    bottomLeft_corner_distance = normalize(math.sqrt((head_x - 0) ** 2 + (head_y - screen_height) ** 2), math.sqrt(2*screen_height*screen_width)) 
    bottomRight_corner_distance = normalize(math.sqrt((head_x - screen_width) ** 2 + (head_y - screen_height) ** 2), math.sqrt(2*screen_height*screen_width)) 
    
    return topLeft_corner_distance, topRight_corner_distance, bottomLeft_corner_distance, bottomRight_corner_distance

# hot encoding para cabeza a una unidad de distancia
# de una parte del cuerpo
def body_close(head, body):

    topLeft_body_close = 0
    top_body_close = 0
    topRight_body_close = 0
    
    Left_body_close = 0
    Right_body_close = 0

    bottomLeft_body_close = 0
    bottom_body_close = 0
    bottomRight_body_close = 0

    head_x, head_y = head
    for part in body:
        #print("head: ", head, "body part: ", part)
        part_x, part_y = part
        if part_x == head_x-BLOCK_SIZE and part_y == head_y-BLOCK_SIZE:
            topLeft_body_close = 1
        elif part_x == head_x and part_y == head_y-BLOCK_SIZE:
            top_body_close = 1
        elif part_x == head_x+BLOCK_SIZE and part_y == head_y-BLOCK_SIZE:
            topRight_body_close = 1
        elif part_x == head_x-BLOCK_SIZE and part_y == head_y:
            Left_body_close = 1
        elif part_x == head_x+BLOCK_SIZE and part_y == head_y:
            Right_body_close = 1
        elif part_x == head_x-BLOCK_SIZE and part_y == head_y+BLOCK_SIZE:
            bottomLeft_body_close = 1
        elif part_x == head_x and part_y == head_y+BLOCK_SIZE:
            bottom_body_close = 1
        elif part_x == head_x+BLOCK_SIZE and part_y == head_y+BLOCK_SIZE:
            bottomRight_body_close = 1

    #print(topLeft_body_close, top_body_close, topRight_body_close, Left_body_close, Right_body_close, bottomLeft_body_close, bottom_body_close, bottomRight_body_close) 
    return topLeft_body_close, top_body_close, topRight_body_close, Left_body_close, Right_body_close, bottomLeft_body_close, bottom_body_close, bottomRight_body_close

# obtengo el estado actual del juego
def get_state(game: SnakeGame):
    state = []

    head_x, head_y = game.head
    head_x = normalize(head_x, game.w)
    head_y = normalize(head_y, game.h)
    food_x, food_y = game.food
    food_x = normalize(food_x, game.w)
    food_y = normalize(food_y, game.h)        
    is_approaching_food = moving_towards_food(game.head, game.food, game.direction)
    food_distance = normalize(distance_to_food(game.head, game.food), math.sqrt(2*game.w*game.h))
    state = [head_x, head_y, food_x, food_y, is_approaching_food, food_distance]
        
    north_dist, east_dist, south_dist, west_dist = distance_to_walls(game.head, game.w, game.h)
    state += [north_dist, east_dist, south_dist, west_dist]
        
    is_food_up, is_food_right, is_food_down, is_food_left = is_food(game.head, game.food)
    state += [is_food_up, is_food_right, is_food_down, is_food_left]
        
    topLeft_corner_distance, topRight_corner_distance, bottomLeft_corner_distance, bottomRight_corner_distance = corner_distances(game.head, game.w, game.h)
    state += [topLeft_corner_distance, topRight_corner_distance, bottomLeft_corner_distance, bottomRight_corner_distance]
    
    topLeft_body_close, top_body_close, topRight_body_close, Left_body_close, Right_body_close, bottomLeft_body_close, bottom_body_close, bottomRight_body_close = body_close(game.head, game.body)
    state += [topLeft_body_close, top_body_close, topRight_body_close, Left_body_close, Right_body_close, bottomLeft_body_close, bottom_body_close, bottomRight_body_close]
    
    return state

def play_game(network):
    game = SnakeGame()
    prev_game_score = 0
    ai_score = 0
    movements = 0
    
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
        movements += 1

        # incremento el score de la ia si come la fruta
        if game.score > prev_game_score:
            ai_score += game.score   
        prev_game_score = game.score
        
        # fuerzo a terminar las serpientes que dan vueltas en circulos
        if(movements >= 100 and game.score == 0):
            game.game_over = True
    
    duration = game_duration(game.start_time)
    fitness_score = game.score*100 - duration
    return fitness_score, game.score

