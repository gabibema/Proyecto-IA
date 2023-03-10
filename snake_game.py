import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import time
import random
from enum import Enum
import numpy as np
from collections import namedtuple
pygame.init()
font = pygame.font.Font(None,25) 
 
SPEED = 40
BLOCK_SIZE = 20
 
# defining colors
black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point','x , y')

# Initialising pygame
pygame.init()
 
# Initialise game window
class SnakeGame:
    def __init__(self):
        self.w = 640
        self.h = 480
        pygame.display.set_caption('Snake Game')
        self.display  = pygame.display.set_mode((self.w, self.h)) 
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()
        self.reset()

    def __place_food(self):

        x = random.randint(0,(self.w-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        y = random.randint(0,(self.h-BLOCK_SIZE)//BLOCK_SIZE)*BLOCK_SIZE
        self.food = Point(x,y)        
        if(self.food in self.body):
            self.__place_food()      

    def reset (self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2,self.h/2)
        self.body = [self.head,
                    Point(self.head.x-BLOCK_SIZE,self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE),self.head.y)]
        self.score = 0
        self.food = None
        self.__place_food()
        self.game_over = False
        self.frame_iteration = 0          

    def __get_move(self):
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()
            if(event.type == pygame.KEYDOWN):
                if(event.key == pygame.K_LEFT and self.direction != Direction.RIGHT):
                    self.direction = Direction.LEFT
                elif(event.key == pygame.K_RIGHT and self.direction != Direction.LEFT):
                    self.direction = Direction.RIGHT
                elif(event.key == pygame.K_UP and self.direction != Direction.DOWN):
                    self.direction = Direction.UP
                elif(event.key == pygame.K_DOWN and self.direction != Direction.UP):
                    self.direction = Direction.DOWN
                return
                    
    def get_state(self):
        head = self.head
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            (dir_u and self.is_collision(point_u))or
            (dir_d and self.is_collision(point_d))or
            (dir_l and self.is_collision(point_l))or
            (dir_r and self.is_collision(point_r)),
                
            (dir_u and self.is_collision(point_r))or
            (dir_d and self.is_collision(point_l))or
            (dir_u and self.is_collision(point_u))or
            (dir_d and self.is_collision(point_d)),

            (dir_u and self.is_collision(point_r))or
            (dir_d and self.is_collision(point_l))or
            (dir_r and self.is_collision(point_u))or
            (dir_l and self.is_collision(point_d)),


            dir_l,
            dir_r,
            dir_u,
            dir_d,

            self.food.x < self.head.x, 
            self.food.x > self.head.x,
            self.food.y < self.head.y, 
            self.food.y > self.head.y,
        ] 
        
        return np.array(state,dtype=int)
    
    def check_quit(self):
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.quit()
                quit()
            
    def __get_direction_from_action(self,action):
        clock_wise = [Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action,[1,0,0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left Turn
        
        self.check_quit()
        self.direction = new_dir

    def play_step(self,action = None):
        self.frame_iteration+=1
        if(action == None):
            self.__get_move()
        else:
            self.__get_direction_from_action(action)

        self._move(self.direction)
        self.body.insert(0,self.head)
    
        reward = 0  
        self.game_over = False 

        if(self.is_collision() or self.frame_iteration > 100*len(self.body) ):
            self.game_over=True
            reward = -10
            return reward,self.game_over,self.score
            
        elif(self.head == self.food):
            self.score+=1
            reward=10
            self.__place_food()
        else:
            self.body.pop()

        
        self.__update_ui()
        self.clock.tick(SPEED)
        
        return reward,self.game_over,self.score
    
    def __update_ui(self):
        self.display.fill(black)
        for pt in self.body:
            pygame.draw.rect(self.display,green,pygame.Rect(pt.x,pt.y,BLOCK_SIZE,BLOCK_SIZE))
        
        pygame.draw.rect(self.display,red,pygame.Rect(self.food.x,self.food.y,BLOCK_SIZE,BLOCK_SIZE))
        text = font.render("Score: "+str(self.score),True,white)
        self.display.blit(text,[0,0])
        pygame.display.flip()
    
    def _move(self, action):
        x = self.head.x
        y = self.head.y

        if action == Direction.RIGHT:
            x += BLOCK_SIZE
        elif action == Direction.LEFT:
            x -= BLOCK_SIZE
        elif action == Direction.DOWN:
            y += BLOCK_SIZE
        elif action == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)

    def is_collision(self, pt=None):
        if(pt is None):
            pt = self.head
            
        if(pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y<0):
            return True
        if(pt in self.body[1:]):
            return True
        return False