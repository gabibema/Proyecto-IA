# importing libraries
import pygame
import time
import random
from enum import Enum
import numpy as np
from collections import namedtuple
pygame.init()
font = pygame.font.Font(None,25) 
 
snake_speed = 15
BLOCK_SIZE = 10
 
# Window size
window_x = 720
window_y = 480
 
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
    
SPEED = 15
Point = namedtuple('Point','x , y')

# Initialising pygame
pygame.init()
 
# Initialise game window

class SnakeGame:
    def __init__(self):
        self.w = 720
        self.h = 480
        pygame.display.set_caption('Snake Game')
        self.display  = pygame.display.set_mode((window_x, window_y))
        self.clock = pygame.time.Clock()
        
        self.reset()

    def __place_food(self):
        self.food = Point(random.randrange(1, (window_x//10)) * BLOCK_SIZE,
                          random.randrange(1, (window_y//10)) * BLOCK_SIZE)        
        if(self.food in self.body):
            self._place__food()      


    def reset (self):
        self.direction = Direction.RIGHT
        self.head = Point(100,50)
        self.body = [self.head, Point(90, 50),
                     Point(80, 50)]
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
                if(event.key == pygame.K_LEFT):
                    self.direction = Direction.LEFT
                elif(event.key == pygame.K_RIGHT):
                    self.direction = Direction.RIGHT
                elif(event.key == pygame.K_UP):
                    self.direction = Direction.UP
                elif(event.key == pygame.K_DOWN):
                    self.direction = Direction.DOWN

    def play_step(self,direction = None):
        self.frame_iteration+=1

        if(direction == None):
            self.__get_move()
        self._move(self.direction)
        self.body.insert(0,self.head)
    
        reward = 0  
        self.game_over = False 

        if(self.is_collision() or self.frame_iteration > 100*len(self.body) ):
            self.game_over=True
            reward = -10
            return reward,self.game_over,self.score
            
        if(self.head == self.food):
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

    def _move(self,action):
        x = self.head.x
        y = self.head.y
        if(self.direction == Direction.RIGHT):
            x+=BLOCK_SIZE
        elif(self.direction == Direction.LEFT):
            x-=BLOCK_SIZE
        elif(self.direction == Direction.DOWN):
            y+=BLOCK_SIZE
        elif(self.direction == Direction.UP):
            y-=BLOCK_SIZE
        self.head = Point(x,y)
        
    def is_collision(self, pt=None):
        if(pt is None):
            pt = self.head
            
        if(pt.x>self.w-BLOCK_SIZE or pt.x<0 or pt.y>self.h - BLOCK_SIZE or pt.y<0):
            return True
        if(pt in self.body[1:]):
            return True
        return False