import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGame:
    def __init__(self, width=640, height=480, speed=50):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.speed = speed
        self.max_steps_without_food = 500  # 初始化最大步數
        
        # Initialize game state
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.width//2, self.height//2)
        self.snake = [
            self.head,
            Point(self.head.x-20, self.head.y),
            Point(self.head.x-40, self.head.y)
        ]
        self.score = 0
        self.food = self._place_food()
        self.frame_iteration = 0
        return self.get_state()

    def _place_food(self):
        x = random.randint(0, (self.width-20)//20) * 20
        y = random.randint(0, (self.height-20)//20) * 20
        food = Point(x, y)
        while food in self.snake:
            x = random.randint(0, (self.width-20)//20) * 20
            y = random.randint(0, (self.height-20)//20) * 20
            food = Point(x, y)
        return food

    def get_state(self):
        head = self.snake[0]
        
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # 計算標準化的蛇身長度 (除以遊戲區域可能的最大長度)
        max_possible_length = (self.width // 20) * (self.height // 20)
        normalized_length = len(self.snake) / max_possible_length

        # 計算到食物的曼哈頓距離並標準化
        manhattan_distance = abs(self.food.x - head.x) + abs(self.food.y - head.y)
        max_possible_distance = self.width + self.height
        normalized_distance = manhattan_distance / max_possible_distance

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y,  # food down
            
            # Snake length (normalized)
            normalized_length,
            
            # Distance to food (normalized)
            normalized_distance
        ]
        return np.array(state, dtype=float)

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.width - 20 or pt.x < 0 or pt.y > self.height - 20 or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def step(self, action):
        self.frame_iteration += 1
        reward = 0
        game_over = False
        
        # Convert action [straight, right, left] to new direction
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if action == [1, 0, 0]:  # straight
            new_dir = clock_wise[idx]
        elif action == [0, 1, 0]:  # right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # left turn
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += 20
        elif self.direction == Direction.LEFT:
            x -= 20
        elif self.direction == Direction.DOWN:
            y += 20
        elif self.direction == Direction.UP:
            y -= 20

        self.head = Point(x, y)

        # Check if game is over
        if self._is_collision() or self.frame_iteration > self.max_steps_without_food:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Move snake
        self.snake.insert(0, self.head)
        
        # Check if snake got food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
            self.frame_iteration = 0  # 重置步數計數器
        else:
            self.snake.pop()
        
        return reward, game_over, self.score

    def render(self):
        for event in pygame.event.get():  # 處理事件隊列
            if event.type == pygame.QUIT:
                pygame.quit()
                return
                
        self.display.fill((0,0,0))
        
        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt.x, pt.y, 20, 20))
            
        pygame.draw.rect(self.display, (255,0,0), pygame.Rect(self.food.x, self.food.y, 20, 20))
        
        pygame.display.flip()
        self.clock.tick(self.speed)
