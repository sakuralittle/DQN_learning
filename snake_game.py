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
        # 確保pygame已初始化
        if not pygame.get_init():
            pygame.init()
            
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

        # 檢查危險 - 直接檢查當前方向的正前方、右側和左側是否有障礙物
        danger_straight = (dir_r and self._is_collision(point_r)) or (dir_l and self._is_collision(point_l)) or (dir_u and self._is_collision(point_u)) or (dir_d and self._is_collision(point_d))
        danger_right = (dir_u and self._is_collision(point_r)) or (dir_d and self._is_collision(point_l)) or (dir_l and self._is_collision(point_u)) or (dir_r and self._is_collision(point_d))
        danger_left = (dir_d and self._is_collision(point_r)) or (dir_u and self._is_collision(point_l)) or (dir_r and self._is_collision(point_u)) or (dir_l and self._is_collision(point_d))
        
        # 食物相對頭部的位置（簡化為四個方向）
        food_left = self.food.x < head.x
        food_right = self.food.x > head.x
        food_up = self.food.y < head.y
        food_down = self.food.y > head.y
        
        state = [
            # 危險區域檢測 (3)
            danger_straight,
            danger_right,
            danger_left,
            
            # 移動方向 (4)
            dir_l, dir_r, dir_u, dir_d,
            
            # 食物相對位置 (4)
            food_left, food_right, food_up, food_down
        ]
        
        return np.array(state, dtype=int)

    def _check_trapped(self):
        """檢查蛇是否有被困的風險"""
        head = self.snake[0]
        # 檢查周圍四個方向
        available_moves = 0
        for next_pos in [Point(head.x-20, head.y), Point(head.x+20, head.y),
                        Point(head.x, head.y-20), Point(head.x, head.y+20)]:
            if not self._is_collision(next_pos):
                available_moves += 1
        return available_moves <= 1  # 如果可用移動方向<=1，認為有被困風險

    def _get_body_relative_positions(self):
        """獲取身體相對於頭部的位置信息"""
        head = self.snake[0]
        # 初始化四個方向的身體段數量
        left = right = up = down = 0
        
        for segment in self.snake[1:]:
            if segment.x < head.x:
                left += 1
            elif segment.x > head.x:
                right += 1
            if segment.y < head.y:
                up += 1
            elif segment.y > head.y:
                down += 1
        
        # 標準化
        total = len(self.snake) - 1
        return [left/total if total > 0 else 0,
                right/total if total > 0 else 0,
                up/total if total > 0 else 0,
                down/total if total > 0 else 0]

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

    def _move(self, action):
        # action是一個整數：0(直行)，1(右轉)，2(左轉)
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if action == 0:  # straight
            new_dir = clock_wise[idx]
        elif action == 1:  # right turn
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

        # Move snake
        self.snake.insert(0, self.head)
        
        # Check if snake got food
        if self.head == self.food:
            self.score += 1
            self.food = self._place_food()
            self.frame_iteration = 0  # 重置步數計數器
        else:
            self.snake.pop()

    def step(self, action):
        self.frame_iteration += 1
        
        # 收集移動前的信息
        old_head = self.head
        old_distance = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)
        
        # 移動
        self._move(action)
        
        # 收集移動後的信息
        new_distance = abs(self.food.x - self.head.x) + abs(self.food.y - self.head.y)
        
        reward = 0
        game_over = False
        
        # 檢查碰撞
        if self._is_collision():
            game_over = True
            reward = -10
            return self.get_state(), reward, game_over, self.score
            
        # 檢查是否吃到食物
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
            self.frame_iteration = 0
        else:
            # 靠近食物的獎勵
            if new_distance < old_distance:
                reward = 0.1
            else:
                reward = -0.05
                
            # 獎勵移動，避免原地打轉
            distance_moved = abs(self.head.x - old_head.x) + abs(self.head.y - old_head.y)
            if distance_moved > 0:
                reward += 0.01
        
        # 超時懲罰，但時間更短以加快學習
        if self.frame_iteration > 50 * len(self.snake):
            game_over = True
            reward = -1
        
        return self.get_state(), reward, game_over, self.score

    def render(self):
        # 確保pygame已初始化
        if not pygame.get_init():
            pygame.init()
            
        for event in pygame.event.get():  # 處理事件隊列
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
        self.display.fill((0,0,0))
        
        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), pygame.Rect(pt.x, pt.y, 20, 20))
            
        pygame.draw.rect(self.display, (255,0,0), pygame.Rect(self.food.x, self.food.y, 20, 20))
        
        pygame.display.flip()
        self.clock.tick(self.speed)
