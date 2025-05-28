import pygame
import numpy as np
from snake_game import SnakeGame
from dqn_agent import DQNAgent
import time

def test():
    game = SnakeGame(speed=30)  # 降低速度以便觀察
    state_size = 13  # 更新為新的state size（包含到食物的距離）
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    agent.load("best_model.pth")  # 載入最佳模型
    
    # 設置為純利用模式（不探索）
    agent.epsilon = 0
    
    n_games = 50
    scores = []
    
    for game_i in range(n_games):
        state = game.reset()
        done = False
        score = 0
        
        while not done:
            for event in pygame.event.get():  # 處理主循環中的事件
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                    
            action_idx = agent.act(state)
            action = [0] * 3
            action[action_idx] = 1
            
            reward, done, score = game.step(action)
            state = game.get_state()
            
            game.render()
            
            if done:
                scores.append(score)
                print(f'Game {game_i}, Score {score}, Avg Score {np.mean(scores):.2f}')
                time.sleep(1)  # 遊戲結束時暫停1秒
    
    print(f'\nResults after {n_games} games:')
    print(f'Average Score: {np.mean(scores):.2f}')
    print(f'Max Score: {max(scores)}')
    print(f'Min Score: {min(scores)}')
    
    pygame.quit()

if __name__ == '__main__':
    test()
