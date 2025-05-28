import pygame
import time
import numpy as np
from snake_game import SnakeGame
from dqn_agent import DQNAgent
import matplotlib
matplotlib.use('Agg')  # 使用Agg後端，避免與pygame衝突
import matplotlib.pyplot as plt
import os

def plot_training_history(scores, mean_scores, filename='training_plot.png'):
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
    plt.savefig(filename)
    plt.close()

def train():
    pygame.init()
    game = SnakeGame(speed=50)
    state_size = 13  # 更新state大小為13（包含到食物的距離特徵）
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    
    last_update_time = time.time()
    start_time = time.time()
    update_interval = 1.0
    
    max_training_time = 7200  # 2小時 = 7200秒
    target_score = 150  # 目標分數
    
    try:
        game_i = 0
        while True:
            # 檢查訓練時間
            if time.time() - start_time > max_training_time:
                print("\nReached 2 hours training time limit")
                break
                
            state = game.reset()
            done = False
            score = 0
            
            while not done:
                # 處理 pygame 事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                action_idx = agent.act(state)
                action = [0] * 3
                action[action_idx] = 1
                
                reward, done, score = game.step(action)
                next_state = game.get_state()
                
                agent.remember(state, action_idx, reward, next_state, done)
                agent.replay()
                state = next_state
                
                game.render()
                
                if done:
                    agent.update_target_model()
                    scores.append(score)
                    total_score += score
                    mean_score = total_score / (game_i + 1)
                    mean_scores.append(mean_score)
                    
                    if score > record:
                        record = score
                        agent.save("best_model.pth")
                        
                        # 檢查是否達到目標分數
                        if score >= target_score:
                            print(f"\nReached target score of {target_score}!")
                            break
                    
                    print(f'Game {game_i}, Score {score}, Record {record}, Time: {(time.time()-start_time)/60:.1f}m')
                    
                    # 更新圖表到文件
                    current_time = time.time()
                    if current_time - last_update_time >= update_interval:
                        plot_training_history(scores, mean_scores)
                        last_update_time = current_time
                    
                    time.sleep(0.5)  # 每局結束後暫停一下
            
            game_i += 1
            
            # 如果內層循環因為達到目標分數而break，外層也要break
            if score >= target_score:
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        pygame.quit()
        # 保存最終圖表
        plot_training_history(scores, mean_scores)
        # 保存最終模型
        agent.save("final_model.pth")
        
        # 輸出訓練統計
        training_time = (time.time() - start_time) / 60  # 轉換為分鐘
        print(f"\nTraining Summary:")
        print(f"Total training time: {training_time:.1f} minutes")
        print(f"Games played: {game_i}")
        print(f"Best score: {record}")
        print(f"Final average score: {mean_scores[-1]:.2f}")

if __name__ == '__main__':
    train()
