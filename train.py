import pygame
import time
import numpy as np
from snake_game import SnakeGame
from dqn_agent import DQNAgent
import matplotlib
matplotlib.use('Agg')  # 使用Agg後端，避免與pygame衝突
import matplotlib.pyplot as plt
import os
import pandas as pd
from datetime import datetime

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
    state_size = 11  # 更新為簡化後的state size (3個危險檢測 + 4個方向 + 4個食物位置)
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
    max_games = 300  # 最大訓練次數
    
    try:
        game_i = 0
        while True:
            # 檢查訓練時間和最大訓練次數
            if time.time() - start_time > max_training_time:
                print("\nReached 2 hours training time limit")
                break
            
            if game_i >= max_games:
                print(f"\nReached maximum number of games: {max_games}")
                break
                
            state = game.reset()
            done = False
            score = 0
            
            while not done:
                # 處理 pygame 事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                # 獲取動作
                action = agent.get_action(state)
                
                # 執行動作
                next_state, reward, done, score = game.step(action)
                
                # 儲存經驗
                agent.remember(state, action, reward, next_state, done)
                
                # 訓練網絡
                agent.train()
                
                state = next_state
                
                # 更新顯示
                game.render()
                
                if done:
                    # 遊戲結束，更新統計信息
                    game.reset()
                    agent.update_target_model()
                    
                    # 更新分數記錄
                    scores.append(score)
                    total_score += score
                    mean_score = total_score / (game_i + 1)
                    mean_scores.append(mean_score)
                    
                    if score > record:
                        record = score
                    
                    current_time = time.time()
                    if current_time - last_update_time >= update_interval:
                        print(f'Game {game_i}, Score {score}, Record {record}, Time: {(current_time-start_time)/60:.1f}m')
                        last_update_time = current_time
                        
                        # 保存訓練圖表
                        plot_training_history(scores, mean_scores)
                    
                    # 檢查是否達到目標分數
                    if score >= target_score:
                        print(f"\nReached target score of {target_score}!")
                        return
            
            game_i += 1
            
            # 如果內層循環因為達到目標分數而break，外層也要break
            if score >= target_score:
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:


        pygame.quit()
        # 保存最終圖表
        # 保存訓練數據到Excel
    try:
        # 創建時間戳作為文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = os.path.join("training_results", f"training_results_{timestamp}.xlsx")
        
        # 準備數據
        data = {
            "Game": list(range(len(scores))),
            "Score": scores,
            "Average Score": mean_scores
        }
        
        # 創建DataFrame並保存為Excel
        df = pd.DataFrame(data)
        df.to_excel(excel_path, index=False)
        
        print(f"\nTraining results saved to {excel_path}")
    except Exception as e:
        print(f"Failed to save Excel file: {e}")
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
