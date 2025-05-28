import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998  # 稍微降低衰減速率，使探索持續更久
        self.learning_rate = 0.001
        self.update_rate = 5
        self.update_counter = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DQNNet(state_size, hidden_size, action_size).to(self.device)
        self.target_model = DQNNet(state_size, hidden_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            act_values = self.model(state)
            return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        self.update_counter += 1
        
        if self.update_counter % self.update_rate != 0:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        # 先將數據轉換為numpy數組
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # 然後轉換為tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get max Q values for next states from target model
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update model
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu'), weights_only=True))
        self.target_model.load_state_dict(self.model.state_dict())
