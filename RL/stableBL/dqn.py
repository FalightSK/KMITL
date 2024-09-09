import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return x

class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def reset(self):
        self.buffer = None
        self.buffer = deque(maxlen=self.max_size)

    def size(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 episodes=1000, lr=1e-4, gamma=0.99, batchsize=64, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=50000, 
                 target_update=1000, memory_size=int(1e6), frequency=4):
        # Hyperparameters side
        self.episodes = episodes
        self.lr = lr
        self.gamma = gamma
        self.batchsize = batchsize
        self.epsilon_start, self.epsilon_end, self.epsilon_decay = epsilon_start, epsilon_end, epsilon_decay
        self.target_update = target_update
        self.frequency = frequency
        self.replay_buffer = ReplayBuffer(max_size=memory_size)
        
        # ML Side
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_model = DQN(state_dim, action_dim).to(self.device)
        self.target_policy = DQN(state_dim, action_dim).to(self.device)
        self.target_policy.load_state_dict(self.policy_model.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=self.lr)
        
    def select_action(self, state, epsilon, action_dim):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device)
                return self.policy_model(state).argmax(dim=1).item()
        else:
            return random.randrange(action_dim)
        
    def train(self):
        
        if self.replay_buffer.size() < self.batchsize:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batchsize)
        
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.IntTensor(action).unsqueeze(1).to(self.device, dtype=torch.int64)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        q_value = self.policy_model(state).gather(1, action)
        next_q_value = self.target_policy(next_state).max(1)[0].detach()
        expected_q = reward + self.gamma * next_q_value * (1 - done)
        
        loss = self.criterion(q_value.squeeze(), expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        