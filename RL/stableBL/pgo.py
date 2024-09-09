import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.distributions import Categorical

class DiscretePGO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscretePGO, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation
        x = self.fc2(x)  # Linear output for actions
        return torch.softmax(x, dim=-1)  # Softmax to output action probabilities

# REINFORCE agent
class DiscreteAgent:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.99):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_network = DiscretePGO(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma  # Discount factor
        self.episode_rewards = []
        self.episode_log_probs = []

    def select_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)  # Convert state to tensor
        action_probs = self.policy_network(state)  # Get action probabilities
        dist = Categorical(action_probs)  # Create categorical distribution
        action = dist.sample()  # Sample an action from the distribution
        log_prob = dist.log_prob(action)  # Get log probability of the action
        return action.item(), log_prob

    def store_outcome(self, reward, log_prob):
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)

    def update_policy(self):
        discounted_rewards = []
        R = 0
        for r in reversed(self.episode_rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(np.array(discounted_rewards))
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        policy_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
        for log_prob, reward in zip(self.episode_log_probs, discounted_rewards):
            #policy_loss.append(-log_prob * reward)
            policy_loss += -log_prob*reward

        #policy_loss = torch.tensor(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.episode_rewards = []
        self.episode_log_probs = []


from torch.distributions import Normal

class ContinuousPGO(nn.Module):
    def __init__(self, input_size, output_size):
        super(ContinuousPGO, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean_layer = nn.Linear(128, output_size)  # Output layer for the mean
        self.log_std = nn.Parameter(torch.zeros(1, output_size))  # Log standard deviation as a parameter

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.mean_layer(x)  # Mean of the Gaussian distribution
        std = torch.exp(self.log_std)  # Standard deviation of the Gaussian (log_std is optimized)
        return mean, std

class ContinuousAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_network = ContinuousPGO(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma  # Discount factor
        self.episode_rewards = []
        self.episode_log_probs = []

    def select_action(self, state, action_low, aciton_high):
        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        mean, std = self.policy_network(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum() 
        return action.clamp(action_low, aciton_high).item(), log_prob

    def store_outcome(self, reward, log_prob):
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_prob)

    def update_policy(self):
        discounted_rewards = []
        R = 0
        for r in reversed(self.episode_rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        policy_loss = torch.tensor(0, dtype=torch.float64).to(self.device)
        for log_prob, reward in zip(self.episode_log_probs, discounted_rewards):
            # policy_loss.append(-log_prob * reward)
            policy_loss += -log_prob*reward

        # policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.episode_rewards = []
        self.episode_log_probs = []