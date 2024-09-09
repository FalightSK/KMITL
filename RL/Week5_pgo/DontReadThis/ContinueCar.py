import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import random

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPISODES = 500
GAMMA = 0.99
LR = 0.0001
BATCH_SIZE = 256
MEMORY_SIZE = 10000
MAX_STEP = 175

# Define the neural network
class PolicyGradientNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyGradientNN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc4_mu = nn.Linear(64, action_size)
        self.fc4_logvar = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = torch.tanh(self.fc4_mu(x))
        logvar = self.fc4_logvar(x)
        logvar = torch.clamp(logvar, -20, 2)  # Clipping log variance
        sigma = torch.exp(logvar)
        return mu, sigma

# Initialize the environment
env = gym.make("MountainCarContinuous-v0")

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

print(state_size, action_size)

torch.manual_seed(42)

# Initialize the Policy Gradient
policy_net = PolicyGradientNN(state_size, action_size).to(device)
best_net = PolicyGradientNN(state_size, action_size).to(device)

# Check if the models are on the GPU
print(f"Policy Net on device: {next(policy_net.parameters()).device}")

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = []

def select_action(observation, net):
    state = torch.tensor(observation, dtype=torch.float32).to(device)
    state = state.unsqueeze(0)

    mu, sigma = net(state)

    action_prob = torch.distributions.Normal(loc=mu, scale=sigma)
    action = action_prob.sample()
    log_probs = action_prob.log_prob(action).sum()

    action = action.cpu().numpy()
    action = np.clip(action, env.action_space.low, env.action_space.high)
    return [action.item()], log_probs

def learn():
    states, actions, rewards, next_states, dones, log_probs = zip(*memory)
    
    R = 0
    returns = []
    for r in rewards[::-1]:
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    log_probs = torch.stack(log_probs).to(device)

    loss = -log_probs * returns
    loss = loss.mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Training loop
rewards = []
mx_rewards = -1e9
minstep = 1e9

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    step = 0
    
    while len(memory) < MEMORY_SIZE:
        action, log_prob = select_action(state, policy_net)
        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done, log_prob))
        state = next_state

        step += 1
        
        if step > minstep:
            reward -= 1000
        
        total_reward += reward
        
        if done or step > minstep:
            print(step)
            if step < minstep:
                minstep = step + 100
            break
    
    if total_reward > mx_rewards:
        mx_rewards = total_reward
        best_net.load_state_dict(policy_net.state_dict())
    
    loss = learn()
    
    memory = []
    
    rewards.append(total_reward)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Loss: {loss}")

# Plotting rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Policy Gradient Training Performance')
plt.show()

# Re-initialize the environment for visualization
env = gym.make("MountainCarContinuous-v0", render_mode="human")

# Function to play the environment
def play_environment(env, policy_net, episodes=1):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            if step > 500:
                break
            
            env.render()
            action, _ = select_action(state, policy_net)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            step += 1
            
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# Play the environment after training
play_environment(env, best_net)

# Close the environment
env.close()

torch.save(best_net.state_dict(), "best3.pt")