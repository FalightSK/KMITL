import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPISODES = 5000
GAMMA = 0.95
LR = 0.005
BATCH_SIZE = 512
MEMORY_SIZE = 100000


# Define the neural network
class PolicyGradientNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyGradientNN, self).__init__()
        self.fc1 = nn.Linear(state_size, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        return self.fc3(x)

# Initialize the environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(state_size, action_size)

# Initialize the Policy Gradient
policy_net = PolicyGradientNN(state_size, action_size).to(device)
best_net = PolicyGradientNN(state_size, action_size).to(device)
best_net.load_state_dict(policy_net.state_dict())

# Check if the models are on the GPU
print(f"Policy Net on device: {next(policy_net.parameters()).device}")

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = []


def select_action(observation, net):
    state = torch.tensor([observation]).to(device)

    m= nn.Softmax(dim=1)
    probabilities = m(net(state))

    action_prob = torch.distributions.Categorical(probabilities)
    action = action_prob.sample()
    log_probs = action_prob.log_prob(action)

    return action.item(), log_probs

def learn():
   
    states, actions, rewards, next_states, dones, log_prob_ = zip(*memory)
    
    G = np.zeros(shape=(len(memory)))
    print(G.shape)
    for t in range(len(memory)):
        G_sum = 0
        discount = 1
        for k in range(t, len(memory)):
            G_sum += rewards[k] * discount
            discount *= GAMMA
        G[t] = G_sum

    G = torch.tensor(G, dtype=torch.float32).to(device)

    loss = 0
    for g, logprob in zip(G, log_prob_):
        loss += -g * logprob
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training loop
rewards = []
mx_reward = -1e9

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    for i in range(15000):
        
        if done:
            break
        
        action, log_prob = select_action(state, policy_net)
        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done, log_prob))
        state = next_state

        total_reward += reward

    
    learn()
    # reset memory
    memory = []
    
    if mx_reward < total_reward:
        mx_reward = total_reward
        best_net.load_state_dict(policy_net.state_dict())
    
    if total_reward >= 10000:
        break
    
    rewards.append(total_reward)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}")

# Plotting rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN Training Performance')
plt.show()

# Re-initialize the environment for visualization
env = gym.make("CartPole-v1", render_mode="human")

# Function to play the environment
def play_environment(env, policy_net, episodes=1):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            action, _ = select_action(state, policy_net)  # Select action with epsilon=0 (greedy)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# Play the environment after training
play_environment(env, best_net)

# Close the environment
env.close()