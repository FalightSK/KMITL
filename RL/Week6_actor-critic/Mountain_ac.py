import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x
    
class Discrete_actor(nn.Module):
    def __init__(self, state_size, output_size, hidden_size):
        super(Discrete_actor, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return F.softmax(x)
    
class Continue_actor(nn.Module):
    def __init__(self, state_size, output_size, hidden_size):
        super(Continue_actor, self).__init__()
        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.l_logvar = nn.Linear(hidden_size, 1)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        logvar = self.l_logvar(x)
        x = F.relu(self.linear2(x))
        logvar = torch.clamp(logvar, -20, 2)  # Clipping log variance
        sigma = torch.exp(logvar)
        return F.tanh(x), sigma
    
EPISODES = 250
GAMMA = 0.99
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
BATCH_SIZE = 128
MEMORY_SIZE = BATCH_SIZE * 1
MAX_STEP = 5000

torch.manual_seed(42)

env = gym.make("MountainCarContinuous-v0")
state_size = env.observation_space.shape[0]
action_size = 1

print(state_size, action_size)

policy_net = Continue_actor(state_size, action_size, 128).to(device)
critic_net = ValueNetwork(state_size, 128).to(device)

policy_best = Continue_actor(state_size, action_size, 128).to(device)
critic_best = ValueNetwork(state_size, 128).to(device)

# Critic Q value
target_critic = ValueNetwork(state_size, 128).to(device)
target_critic.load_state_dict(critic_net.state_dict())
target_critic.eval()

optimizer_actor = optim.Adam(policy_net.parameters(), lr=LR_ACTOR)
optimizer_critic = optim.Adam(critic_net.parameters(), lr=LR_CRITIC)
memory = deque(maxlen=MEMORY_SIZE)

def select_action(observation, net):
    state = torch.tensor(observation, dtype=torch.float32).to(device)
    state = state.unsqueeze(0)
    
    probs, sigma = net(state)
    m = torch.distributions.Normal(loc=probs, scale=sigma)
    action = m.sample()
    logprob = m.log_prob(action).sum()
    
    return [action.item()], logprob

def learn():
    if len(memory) < BATCH_SIZE:
        return
    # minibatch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones, log_probs = zip(*memory)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)
    log_probs = torch.stack(log_probs).unsqueeze(1).to(device)

    #Train Critic
    optimizer_critic.zero_grad()
    
    critic_value = critic_net(states)
    next_critic_value = target_critic(next_states)

    target_v = rewards + GAMMA*next_critic_value*(1-dones)
    
    critic_loss = nn.MSELoss()(critic_value,target_v)
   
    critic_loss.backward()
    optimizer_critic.step()

    #Train actor
    optimizer_actor.zero_grad()

    critic_value = target_critic(states)
    next_critic_value = target_critic(next_states)
    target_v = rewards + GAMMA*next_critic_value*(1-dones)
    delta = target_v - critic_value

    actor_loss = -log_probs * delta

    actor_loss = actor_loss.sum()
    
    actor_loss.backward()
    optimizer_actor.step()

def update_target_net():
    target_critic.load_state_dict(critic_net.state_dict())
    
# Training loop
rewards = []
global_step = 0

mx_reward = -1e9

for episode in range(EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        action, log_prob = select_action(state, policy_net)
        next_state, reward, done, _, _ = env.step(action)
        
        memory.append((state, action, reward, next_state, done, log_prob))
        state = next_state
        
        step += 1
        if step > MAX_STEP:
            done = True
        
        total_reward += reward

        global_step += 1
        if global_step % BATCH_SIZE == 0:
            learn()
            memory = []

    if mx_reward < total_reward:
        mx_reward = total_reward
        policy_best.load_state_dict(policy_net.state_dict())
        critic_best.load_state_dict(critic_net.state_dict())
    
    rewards.append(total_reward)

    if episode % 10 == 0:
        update_target_net()
        print(f"Episode: {episode}, Reward: {total_reward}")

# Plotting rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Actor Critic Training Performance')
plt.show()

# Re-initialize the environment for visualization
env = gym.make("MountainCarContinuous-v0", render_mode="human")

# Function to play the environment
def play_environment(env, net, episodes=5):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            action, _ = select_action(state, net)  # Select action with epsilon=0 (greedy)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        print(f"Episode: {episode+1}, Total Reward: {total_reward}")

# Play the environment after training
play_environment(env, policy_best)

# Close the environment
env.close()


