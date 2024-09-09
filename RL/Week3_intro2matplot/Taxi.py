import gymnasium as gym
import numpy as np

# Step 1: Create the FrozenLake environment.
# The 'is_slippery' parameter determines whether the lake is slippery (true) or not (false).
env = gym.make('Taxi-v3', render_mode="rgb-arrayar") #render_mode="human" is to render the traning process
#env = gym.make('FrozenLake-v1', is_slippery=True, render_mode="rgb_array") #render_mode="rgb_array" is not to show traning process(faster process)

# Step 2: Initialize the Q-Table with zeros.
# The Q-Table is a matrix where we have a row for each state and a column for each action.
# The number of states and actions are determined by the environment.
n_states = env.observation_space.n
n_actions = env.action_space.n
Q = np.zeros((n_states, n_actions))

# Step 3: Define the hyperparameters for the Q-Learning algorithm.
alpha = 0.25  # Learning rate - determines how much new information overrides the old information.
gamma = 0.99  # Discount factor - determines the importance of future rewards.
epsilon = 1.0  # Exploration rate - controls the exploration-exploitation trade-off.
epsilon_decay = 0.995  # Decay rate for exploration - reduces epsilon after each episode.
min_epsilon = 0.01  # Minimum exploration rate.
n_episodes = 2000  # Number of episodes to train on.
max_steps = 100  # Maximum steps per episode.

# List to store the total rewards per episode
rewards = []

# Data for plot
Q_nState = []
Epsilon_array = []
avg_rewards = []

# Step 4: Implement the Q-Learning algorithm.
# Training loop
for episode in range(n_episodes):
    state, _ = env.reset()  # Reset the environment to the initial state at the beginning of each episode.
    total_reward = 0  # Initialize total reward for this episode.
    
    Q10State = np.zeros((10,))

    for i in range(max_steps):
        # Exploration-exploitation trade-off
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore: select a random action.
        else:
            action = np.argmax(Q[state, :])  # Exploit: select the action with the highest Q-value for the current state.

        # Take the selected action and observe the outcome.
        next_state, reward, done, _, _ = env.step(action)

        # Update the Q-Table using the Q-Learning update rule.
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        
        Q10State = np.fmax(np.array(Q[:10, :].copy()).max(axis=1), Q10State)

        # Update the current state to the next state.
        state = next_state
        total_reward += reward  # Accumulate the reward for this episode.

        if done:  # If the episode ends, break the loop.
            
            if n_episodes - episode <= 100:
                avg_rewards.append(total_reward/(i+1))
            
            break

    # Decay epsilon to reduce exploration over time.
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    rewards.append(total_reward)  # Store the total reward for this episode.
    
    Q_nState.append(Q10State)
    Epsilon_array.append(epsilon)

    if (episode + 1) % 100 == 0:  # Print progress every 100 episodes.
        print(f'Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}')

print("Training finished.\n")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
x = [i for i in range(n_episodes)]

axes[0][0].plot(x, Q_nState)
axes[0][0].legend([f"state-{i}" for i in range(np.array(Q_nState).shape[0])], loc='upper right')
axes[0][0].set_title("Max Q-Value for first 10 state")
axes[0][0].set_ylabel("Max-Q")
axes[0][0].set_xlabel("Episodes")

axes[0][1].plot(x, rewards)
axes[0][1].set_title("Total rewards")
axes[0][1].set_ylabel("Total reward")
axes[0][1].set_xlabel("Episodes")

axes[1][0].plot(x, Epsilon_array)
axes[1][0].set_title("Epsilon Decay")
axes[1][0].set_ylabel("Epsilon")
axes[1][0].set_xlabel("Episodes")

axes[1][1].plot([i for i in range(n_episodes-100, n_episodes)], avg_rewards)
axes[1][1].set_title("Average rewards last 10 Episodes")
axes[1][1].set_ylabel("Average reward")
axes[1][1].set_xlabel("Episodes")

plt.show()

# Step 5: Evaluate the learned policy.
# Run the trained agent in the environment to see how well it performs.
n_eval_episodes = 100  # Number of evaluation episodes.
total_rewards = 0  # Initialize total rewards for evaluation.

env = gym.make('Taxi-v3', render_mode="human")

for _ in range(n_eval_episodes):
    state, _ = env.reset()  # Reset the environment to the initial state.
    episode_reward = 0  # Initialize reward for this episode.

    for _ in range(max_steps):
        action = np.argmax(Q[state, :])  # Select the action with the highest Q-value for the current state.
        next_state, reward, done, _, _ = env.step(action)
        state = next_state  # Update the current state to the next state.
        episode_reward += reward  # Accumulate the reward for this episode.

        if done:  # If the episode ends, break the loop.
            break

    total_rewards += episode_reward  # Accumulate the total rewards over all episodes.

average_reward = total_rewards / n_eval_episodes  # Calculate the average reward over the evaluation episodes.
print(f'Average Reward over {n_eval_episodes} episodes: {average_reward}')
