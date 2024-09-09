import gym
from tqdm import tqdm
import numpy as np

def discretize_state(state, bins):
    discretized = []
    for i in range(len(state)):
        discretized.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(discretized)

def create_bins(num_bins, low, high):
    return [np.linspace(low[i], high[i], num_bins + 1)[1:-1] for i in range(len(low))]

env = gym.make("CartPole-v1")
env.action_space.seed(82)
np.random.seed(82)

n_episodes = 5000
max_steps = 200
learning_rate = 0.055
discount_factor = 0.895
epsilon = 1.2
epsilon_min = 0.01
epsilon_decay = 0.998
num_bins = 24

obs_space_low = env.observation_space.low
obs_space_high = env.observation_space.high

bins = create_bins(num_bins, obs_space_low, obs_space_high)

q_table = np.zeros((num_bins, num_bins, num_bins, num_bins, env.action_space.n))

rewards = []

for episode in tqdm(range(n_episodes)):
    observation, info = env.reset(seed=82)
    state = discretize_state(observation, bins)

    for step in range(max_steps):
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_observation, reward, terminated, truncated, info = env.step(action)
        next_state = discretize_state(next_observation, bins)

        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + discount_factor * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += learning_rate * td_error

        state = next_state

        if terminated or truncated:
            break
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

env.close()

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=82)
state = discretize_state(observation, bins)

for step in range(2000):
    action = np.argmax(q_table[state])
    observation, reward, terminated, truncated, info = env.step(action)
    state = discretize_state(observation, bins)
    
    if terminated or truncated:
        print("error")
        break

env.close()
