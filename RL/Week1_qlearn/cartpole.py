import gym
from tqdm import tqdm
import numpy as np

n=1000

env = gym.make("CartPole-v1", render_mode="human")
env.action_space.seed(82)

observation, info = env.reset(seed=82)

for _ in tqdm(range(n)):
    action = env.action_space.sample()
    observation,reward, terminated, truncated, info = env.step(action)
    print("info : ",info);
    
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()