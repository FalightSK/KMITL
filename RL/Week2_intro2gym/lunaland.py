import pygame   
from pygame.event import Event
import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()
action = 0

action_dict = {
    "w": 2, # Up
    "a": 1, # Left
    "d": 3  # Right
}

for _ in range(1000):
    
    # Loop
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            print(event.key)
            if chr(event.key) in list(action_dict.keys()):
                action = action_dict[chr(event.key)]
        elif event.type == pygame.KEYUP:
            action = 0
    
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()