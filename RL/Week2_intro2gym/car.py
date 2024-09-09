import gymnasium as gym
from gym.utils import play

# env = play.play(
#     gym.make("Taxi-v3", render_mode='rgb_array').env, 
#     zoom=1,  
#     keys_to_action={
#         "s":0, # Down 
#         "w":1, # up
#         "d":2, # right
#         "a":3, # left
#         "1":4, # pickup
#         "2":5 # drop
#     }, 
#     noop=4
# )

# env = play.play(
#     gym.make("Acrobot-v1", render_mode='rgb_array').env, 
#     zoom=1,  
#     keys_to_action={
#         "s":0, # -1 Force
#         "w":1, # 0 Force
#         "d":2, # 1 Force
#     }, 
#     noop=1
# )

env = play.play(
    gym.make("CliffWalking-v0", render_mode='rgb_array').env, 
    zoom=1,  
    keys_to_action={
        "w":0, # Up
        "d":1, # Right
        "s":2, # Down,
        "a":3, # Left
    }, 
    noop=0
)