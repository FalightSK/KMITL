import gym
from gym.utils import play
env = play.play(gym.make('MountainCar-v0', render_mode='rgb_array').env, zoom=1,  keys_to_action={"2":2, "1":0}, noop=1)