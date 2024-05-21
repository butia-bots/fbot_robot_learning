import gymnasium as gym
from gym_fbot.doris_gazebo_env import DorisGazeboEnv

gym.register('gym_fbot/DorisGazebo-v0', DorisGazeboEnv)