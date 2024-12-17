import gymnasium as gym
from fbot_robot_learning.doris_gazebo_env import DorisGazeboEnv

gym.register('gym_fbot/DorisGazebo-v0', DorisGazeboEnv)