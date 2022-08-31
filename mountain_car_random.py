import gym
import pygame
from teleop import play
mapping = {"0": 0, "1":1, "2":2}
env = gym.make("MountainCar-v0",render_mode='single_rgb_array') 
demos = play(env, keys_to_action=mapping)