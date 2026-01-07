import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import gym
import pygame
from teleop import play
mapping = {(pygame.K_LEFT,): 0,
           (pygame.K_RIGHT,): 2,
           (): 1  # Coast (no keys pressed)
           }
env = gym.make("MountainCar-v0",render_mode='single_rgb_array') 
demos = play(env, keys_to_action=mapping)
