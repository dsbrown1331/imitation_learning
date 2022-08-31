import gym
import pygame
from teleop import play
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
env = gym.make("MountainCar-v0",render_mode='single_rgb_array') 
demos = play(env, keys_to_action=mapping)