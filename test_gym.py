import gym
env = gym.make('MountainCar-v0')

env.reset()

done = False
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
    
env.close()