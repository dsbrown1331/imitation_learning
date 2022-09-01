import gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


device = torch.device('cpu')


def collect_human_demos(num_demos):
    mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
    env = gym.make("MountainCar-v0",render_mode='single_rgb_array') 
    demos = collect_demos(env, keys_to_action=mapping, num_demos=num_demos, noop=1)
    return demos


def torchify_demos(sas_pairs):
    states = []
    actions = []
    next_states = []
    for s,a, s2 in sas_pairs:
        states.append(s)
        actions.append(a)
        next_states.append(s2)

    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)

    obs_torch = torch.from_numpy(np.array(states)).float().to(device)
    obs2_torch = torch.from_numpy(np.array(next_states)).float().to(device)
    acs_torch = torch.from_numpy(np.array(actions)).to(device)

    return obs_torch, acs_torch, obs2_torch


def train_policy(obs, acs, nn_policy, num_train_iters):
    pi_optimizer = Adam(nn_policy.parameters(), lr=0.1)
    #action space is discrete so our policy just needs to classify which action to take
    #we typically train classifiers using a cross entropy loss
    loss_criterion = nn.CrossEntropyLoss()
    
    # run BC using all the demos in one giant batch
    for i in range(num_train_iters):
        #zero out automatic differentiation from last time
        pi_optimizer.zero_grad()
        #run each state in batch through policy to get predicted logits for classifying action
        pred_action_logits = nn_policy(obs)
        #now compute loss by comparing what the policy thinks it should do with what the demonstrator didd
        loss = loss_criterion(pred_action_logits, acs) 
        print("iteration", i, "bc loss", loss)
        #back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        #perform update on policy parameters
        pi_optimizer.step()



class PolicyNetwork(nn.Module):
    '''
        Simple neural network with two layers that maps a 2-d state to a prediction
        over which of the three discrete actions should be taken.
        The three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super().__init__()

        #This layer has 2 inputs corresponding to car position and velocity
        self.fc1 = nn.Linear(2, 8)  
        #This layer has three outputs corresponding to each of the three discrete actions
        self.fc2 = nn.Linear(8, 3)  



    def forward(self, x):
        #this method performs a forward pass through the network, applying a non-linearity (ReLU) on the 
        #outputs of the first layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    

#evaluate learned policy
def evaluate_policy(pi, num_evals, human_render=True):
    if human_render:
        env = gym.make("MountainCar-v0",render_mode='human') 
    else:
        env = gym.make("MountainCar-v0") 

    policy_returns = []
    for i in range(num_evals):
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            #take the action that the network assigns the highest logit value to
            #Note that first we convert from numpy to tensor and then we get the value of the 
            #argmax using .item() and feed that into the environment
            action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
            # print(action)
            obs, rew, done, info = env.step(action)
            total_reward += rew
        print("reward for evaluation", i, total_reward)
        policy_returns.append(total_reward)

    print("average policy return", np.mean(policy_returns))
    print("min policy return", np.min(policy_returns))
    print("max policy return", np.max(policy_returns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")

    args = parser.parse_args()

    #collect human demos
    demos = collect_human_demos(args.num_demos)

    #process demos
    obs, acs, _ = torchify_demos(demos)

    #train policy
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    #evaluate learned policy
    evaluate_policy(pi, args.num_evals)

