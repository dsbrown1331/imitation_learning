import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from mountain_car_bc import collect_human_demos, torchify_demos, train_policy, PolicyNetwork, evaluate_policy


device = torch.device('cpu')


def collect_random_interaction_data(num_iters):
    state_next_state = []
    actions = []
    env = gym.make('MountainCar-v0')
    for _ in range(num_iters):
        obs = env.reset()

        done = False
        while not done:
            a = env.action_space.sample()
            next_obs, reward, done, info = env.step(a)
            state_next_state.append(np.concatenate((obs,next_obs), axis=0))
            actions.append(a)
            obs = next_obs
    env.close()

    return np.array(state_next_state), np.array(actions)




class InvDynamicsNetwork(nn.Module):
    '''
        Neural network with that maps (s,s') state to a prediction
        over which of the three discrete actions was taken.
        The network should have three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super().__init__()

        #This network should take in 4 inputs corresponding to car position and velocity in s and s'
        # and have 3 outputs corresponding to the three different actions

        #################
        #TODO:
        #################

    def forward(self, x):
        #this method performs a forward pass through the network
        ###############
        #TODO:
        ###############
        return x
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")

    args = parser.parse_args()


    #collect random interaction data
    num_interactions = 5
    s_s2, acs = collect_random_interaction_data(num_interactions)
    #put the data into tensors for feeding into torch
    s_s2_torch = torch.from_numpy(np.array(s_s2)).float().to(device)
    a_torch = torch.from_numpy(np.array(acs)).long().to(device)


    #initialize inverse dynamics model
    inv_dyn = InvDynamicsNetwork()  #TODO: need to fill in the blanks in this method
    ##################
    #TODO: Train the inverse dyanmics model, no need to be fancy you can do it in one full batch via gradient descent if you like
    ##################



    #collect human demos
    demos = collect_human_demos(args.num_demos)

    #process demos
    obs, acs_true, obs2 = torchify_demos(demos)

    #predict actions
    state_trans = torch.cat((obs, obs2), dim = 1)
    outputs = inv_dyn(state_trans)
    _, acs = torch.max(outputs, 1)

    #train policy using predicted actions for states this should use your train_policy function from your BC implementation
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    #evaluate learned policy
    evaluate_policy(pi, args.num_evals)

