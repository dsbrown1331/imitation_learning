import gym
import argparse
import pygame
from teleop import collect_demos
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
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
    #TODO: you may need to tune the learning rate some
    ##################
    learning_rate = 0.01
    optimizer = Adam(inv_dyn.parameters(), lr=learning_rate)
    #action space is discrete so our policy just needs to classify which action to take
    #we typically train classifiers using a cross entropy loss
    loss_criterion = nn.CrossEntropyLoss()
    
    # train inverse dynamics model in one big batch
    ####################
    #TODO you may need to tune the num_train_iters
    ####################
    num_train_iters = 1000  #number of times to run gradient descent on training data
    for i in range(num_train_iters):
        #zero out automatic differentiation from last time
        optimizer.zero_grad()
        #run each state in batch through policy to get predicted logits for classifying action
        pred_action_logits = inv_dyn(s_s2_torch)
        #now compute loss by comparing what the policy thinks it should do with what the demonstrator didd
        loss = loss_criterion(pred_action_logits, a_torch) 
        print("iteration", i, "bc loss", loss)
        #back propagate the error through the network to figure out how update it to prefer demonstrator actions
        loss.backward()
        #perform update on policy parameters
        optimizer.step()

    #check performance for debugging
    outputs = inv_dyn(s_s2_torch[:10])
    _, predicted = torch.max(outputs, 1)
    print("checking predictions on first 10 actions from random interaction data")
    print("predicted actions", predicted)
    print("actual actions", acs[:10])

    #collect human demos
    demos = collect_human_demos(args.num_demos)

    #process demos
    obs, acs_true, obs2 = torchify_demos(demos)

    #estimate actions
    state_trans = torch.cat((obs, obs2), dim = 1)
    outputs = inv_dyn(state_trans)
    _, acs = torch.max(outputs, 1)
    #check accuracy on demos for debugging
    print("checking predicted demonstrator actions with actual actions from demonstrations")
    print("predicted", acs[:20])
    print("actual", acs_true[:20])

    #train policy using predicted actions for states
    pi = PolicyNetwork()
    train_policy(obs, acs, pi, args.num_bc_iters)

    #evaluate learned policy
    evaluate_policy(pi, args.num_evals)

