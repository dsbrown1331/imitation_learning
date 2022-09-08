# Homework 1 for CS 6960 Human-AI Alignment


### Installation Ubuntu
First it is recommended that you install anaconda: <https://www.anaconda.com/products/distribution> a popular python distribution and software management platform.

Next, git clone this repository.

Next navigate to the reponsitory
```
cd imitation_learning
```
then install the dependencies by running
```
conda env create -f environment.yml
```
This should install [Open AI Gym](https://www.gymlibrary.dev/) and [PyTorch](https://pytorch.org/get-started/locally/). Note that it will install the PyTorch cpu version so you don't need a GPU for these experiments and it is unlikely that having a GPU will help since we are dealing with low-dimensional state spaces and small neural networks.

Before you can run any of the code in this repo you need the libraries installed (what you just did with the above command, and you need to activate the environment
```
conda activate imitation_learning
```

To test the code run
```
python test_gym.py
```
You should see a visualization of a car moving back and forth at the bottom of a valley.

### Installation other platforms
Currently still debugging this, but try 
```
conda env create -f environment_basic.yml
```
then 
```
conda activate imitation_learning
```
then
```
pip install gym[classic_control,other]==0.25.2
```
and hopefully that will install everything needed in a more platform independent way.


### PyTorch Primer
If you have never used PyTorch before, I'd recommend going through the 60-minute blitz tutorial: <https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html>


### OpenAI Gym Primer
If you've never used OpenAI gym before, I'd recommmend reading through the beginning of this tutorial: <https://blog.paperspace.com/getting-started-with-openai-gym/>. You can stop when you get to the section on Wrappers.



Look at the code in `test_gym.py`. This code runs one episode of MountainCar (200 time steps) using a random policy that samples uniformly from the action space.

In MountainCar there are three actions: 0 (accelerate left), 1 (don't accelerate), 2 (accelerate right). You can read more about MountainCar here: <https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py>. The state-space is 2-dimensional: given a state s, s[0] is the x-position of the car and s[1] is the car's velocity.

The goal of MountainCar is to have the car drive to the flag on the top of the hill to the right. The car gets -1 reward for every step and gets 200 steps to try and get out of the hill. Because the reward is -1 per timestep the optimal policy is to get out of the valley in as few timesteps as possible.


<strong>You will need to type up your responses to the following parts of the homework and submit your responses and code via Canvas to get credit for this homework assignment. You are encouraged to talk about the homework with other students and share resources, but please do not share or copy code.</strong>.

## Part 1:

Run the following code again
```
python test_gym.py
```
What do you notice happening? MountainCar is a classic RL problem known for being a difficult exploration problem. Why do you think a trial and error method like RL would struggle with MountainCar? Hint: RL uses the reward as a signal for what is good and bad so why might that be a problem in MountainCar?

## Part 2:
You will now learn how to solve the MountainCar task by driving the car yourself.
Run
```
python mountain_car_play.py
```
Use the arrow keys to control the car by accelerating left and right. Note that if you run out of time you get a reward of -200.0 and the car will reset to the bottom of the hill. If you get to the flag in less than 200 steps you can get a higher score. Once you reach the flag the environment will restart at the bottom of the hill.

Keep practicing until you can reliably get out of the valley. You can see your score for each epsiode output on the command line. 
Experiment with different strategies: e.g.
1. Going right, then left, then right all the way up the hill
2. Going left, then right all the way up the hill.
3. Left, right, left, right up the hill.
Which strategy do you like best? What is the best score you can get as a human demonstrator?

## Part 3: 

Now we will teach a simple behavioral cloning (BC) agent to drive itself out of the valley.

First take a look at `mountain_car_bc.py` and try to get a basic understanding of what is happening. By default this code will collect a single demonstration, then parse the states and actions into tensors for easy input to a PyTorch neural network. It then trains a policy to imitate the demonstrations and evaluates the policy by testing it on different initial states. The commmand line will output the average, min, and max total rewards.

Try it out by running
```
python mountain_car_bc.py
```
Try to give a good demonstration. Then watch what the agent has learned. Does it do a good job imitating? Does it ever get stuck? What is the average, min, and max?

## Part 4

Let's give more than one demonstration. Run the following to give 5 good demonstrations. If you mess up during one demo, feel free to restart until you give 5 good demos and try to keep to a consistent strategy.
```
python mountain_car_bc.py --num_demos 5
```
Report the average min and max returns. Did it perform any better? Why or why not? Does the agent copy the strategy you used? 

## Part 4
What do you think will happen if we give good and bad demonstrations?
You will now give two demonstrations. For the first one, just press the right arrow key for the entire episode until it restarts. Then for the second demo, give a good demonstration that quickly gets out of the valley.
```
python mountain_car_bc.py --num_demos 2
```
What does the policy learn? Why might bad demonstrations be a problem? Briefly describe one potential idea for making BC robust to bad demonstrations.

## Part 5
Let's teach the agent to do something else. Give 5 demonstrations that drive back and forth going up and down the sides of the valley, but without going out and reaching the flag.  Run BC and see if the agent learns this new skill:
```
python mountain_car_bc.py --num_demos 5
```
Were you able to teach the agent to oscillate without ending the episode early?

## Part 6
Describe what changes you would need to make to the code in `mountain_car_bc.py` to implement BCO. Answer this question before starting Part 7 and before looking at `mountain_car_bco.py`.

## Part 7
Implement and test BC(0) by training an inverse dynamics model. You can find starter code in `mountain_car_bco.py`. Report how well it works and what you tried to get it to work.


