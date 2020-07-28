# Cartpole Problem
* This repository contains code for training an agent on the Cartpole-v0 environment of OpenAI Gym.
* The agent consists of Double DQN with priortized experience replay.

## Implementation Details
* The state representation is stack of 4 cropped and downsampled frames instead of the standard 4-tuple state for cartpole environment.
* The reward on transition to terminal state is taken as negative.
* Since the env is unwrapped, there is no upper limit on number of steps allowed in an episode.
* The hyperparameters for priortized replay memory are taken from the paper itself.

## Results
* The model is trained for 400 episodes.
* The 100 episode moving average is around 100 at the end of training.
![](/images/cartpole-demo.gif) ![](/images/graph.png)

## Usage
* Simply run ```python3 train.py```
* After training the model will be saved in the current directory.
* Number of episodes etc. can be easily modified in the script.

## Credits
* This project is adapted from the PyTorch's [DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) 
