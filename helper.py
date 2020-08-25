import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import copy
import torch
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
BETA_DECAY = 10000
BETA_START = 0.4
BETA_END = 1.0

class PriortizedReplayMemory(object):
    """
    class which implements priortized experience replay, maintains
    the following variables:
    - memory: stores transitions
    - probs: stores the unnormalized probabilities (exponentiated by alpha)
    - position: position in memory array
    - prob_sum: sum of probabilities
    - max_weight: max of importance sampling weights
    - alpha, beta: hyperparameters (see paper for details)
    """

    def __init__(self, capacity):
        """
        initialize all the class variables
        """
        self.capacity = capacity
        self.memory = []
        self.probs = []
        self.position = 0
        self.prob_sum = 0
        self.max_weight = 0
        self.alpha = 0.5
        self.beta = 0.4

    def push(self, *args):
        """
        :param args: fields of the transition named tuple
        pushes the transition onto the memory and
        initializes the probabilities of the transitions and importance sampling weights
        """
        if len(self.memory) == self.capacity:
          self.prob_sum -= self.probs[self.position]
        else:
          self.memory.append(None)
          self.probs.append(0)

        if len(self.memory) == 1:
          self.probs[self.position] = 1.
        else:
          self.probs[self.position] = max(self.probs)

        self.prob_sum += self.probs[self.position]
        candidate_weight = 1/(self.probs[self.position])**self.beta
        if candidate_weight > self.max_weight:
          self.max_weight = candidate_weight

        self.memory[self.position] = (Transition(*args), candidate_weight, self.position)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        :param batch_size: the batch_size number of samples to be sampled
        :returns: the sampled batch
        """
        self.memory.append(1)
        self.probs.append(0.0)
        prob_distribution = np.divide(self.probs, self.prob_sum)
        sample_batch = list(np.random.choice(self.memory, size=batch_size, replace=False, p=prob_distribution))
        self.memory.pop() 
        self.probs.pop()
        return sample_batch

    def update(self, positions, TD_errors):
        """
        :param positions: the positions which were sampled
        :param TD_errors: corresponding td errors calculated
        updates the probabilities and IS weights of the sampled transitions in the memory
        and recalculates the max weight
        """
        for i, position in enumerate(positions):
            self.prob_sum -= self.probs[position]
            self.probs[position] = TD_errors[i]**self.alpha
            self.prob_sum += self.probs[position]
            candidate_weight = 1 / (self.probs[position])**self.beta
            self.memory[position] = (self.memory[position][0], candidate_weight, position)

        self.max_weight = max(self.weights)

    def decay_beta(self, steps_done):
        """
        :param steps_done: number of time steps done
        decays the beta hyperpararmeter
        """
        if steps_done <= BETA_DECAY:
            self.beta = BETA_START + ((BETA_END - BETA_START) * steps_done) / BETA_DECAY
        else:
            self.beta = BETA_END

    def __len__(self):
        return len(self.memory)

def plot_durations(episode_durations):
    """
    :param episode_durations: array containing the durations of each episode
    plots the episode vs. duration graph alongwith 100 episode moving average
    """
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

def save_model(model, optimizer, path):
    """
    :param model: the pytorch DQN model
    :param optimizer: optimizer
    :param path: save the checkpoint at path
    """
    weights = copy.deepcopy(model.state_dict())
    optimizer_weights = copy.deepcopy(optimizer.state_dict())
    torch.save({
              'model_state_dict': weights,
              'optimizer_state_dict': optimizer_weights,
              }, path)
  
def load_model(model1, model2, optimizer, path):
    """
    :param model1: policy DQN network
    :param model2: target DQN network
    :param optimizer: optimizer
    :path: load from path
    """
    chkpoint = torch.load(path)
    model1.load_state_dict(chkpoint['model_state_dict'])
    model2.load_state_dict(model1.state_dict())
    optimizer.load_state_dict(chkpoint['optimizer_state_dict'])

