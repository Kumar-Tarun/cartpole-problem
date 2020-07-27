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

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.probs = []
        self.position = 0
        self.prob_sum = 0
        self.max_weight = 0
        self.weights = []
        self.alpha = 0.5
        self.beta = 0.4

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) == self.capacity:
          self.prob_sum -= self.probs[self.position]
        else:
          self.memory.append(None)
          self.probs.append(0)
          self.weights.append(0)  

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
        self.memory.append(1)
        self.probs.append(0.0)
        prob_distribution = np.divide(self.probs, self.prob_sum)
        sample_batch = list(np.random.choice(self.memory, size=batch_size, replace=False, p=prob_distribution))
        self.memory.pop() 
        self.probs.pop()
        return sample_batch

    def update(self, positions, TD_errors):
        for i, position in enumerate(positions):
            self.prob_sum -= self.probs[position]
            self.probs[position] = TD_errors[i]**self.alpha
            self.prob_sum += self.probs[position]
            candidate_weight = 1 / (self.probs[position])**self.beta
            self.memory[position] = (self.memory[position][0], candidate_weight, position)

        self.max_weight = max(self.weights)

    def decay_beta(self, steps_done):
        if steps_done <= BETA_DECAY:
            self.beta = BETA_START + ((BETA_END - BETA_START) * steps_done) / BETA_DECAY
        else:
            self.beta = BETA_END

    def __len__(self):
        return len(self.memory)

def plot_durations(episode_durations):
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
    weights = copy.deepcopy(model.state_dict())
    optimizer_weights = copy.deepcopy(optimizer.state_dict())
    torch.save({
              'model_state_dict': weights,
              'optimizer_state_dict': optimizer_weights,
              }, path)
  
def load_model(model1, model2, optimizer, path):
    chkpoint = torch.load(path)
    model1.load_state_dict(chkpoint['model_state_dict'])
    model2.load_state_dict(model1.state_dict())
    optimizer.load_state_dict(chkpoint['optimizer_state_dict'])

