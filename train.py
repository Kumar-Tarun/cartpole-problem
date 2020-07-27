import gym
import random
import math
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from itertools import count
from PIL import Image
from helper import PriortizedReplayMemory, plot_durations, save_model, load_model, Transition
from tqdm import tqdm
from model import DQN

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
RESULT_UPDATE = 25

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CartPoleAgent(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped
        self.resize = T.Compose([T.ToPILImage(), 
                      T.Resize(40, interpolation=Image.CUBIC),
                      T.ToTensor()])
        self.env.reset()
        init_screen = self.get_screen()
        self.env.reset()
        _, _, screen_height, screen_width = init_screen.shape

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n

        self.policy_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net = DQN(screen_height, screen_width, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.0001)
        self.memory = PriortizedReplayMemory(10000)

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        
        return self.resize(screen).unsqueeze(0).to(device)

    def select_action(self, state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        weights = [t[1] for t in transitions]
        positions = [t[2] for t in transitions]
        transitions = [t[0] for t in transitions]
        max_weight = self.memory.max_weight

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        # DDQN: separate action selection and evaluation
        with torch.no_grad():
          new_actions = self.policy_net(non_final_next_states).max(1)[1].view(-1, 1)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, new_actions).detach().view(-1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        # calculate TD error

        TD_error = (expected_state_action_values.unsqueeze(1) - state_action_values).detach().view(-1).cpu().numpy()
        TD_error = np.abs(TD_error)

        # calculate importance sampling weights
        IS_weights = np.divide(weights, max_weight)
        # multiply the weights to the loss function to inject into the weight update
        square_indices = np.where(TD_error <= 1)[0]
        for i, w in enumerate(IS_weights):
          if i in square_indices:
            factor = torch.tensor(np.sqrt(w), device=device, dtype=torch.float32)
          else:
            factor = torch.tensor(w, device=device, dtype=torch.float32)

          state_action_values[i] *= factor
          expected_state_action_values[i] *= factor

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # update priorities of the replayed transitions
        self.memory.update(positions, TD_error)

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def train(self, num_episodes=300, use_prev=None, save_path=None):
        episode_durations = []
        if use_prev is not None:
            load_model(self.policy_net, self.target_net, self.optimizer, use_prev)
        total_time_steps = 0
        for i_episode in tqdm(range(num_episodes)):
            # Initialize the environment and state
            self.env.reset()
            screen_1 = self.get_screen()
            screen_2 = self.get_screen()
            screen_3 = self.get_screen()
            screen_4 = self.get_screen()
            # state representation is a stack of previous 4 frames
            state = torch.stack([screen_4, screen_3, screen_2, screen_1], dim=1).view(1, -1, *screen_1.shape[2:])
            for t in count():
                # Select and perform an action
                total_time_steps += 1
                self.memory.decay_beta(total_time_steps)
                action = self.select_action(state, total_time_steps)
                _, reward, done, _ = self.env.step(action.item())
                # Observe new state
                last_state = state
                current_screen = self.get_screen()
                if not done:
                  next_state = torch.roll(state, 3, 1)
                  next_state[:, 0:3, :, :] = current_screen
                else:
                  next_state = None
                  reward = -30.0

                # Store the transition in memory
                reward = torch.tensor([reward], device=device)
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    episode_durations.append(t + 1)
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if i_episode != 0 and i_episode % RESULT_UPDATE == 0:
                print(f'After {i_episode} episodes, total reward: {total_time_steps}')
                print(f'Average reward per episode: {total_time_steps/i_episode}')

        plot_durations(episode_durations)
        if save_path is not None:
          save_model(self.policy_net, self.optimizer, save_path)
        print('Complete')
        print('Total time steps: ', total_time_steps)
        self.env.render()
        self.env.close()

if __name__ == '__main__':
    agent = CartPoleAgent()
    agent.train(num_episodes=100, save_path='DDQN_PR.tar')
