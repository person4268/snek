import random
import snake
import snakerenderer
import gymnasium as gym
import numpy as np
import typing
from itertools import count
import pygame
import math
from gymnasium import spaces
from gymnasium.envs.registration import register
import hyperparams as hyp

from collections import defaultdict, deque, namedtuple

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Hack: pygame Surface can't be pickled so we can't store a reference to it inside the class
renderer = snakerenderer.SnakeRenderer(1)

# I think gym's api is a bit simpler to deal with than torchrl.env.GymLikeEnv but you can use that if you want
class SnakeEnv(gym.Env):
  __metadata__ = {'render.modes': ['human']}

  def __init__(self):
    super(SnakeEnv, self).__init__()
    self.snake = snake.SnakeGame()
    self.sample_random_amt = 0
    self.sample_model_amt = 0
    self.last_rew = 0
    self.timer = pygame.time.Clock()
    # Right, left, up, down
    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Box(low=0, high=1,
                                        shape=(1, snake.GRID[0], snake.GRID[1]), dtype=np.float16)
    self.reward_range = (-float(hyp.PEN_DEATH), math.inf)

  def calc_food_dist(self, food: snake.Point) -> float:
    return math.sqrt((food.x - self.snake.head.x)**2 + (food.y - self.snake.head.y)**2)

  def step(self, action: int):
    self.snake.change_direction(snake.Direction(action))
    game_over, food_collected, score = self.snake.tick()
    timeout = self.snake.ticks_alive > hyp.MAX_TICKS_ALIVE
    reward = 0
    reward += int(food_collected) * hyp.REW_FOOD
    reward += 1 / (self.calc_food_dist(self.snake.food) *  hyp.FOOD_DIST_AWARD + 1)
    reward += 1 / (self.calc_food_dist(self.snake.food2) * hyp.FOOD_DIST_AWARD + 1)
    reward += 1 / (self.calc_food_dist(self.snake.food3) * hyp.FOOD_DIST_AWARD + 1)
    reward += hyp.REW_ALIVE
    reward -= hyp.PEN_DEATH * int(game_over)
    reward -= hyp.PEN_TIMEOUT * int(timeout)
    self.last_rew = reward
    # self.timer.tick(60)
    self.render()
    # obs (scaled to 1), reward, terminated, truncated, info
    return np.expand_dims(self.snake.get_state() / 3, axis=0), reward, game_over, timeout, {}

  def reset(self, seed: int | None = None, options = None,):
    self.snake.reset()
    self.sample_random_amt = 0
    self.sample_model_amt = 0
    return np.expand_dims(self.snake.get_state() / 3, axis=0), {}
  
  def render(self, mode='human'):
    amt = 0
    if self.sample_model_amt + self.sample_random_amt != 0:
      amt = self.sample_random_amt / (self.sample_random_amt + self.sample_model_amt)
    renderer.render([self.snake], f" %R: {amt:.2f} REW: {self.last_rew:.2f}")
    pass

register(
     id="snake",
     entry_point="main:SnakeEnv",
     max_episode_steps=None,
)

env = gym.make("snake")

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=2, device=device)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=4, stride=1, padding=2, device=device)
        self.layer1 = nn.Linear(17 * 17 * 16, 128)
        self.layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE number of transitions sampled from the replay buffer
BATCH_SIZE = 128
# should be a constant between :math:`0` and :math:`1`
# that ensures the sum converges. A lower :math:`\gamma` makes 
# rewards from the uncertain far future less important for our agent 
# than the ones in the near future that it can be fairly confident 
# about. It also encourages agents to collect reward closer in time 
# than equivalent rewards that are temporally far away in the future.
GAMMA = 0.99
# starting value of epsilon
EPS_START = 0.9
# final value of epsilon
EPS_END = 0.05
# rate of exponential decay of epsilon, higher means a slower decay
EPS_DECAY = 2000
# update rate of the target network
TAU = 0.006
# learning rate of the ``AdamW`` optimizer
LR = 1e-4

n_actions = env.action_space.n # type: ignore
print(f"obs space shappe {env.observation_space.shape}")
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)

target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            logits: torch.Tensor = policy_net(state)
            sample = logits.max(1).indices.view(1, 1)
            env.unwrapped.sample_model_amt += 1 # type: ignore
            return sample
    else:
        env.unwrapped.sample_random_amt += 1 # type: ignore
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=True):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
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

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

num_episodes = 500

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()