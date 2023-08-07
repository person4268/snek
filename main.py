import snake
import snakerenderer
import gymnasium as gym
import numpy as np
import typing
import pygame
import math
from gymnasium import spaces
from gymnasium.envs.registration import register
import hyperparams as hyp

# todo: delete unused
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import OneHotCategorical
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, Actor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

num_iter = 1000

# Hack: pygame Surface can't be pickled so we can't store a reference to it inside the class
renderer = snakerenderer.SnakeRenderer(1)

# I think gym's api is a bit simpler to deal with than torchrl.env.GymLikeEnv but you can use that if you want
class SnakeEnv(gym.Env):
  __metadata__ = {'render.modes': ['human']}

  def __init__(self):
    super(SnakeEnv, self).__init__()
    self.snake = snake.SnakeGame()
    self.timer = pygame.time.Clock()
    # Right, left, up, down
    self.action_space = spaces.Discrete(4)
    self.observation_space = spaces.Box(low=0, high=1,
                                        shape=snake.GRID, dtype=np.float16)
    self.reward_range = (-float(hyp.PEN_DEATH), math.inf)

  def step(self, action: int):
    self.snake.change_direction(snake.Direction(action))
    game_over, food_collected, score = self.snake.tick()
    timeout = self.snake.ticks_alive > hyp.MAX_TICKS_ALIVE
    reward = 0
    reward += int(food_collected) * hyp.REW_FOOD
    reward += hyp.REW_ALIVE
    reward -= hyp.PEN_DEATH * int(game_over)
    reward -= hyp.PEN_TIMEOUT * int(timeout)
    # self.timer.tick(60)
    # renderer.render([self.snake])
    # obs (scaled to 1), reward, terminated, truncated, info
    return self.snake.get_state() / 2, reward, game_over, timeout, None

  def reset(self, seed: int | None = None, options = None,):
    self.snake.reset()
    return self.snake.get_state()
  
  def render(self, mode='human'):
    # renderer.render([self.snake])
    pass

register(
     id="snake",
     entry_point="main:SnakeEnv",
     max_episode_steps=None,
)


base_env = GymEnv("snake", render_mode="human")
env = TransformedEnv(
    base_env,
    Compose(
        # normalize observations
        ObservationNorm(in_keys=["observation"], standard_normal=True),
        DoubleToFloat(
            in_keys=["observation"],
        ),
        StepCounter(),
    ),
)

class SnakeCNNActor(nn.Module):
  def __init__(self):
    super(SnakeCNNActor, self).__init__()

    self.cnn_layers = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
    )

    self.linear_layers = nn.Sequential(
      # 128 filters * image dims shrunk 2x by 1 maxpool layer
      nn.Linear(128 * (snake.GRID[0] // 2) * (snake.GRID[1] // 2), 512), nn.ReLU(),
      nn.Linear(512, 512), nn.ReLU(),
      nn.Linear(512, 4),
    )

  def forward(self, x: torch.Tensor):
    x = self.cnn_layers(x.unsqueeze(-1).permute(2, 0, 1))
    print("shape of x before:", x.shape, flush=True)
    x = x.flatten()
    print("shape of x:", x.shape, flush=True)
    x = self.linear_layers(x)
    return x
  

class SnakeCNNCritic(nn.Module):
  def __init__(self):
    super(SnakeCNNCritic, self).__init__()

    self.cnn_layers = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )

    self.linear_layers = nn.Sequential(
      # 128 filters * image dims shrunk 2x by 1 maxpool layer
      nn.Linear(128 * (snake.GRID[0] // 2) * (snake.GRID[1] // 2), 512), nn.ReLU(),
      nn.Linear(512, 512), nn.ReLU(),
      nn.Linear(512, 1),
    )

  def forward(self, x):
    x = self.cnn_layers(x.unsqueeze(-1).permute(2, 0, 1))
    x = x.flatten()
    x = self.linear_layers(x)
    return x


model = TensorDictModule(
  SnakeCNNActor(),
  in_keys = ["observation"],
  out_keys = ["right", "left", "up", "down"],
)

critic = ValueOperator(
  SnakeCNNCritic(),
  in_keys = ["observation"],
)

actor = ProbabilisticActor(
  model,
  in_keys=["right", "left", "up", "down"],
  distribution_class=OneHotCategorical,
  return_log_prob=True
)

# the lengths i'll go to for autocomplete
print("calculating normalization constants lmao")
typing.cast(ObservationNorm, typing.cast(Compose, env.transform)[0])\
  .init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
print("Done.")

check_env_specs(env)


print("normalization constant shape:", env.transform[0].loc.shape) # type: ignore

print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("done_spec:", env.done_spec)
print("action_spec:", env.action_spec)
print("state_spec:", env.state_spec)

collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch=1000,
    total_frames=1_000_000
)