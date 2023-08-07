import snake
import snakerenderer
import gymnasium as gym
import numpy as np
import typing
import math
from gymnasium import spaces
from gymnasium.envs.registration import register
import hyperparams as hyp

# todo: delete unused
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
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
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
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
    renderer.render([self.snake])
    # obs (scaled to 1), reward, terminated, truncated, info
    return self.snake.get_state() / 2, reward, game_over, timeout, None

  def reset(self):
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
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(
            in_keys=["observation"],
        ),
        StepCounter(),
    ),
)

# the lengths i'll go to for autocomplete
typing.cast(ObservationNorm, typing.cast(Compose, env.transform)[0])\
  .init_stats(num_iter=1000, reduce_dim=1, cat_dim=1)

print("normalization constant shape:", env.transform[0].loc.shape) # type: ignore

print("observation_spec:", env.observation_spec)
print("reward_spec:", env.reward_spec)
print("done_spec:", env.done_spec)
print("action_spec:", env.action_spec)
print("state_spec:", env.state_spec)