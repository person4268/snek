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

from collections import defaultdict

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import OneHotCategorical
from torch.utils.tensorboard.writer import SummaryWriter
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
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"

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


base_env = GymEnv("snake",
                   #render_mode="human",
                   device=device)
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


# the lengths i'll go to for autocomplete
print("calculating normalization constants lmao")
typing.cast(ObservationNorm, typing.cast(Compose, env.transform)[0])\
  .init_stats(num_iter=7000, reduce_dim=0, cat_dim=0)
print("Done.")

check_env_specs(env)

class SnakeCNNActor(nn.Module):
  def __init__(self):
    super(SnakeCNNActor, self).__init__()

    self.cnn_layers = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, device=device), nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, device=device), nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, device=device), nn.ReLU(),
    )

    self.linear_layers = nn.Sequential(
      nn.LazyLinear(256, device=device), nn.ReLU(),
      nn.LazyLinear(256, device=device), nn.ReLU(),
      nn.LazyLinear(4, device=device),
    )

  def forward(self, x):
    x = x.unsqueeze(-1)
    batched = False
    if len(x.shape) == 3:
      x = x.permute(2, 0, 1)
    else:
      x = x.permute(0, 3, 1, 2)
      batched = True
    x = self.cnn_layers(x)
    if not batched:
      x = x.flatten()
    else:
      x = x.flatten(start_dim=1)
    x = self.linear_layers(x)
    return x
  

class SnakeCNNCritic(nn.Module):
  def __init__(self):
    super(SnakeCNNCritic, self).__init__()

    self.cnn_layers = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, device=device), nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, device=device), nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, device=device), nn.ReLU(),
    )

    self.linear_layers = nn.Sequential(
      nn.LazyLinear(256, device=device), nn.ReLU(),
      nn.LazyLinear(256, device=device), nn.ReLU(),
      nn.LazyLinear(1, device=device),
    )

  def forward(self, x):
    x = x.unsqueeze(-1)
    batched = False
    if len(x.shape) == 3:
      x = x.permute(2, 0, 1)
    else:
      x = x.permute(0, 3, 1, 2)
      batched = True
    x = self.cnn_layers(x)
    if not batched:
      x = x.flatten()
    else:
      x = x.flatten(start_dim=1)
    x = self.linear_layers(x)
    return x


model = TensorDictModule(
  SnakeCNNActor(),
  in_keys = ["observation"],
  out_keys = ["logits"],
)

value_module = ValueOperator(
  SnakeCNNCritic(),
  in_keys = ["observation"],
)

policy_module = ProbabilisticActor(
  model,
  spec=env.action_spec,
  in_keys=["logits"],
  distribution_class=OneHotCategorical,
  return_log_prob=True
)

policy_module(env.reset())
value_module(env.reset())

# print("normalization constant shape:", env.transform[0].loc.shape) # type: ignore

# print("observation_spec:", env.observation_spec)
# print("reward_spec:", env.reward_spec)
# print("done_spec:", env.done_spec)
# print("action_spec:", env.action_spec)
# print("state_spec:", env.state_spec)

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=hyp.frames_per_batch,
    total_frames=hyp.total_frames,
    device=device
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(collector.frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=hyp.gamma, lmbda=hyp.lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    clip_epsilon=hyp.clip_epsilon,
    entropy_bonus=bool(hyp.entropy_eps),
    entropy_coef=hyp.entropy_eps,
    # these keys match by default but we set this for completeness
    value_target_key=advantage_module.value_target_key,
    critic_coef=1.0,
    gamma=0.99,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), hyp.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, hyp.total_frames // hyp.frames_per_batch, 0.0
)

# now we can finally actually get some training done

progress = tqdm(total=hyp.total_frames)
logs = defaultdict(list)
eval_str = ""

writer = SummaryWriter()

# collect each batch of frames
for batch_num, tensordict_data in enumerate(collector):
  for _ in range(hyp.num_epochs):
    # update our advantage, without affecting training
    with torch.no_grad():
      advantage_module(tensordict_data)

    # i think this flattens the data, idk why they didn't just use .flatten
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view.cpu()) # type: ignore , why pytorch why
    for _ in range(hyp.frames_per_batch // hyp.sub_batch_size):
      subdata = replay_buffer.sample(hyp.sub_batch_size)
      loss_vals = loss_module(subdata.to(device))
      loss_value = (
        loss_vals["loss_objective"]
        + loss_vals["loss_critic"]
        + loss_vals["loss_entropy"]
      )
      loss_value.backward()
      torch.nn.utils.clip_grad_norm_(loss_module.parameters(), hyp.max_grad_norm) # type: ignore this time because it's a private function
      optim.step()
      optim.zero_grad()

    steps = tensordict_data.numel()
    logs["reward"].append(tensordict_data["next", "reward"].mean().item()) # type: ignore
    writer.add_scalar("reward", logs["reward"][-1], steps)
    progress.update(steps)
    cum_reward_str = (
      f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item()) # type: ignore
    writer.add_scalar("step_count", logs["step_count"][-1], steps)
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    writer.add_scalar("lr", logs["lr"][-1], steps)
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

    if batch_num % 2 == 0:
      #it's evaluation time
      print("evaluating", batch_num)
      with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        eval_rollout = env.rollout(1000, policy_module)
        env.base_env.render()
        logs["eval reward"].append(eval_rollout["next", "reward"].mean().item()) # type: ignore
        logs["eval reward (sum)"].append(
            eval_rollout["next", "reward"].sum().item() # type: ignore
        )
        logs["eval step_count"].append(eval_rollout["step_count"].max().item()) # type: ignore
        writer.add_scalar("eval reward", logs["eval reward"][-1], steps)
        writer.add_scalar("eval reward (sum)", logs["eval reward (sum)"][-1], steps)
        writer.add_scalar("eval step_count", logs["eval step_count"][-1], steps)
        eval_str = (
            f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
            f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            f"eval step-count: {logs['eval step_count'][-1]}"
        )
        del eval_rollout

    progress.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    scheduler.step()

progress.close()
writer.close()

