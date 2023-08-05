import random
from enum import Enum
from collections import namedtuple
import hyperparams as hp

class Direction(Enum):
  RIGHT = 0
  LEFT = 1
  UP = 2
  DOWN = 3
  
Point = namedtuple('Point', 'x, y')

GRID = (20, 20)

class SnakeGame:
  
  def __init__(self):
    self.reset()

  def reset(self):
    # init game state
    self.direction = Direction.RIGHT
    self.ticks_alive = 0
    
    self.snake = [Point(GRID[0]//2 + 2, GRID[1]//2), 
            Point(GRID[0]//2 + 1, GRID[1]//2),
            Point(GRID[0]//2 + 0, GRID[1]//2)]
    self.head = self.snake[0]
    
    self.score = 0
    self.food = self._place_food()
    
  def _place_food(self):
    x = random.randint(0, GRID[0]-1) 
    y = random.randint(0, GRID[1]-1)
    food = Point(x, y)
    if food in self.snake:
      self._place_food()
    return food 
    
  def tick(self) -> tuple[bool, int]:
    # 1. move
    self._move(self.direction) # update the head
    self.snake.insert(0, self.head)
    
    self.ticks_alive += 1

    # 2. place new food or just move
    if self.head == self.food:
      self.score += 1
      self.food = self._place_food()
    else:
      self.snake.pop()

    # 3. check if game over
    if self._is_collision():
      return True, self.score
    
    # 4. return game over and score
    return False, self.score
  
  def change_direction(self, new_dir: Direction):
    # Annoyingly long, but simplest way to prevent going backwards as we're only really expecting one move per tick
    if self.direction == Direction.RIGHT and new_dir == Direction.LEFT:
      return
    if self.direction == Direction.LEFT and new_dir == Direction.RIGHT:
      return
    if self.direction == Direction.UP and new_dir == Direction.DOWN:
      return
    if self.direction == Direction.DOWN and new_dir == Direction.UP:
      return
    
    self.direction = new_dir

  def _is_collision(self) -> bool:
    # hits boundary
    if self.head.x > GRID[0] or self.head.x < 0 or self.head.y > GRID[1] or self.head.y < 0:
      return True
    # hits itself
    if self.head in self.snake[1:]:
      return True
    
    return False
    
    
  def _move(self, direction: Direction):
    x = self.head.x
    y = self.head.y
    if direction == Direction.RIGHT:
      x += 1
    elif direction == Direction.LEFT:
      x -= 1
    elif direction == Direction.DOWN:
      y += 1
    elif direction == Direction.UP:
      y -= 1
    else:
      raise ValueError
      
    self.head = Point(x, y)

  def calc_reward(self) -> float:
    return hp.REWARD_MULTIPLIER_TIME_ALIVE * self.ticks_alive + hp.REWARD_MULTIPLIER_POINTS * self.score