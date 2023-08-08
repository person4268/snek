import random
from enum import Enum
from collections import namedtuple
import hyperparams as hp
import numpy as np

class Direction(Enum):
  RIGHT = 0
  LEFT = 1
  UP = 2
  DOWN = 3

class GridState(Enum):
  EMPTY = 0
  SNAKE = 1
  FOOD = 2
  HEAD = 3
  
Point = namedtuple('Point', 'x, y')

GRID = (21, 21)

class SnakeGame:
  
  def __init__(self):
    self.reset()

  def create_random_snake_position_and_direction(self) -> tuple[Direction, list[Point]]:
    direction = random.choice(list(Direction))
    snake = []
    direction = random.choice(list(Direction))
    x = random.randint(4, GRID[0]-4)
    y = random.randint(4, GRID[1]-4)
    snake.append(Point(x, y))
    
    if direction == Direction.UP:
        snake.append(Point(x, y+1))
        snake.append(Point(x, y+2))
    elif direction == Direction.DOWN:
        snake.append(Point(x, y-1))
        snake.append(Point(x, y-2))
    elif direction == Direction.LEFT:
        snake.append(Point(x+1, y))
        snake.append(Point(x+2, y))
    elif direction == Direction.RIGHT:
        snake.append(Point(x-1, y))
        snake.append(Point(x-2, y))
    
    print(direction, snake)

    return direction, snake

  def reset(self):
    # init game state
    self.ticks_alive = 0
    
    # self.direction = Direction.RIGHT
    # self.snake = [Point(GRID[0]//2 + 2, GRID[1]//2), 
    #         Point(GRID[0]//2 + 1, GRID[1]//2),
    #         Point(GRID[0]//2 + 0, GRID[1]//2)]
    self.direction, self.snake = self.create_random_snake_position_and_direction()
    self.head = self.snake[0]
    
    self.score = 0
    self.food = self._place_food()
    self._game_over = False

  def get_state(self) -> np.ndarray:
    grid = np.zeros(shape=GRID, dtype=np.float16)
    grid[self.food.x][self.food.y] = GridState.FOOD.value
    for point in self.snake:
      # skip if out of bounds
      if point.x < 0 or point.x >= GRID[0] or point.y < 0 or point.y >= GRID[1]:
        continue
      grid[point.x][point.y] = GridState.SNAKE.value
    if self.head.x < 0 or self.head.x >= GRID[0] or self.head.y < 0 or self.head.y >= GRID[1]:
        return grid
    grid[self.head.x][self.head.y] = GridState.HEAD.value
    return grid
    
  def _place_food(self):
    x = random.randint(0, GRID[0]-1) 
    y = random.randint(0, GRID[1]-1)
    food = Point(x, y)
    if food in self.snake:
      self._place_food()
    return food 
  
  # Returns game over, whether food was collected, and score
  def tick(self) -> tuple[bool, bool, int]:

    if self._game_over:
      return True, False, self.score

    # 1. move
    self._move(self.direction) # update the head
    self.snake.insert(0, self.head)
    
    self.ticks_alive += 1

    food_collected = False

    # 2. place new food or just move
    if self.head == self.food:
      self.score += 1
      self.food = self._place_food()
      food_collected = True

    else:
      self.snake.pop()

    # 3. check if game over
    if self._is_collision():
      self._game_over = True
      return True, food_collected, self.score
    
    # 4. return game over and score
    return False, food_collected, self.score
  
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
    if self.head.x >= GRID[0] or self.head.x < 0 or self.head.y >= GRID[1] or self.head.y < 0:
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
