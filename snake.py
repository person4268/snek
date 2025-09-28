import random
from enum import Enum
from collections import namedtuple
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
  WALL = 4  # <-- ADDED

Point = namedtuple('Point', 'x, y')

# This now represents the outer boundary, including walls.
# The playable area will be inside this.
GRID = (11, 11)

class SnakeGame:
  
  def __init__(self):
    self.reset()

  def create_random_snake_position_and_direction(self) -> tuple[Direction, list[Point]]:
    direction = random.choice(list(Direction))
    snake = []
    # Spawn snake away from walls
    x = random.randint(5, GRID[0]-6)
    y = random.randint(5, GRID[1]-6)
    snake.append(Point(x, y))
    
    # Ensure initial body is also within walls
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

    return direction, snake

  def reset(self):
    self.ticks_alive = 0
    self.direction, self.snake = self.create_random_snake_position_and_direction()
    self.head = self.snake[0]
    
    self.score = 0
    self.food = self._place_food()
    self._game_over = False

  def get_state(self) -> np.ndarray:
    # Create grid and draw walls on the border
    grid = np.zeros(shape=GRID, dtype=np.float64)
    grid[0, :] = GridState.WALL.value
    grid[-1, :] = GridState.WALL.value
    grid[:, 0] = GridState.WALL.value
    grid[:, -1] = GridState.WALL.value
    
    # Draw food, snake, and head
    grid[self.food.y, self.food.x] = GridState.FOOD.value
    for point in self.snake:
      grid[point.y, point.x] = GridState.SNAKE.value
    grid[self.head.y, self.head.x] = GridState.HEAD.value
    return grid
    
  def _place_food(self):
    # Place food within the walls (from 1 to GRID_SIZE-2)
    x = random.randint(1, GRID[0]-2) 
    y = random.randint(1, GRID[1]-2)
    food = Point(x, y)
    if food in self.snake:
      return self._place_food() # Recurse if food is inside the snake
    return food 
  
  def tick(self) -> tuple[bool, bool, int]:
    if self._game_over:
      return True, False, self.score

    self._move(self.direction)
    self.snake.insert(0, self.head)
    self.ticks_alive += 1
    food_collected = False

    if self.head == self.food:
      self.score += 1
      self.food = self._place_food()
      food_collected = True
    else:
      self.snake.pop()

    if self._is_collision():
      self._game_over = True
      return True, food_collected, self.score
    
    return False, food_collected, self.score
  
  def change_direction(self, new_dir: Direction):
    if self.direction == Direction.RIGHT and new_dir == Direction.LEFT: return
    if self.direction == Direction.LEFT and new_dir == Direction.RIGHT: return
    if self.direction == Direction.UP and new_dir == Direction.DOWN: return
    if self.direction == Direction.DOWN and new_dir == Direction.UP: return
    self.direction = new_dir

  def _is_collision(self) -> bool:
    # Hits the wall boundary
    if self.head.x >= GRID[0]-1 or self.head.x <= 0 or self.head.y >= GRID[1]-1 or self.head.y <= 0:
      return True
    # Hits itself
    if self.head in self.snake[1:]:
      return True
    return False
    
  def _move(self, direction: Direction):
    x, y = self.head.x, self.head.y
    if direction == Direction.RIGHT: x += 1
    elif direction == Direction.LEFT: x -= 1
    elif direction == Direction.DOWN: y += 1
    elif direction == Direction.UP: y -= 1
    self.head = Point(x, y)