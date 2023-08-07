import pygame
import snake
import math

# 15 times a 20x20 grid is 300x300, and 4 wide makes that 1200x1200, so reduce this if you need to
SCALE = 15


class SnakeRenderer():
  def __init__(self, num_snakes = 1) -> None:
    # How many pixels wide one snake game is
    self.snake_game_width = snake.GRID[0] * SCALE
    self.snake_game_height = snake.GRID[1] * SCALE
    # How many games we have (horizontally)
    self.snake_game_count = math.ceil(math.sqrt(num_snakes))
    self.num_snakes = num_snakes
    self._has_inited = False
    
    # How many pixels wide to make the viewport
    self.scale = SCALE * self.snake_game_count

  def render(self, games: list[snake.SnakeGame]):
    # Don't create a window until the first time we try and render
    if not self._has_inited:
      pygame.init()
      pygame.display.set_caption(f'{self.num_snakes} Snake Game' + ('s' if self.num_snakes > 1 else '') + ". Sure hope you like snakes.")
      self.screen = pygame.display.set_mode((int(snake.GRID[0]*self.scale), int(snake.GRID[1]*self.scale)))
      self._has_inited = True

    self.screen.fill((0,0,0))
    assert len(games) <= self.snake_game_count**2, f"Too many games! {len(games)} > {self.snake_game_count**2}"
    for i, game in enumerate(games):
      x_start = i % self.snake_game_count * self.snake_game_width
      y_start = i // self.snake_game_count * self.snake_game_height

      # Draw the snake
      for j, point in enumerate(game.snake):
        # Left, top, width, height
        color = (255, 0, 0) if j == 0 else (50, 0, 200)
        pygame.draw.rect(self.screen, color, (x_start + point.x*SCALE, y_start + point.y*SCALE, SCALE, SCALE))

      # Draw the food
      pygame.draw.rect(self.screen, (0, 255, 0), (x_start + game.food.x*SCALE, y_start + game.food.y*SCALE, SCALE, SCALE))

      # Display score and # of ticks alive
      font_size = 20
      margin = 5
      font = pygame.font.SysFont(pygame.font.get_default_font(), font_size)
      text = font.render(f'S: {game.score}, T: {game.ticks_alive}', True, (255, 255, 255))
      self.screen.blit(text, (x_start + margin, y_start + self.snake_game_height - font_size - margin))

      # Draw border between games (ngl there might be an off by one error somewhere)
      pygame.draw.lines(self.screen, (255, 255, 255), False, 
                        [(x_start, y_start), # TL
                         (x_start + self.snake_game_width, y_start), # TL -> TR
                         (x_start + self.snake_game_width, y_start + self.snake_game_height), # TR -> BR
                         (x_start, y_start + self.snake_game_height), # BR -> BL
                         (x_start, y_start)], # BL to TL
                         1)
    pygame.display.flip()


if __name__ == "__main__":
  game = snake.SnakeGame()
  renderer = SnakeRenderer(1)

  while True:
    renderer.render([game])
    for event in pygame.event.get():
      if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_UP:
          game.change_direction(snake.Direction.UP)
        elif event.key == pygame.K_DOWN:
          game.change_direction(snake.Direction.DOWN)
        elif event.key == pygame.K_LEFT:
          game.change_direction(snake.Direction.LEFT)
        elif event.key == pygame.K_RIGHT:
          game.change_direction(snake.Direction.RIGHT)

    game_over, food_collected, score = game.tick()
    if game_over:
      game.reset()
    pygame.time.wait(100)