import snake
import snakerenderer
import torch
import pygame

if __name__ == '__main__':
  # Create 15 snake games
  games = [snake.SnakeGame() for _ in range(15)]
  # Create a renderer
  renderer = snakerenderer.SnakeRenderer(len(games))
  clock = pygame.time.Clock()

  while True:
    for i, game in enumerate(games):
      game.direction = snake.Direction(torch.randint(0, 3, (1,)).item())
      go, score = game.tick()
      if go:
        game.reset()

    clock.tick(40)
    renderer.render(games)
