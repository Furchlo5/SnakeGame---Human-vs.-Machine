"""
human_play.py — Play Snake yourself with arrow keys / WASD.
Run: python human_play.py

Speed: starts slow, increases as the snake grows.
"""

from game import SnakeGameAI

# Speed settings (FPS)
BASE_FPS  = 10    # starting speed (short snake)
MAX_FPS   = 30   # maximum speed (long snake)
FPS_STEP  = 0.5  # extra FPS gained per snake segment grown


def dynamic_fps(snake_len: int) -> int:
    """FPS based on current snake length."""
    fps = BASE_FPS + (snake_len - 3) * FPS_STEP
    return int(min(fps, MAX_FPS))


if __name__ == "__main__":
    game = SnakeGameAI(ai_playing=False, render=True)
    print("Controls: Arrow keys or WASD  |  R = restart  |  Q / Esc = exit")

    while True:
        fps = dynamic_fps(len(game.snake))
        game_over, score = game.play_human_step(fps=fps)

        if game_over:
            print(f"Game Over! Score: {score}")
            game.show_game_over_screen()
            game.reset()
