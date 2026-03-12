"""
ai_demo.py — A random-action demo to verify the RL API is functioning.

This simulates what an RL agent would do:
  - Calls game.play_step(action) in a loop
  - Resets when game_over=True
  - Stops after MAX_GAMES episodes

Run: python ai_demo.py
"""

import random
from game import SnakeGameAI

MAX_GAMES   = 5
RENDER      = True     # set False for headless speed test

if __name__ == "__main__":
    game   = SnakeGameAI(ai_playing=True, render=RENDER)
    scores = []

    for episode in range(1, MAX_GAMES + 1):
        game.reset()
        steps = 0

        while True:
            # Random agent: pick one of [straight, right, left] uniformly
            action_idx = random.randint(0, 2)
            action     = [0, 0, 0]
            action[action_idx] = 1

            reward, game_over, score = game.play_step(action)
            steps += 1

            if game_over:
                scores.append(score)
                print(f"Episode {episode:>3} | Steps: {steps:>5} | "
                      f"Score: {score:>3} | Reward last: {reward:+.2f}")
                break

    print(f"\nAverage score over {MAX_GAMES} random episodes: "
          f"{sum(scores) / len(scores):.2f}")
