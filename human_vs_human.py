"""
human_vs_ai.py — Human vs. AI Snake entry point
=================================================
Left  side : YOU — arrow keys / WASD
Right side : Trained AI model

Run: python human_vs_ai.py

Needs a trained model at ./model/model.pth
Train first with: python train.py
"""

import torch
import pygame
from hvh_game import HumanVsHumanGame

# ── Dynamic speed (same formula as human_play / watch) ───────
BASE_FPS = 10
MAX_FPS  = 30
FPS_STEP = 0.5


def dynamic_fps(snake_len: int) -> int:
    return int(min(BASE_FPS + (snake_len - 3) * FPS_STEP, MAX_FPS))


def main() -> None:
    game = HumanVsHumanGame()
    print("Controls: Arrow keys / WASD  |  P = pause  |  R = restart  |  Q / Esc = exit")

    while True:
        action_screen = game.show_start_screen()
        if action_screen == "leaderboard":
            game.show_leaderboard_screen()
            continue
        elif action_screen != "play":
            break
            
        while True:
            game.reset()
            # ── Play Loop ───────────────
            while True:
                # ── Dynamic FPS ───────────────────────────────────
                fps = dynamic_fps(max(len(game.p1_snake), len(game.p2_snake)))
    
                # ── Step ──────────────────────────────────────────
                p1_done, p2_done, p1_score, p2_score = game.play_step(fps=fps)
    
                # ── BOTH must be dead to end the round ───────────
                if not game.p1_alive and not game.p2_alive:
                    break

            res = game.show_result_screen()   # blocks until PLAY AGAIN, MENU or EXIT
            if res == "menu":
                break  # Break out to the start menu loop
            elif res != "restart":
                pygame.quit(); raise SystemExit


if __name__ == "__main__":
    main()
