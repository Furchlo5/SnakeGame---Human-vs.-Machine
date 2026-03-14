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
from hvai_game import HumanVsAIGame
from model import Linear_QNet

# ── Dynamic speed (same formula as human_play / watch) ───────
BASE_FPS = 10
MAX_FPS  = 30
FPS_STEP = 0.5


def dynamic_fps(snake_len: int) -> int:
    return int(min(BASE_FPS + (snake_len - 3) * FPS_STEP, MAX_FPS))


def main() -> None:
    model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
    loaded = model.load()
    if not loaded:
        print("⚠️  No trained model found at ./model/model.pth")
        print("   Train first: python train.py")
        print("   Continuing with untrained network (random-ish AI).\n")
    model.eval()

    game = HumanVsAIGame()
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
                # ── AI picks action ───────────────────────────────
                if game.ai_alive:
                    state   = game.get_ai_state()
                    state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                    with torch.no_grad():
                        q_vals = model(state_t)
                    idx    = int(q_vals.argmax().item())
                    action = [0, 0, 0]
                    action[idx] = 1
                else:
                    action = [1, 0, 0]   # dummy, snake is dead — ignored in play_step
    
                # ── Dynamic FPS ───────────────────────────────────
                fps = dynamic_fps(max(len(game.human_snake), len(game.ai_snake)))
    
                # ── Step ──────────────────────────────────────────
                h_done, ai_done, h_score, ai_score = game.play_step(action, fps=fps)
    
                # ── BOTH must be dead to end the round ───────────
                if not game.human_alive and not game.ai_alive:
                    break

            res = game.show_result_screen()   # blocks until PLAY AGAIN, MENU or EXIT
            if res == "menu":
                break  # Break out to the start menu loop
            elif res != "restart":
                pygame.quit(); raise SystemExit


if __name__ == "__main__":
    main()
