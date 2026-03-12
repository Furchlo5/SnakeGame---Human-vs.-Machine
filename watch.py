"""
watch.py — Trained model izleme modu
======================================
Eğitilmiş modeli ./model/model.pth'ten yükler ve oynayışını seyreder.
Epsilon = 0 (tamamen model kararı, hiç rastgele hamle yok).

Hız yılan boyuna göre otomatik artar.
Başlangıç ve maksimum hızı aşağıdan ayarlayabilirsin.

Run: python watch.py
"""

import torch
from game  import SnakeGameAI
from model import Linear_QNet
from agent import Agent

# Speed settings (FPS)
BASE_FPS  = 10    # başlangıç hızı (kısa yılan)
MAX_FPS   = 30   # maksimum hız   (uzun yılan)
FPS_STEP  = 0.5  # her segment büyümede eklenen FPS


def dynamic_fps(snake_len: int) -> int:
    fps = BASE_FPS + (snake_len - 3) * FPS_STEP
    return int(min(fps, MAX_FPS))


def watch() -> None:
    model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
    loaded = model.load()
    if not loaded:
        print("⚠️  Kayıtlı model bulunamadı (./model/model.pth).")
        print("   Önce 'python train.py' ile biraz eğit.")
        return
    model.eval()

    agent = Agent.__new__(Agent)
    agent.model = model

    game    = SnakeGameAI(ai_playing=True, render=True)
    record  = 0
    game_num = 0

    print(f"🎮  İzleme modu — hız yılan boyuyla artıyor ({BASE_FPS}→{MAX_FPS} FPS)")
    print("    Yandıktan sonra Restart veya Exit butonuna bas.\n")

    while True:
        state = agent.get_state(game)

        with torch.no_grad():
            state_t  = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = model(state_t)
        idx    = int(q_values.argmax().item())
        action = [0, 0, 0]
        action[idx] = 1

        reward, done, score = game.play_step(action)

        # FPS yılan boyuna göre dinamik
        game.clock.tick(dynamic_fps(len(game.snake)))

        if done:
            game_num += 1
            if score > record:
                record = score
            print(f"Oyun {game_num:>4}  |  Skor: {score:>3}  |  Rekor: {record:>3}")
            game.show_game_over_screen()
            game.reset()


if __name__ == "__main__":
    watch()
