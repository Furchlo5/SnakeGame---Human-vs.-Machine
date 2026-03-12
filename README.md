# 🐍 Snake Game — Human vs. Machine

A modular, OOP-based Snake Game built with **Python + Pygame**, featuring both a human-playable mode and a **Deep Q-Learning (DQN) AI agent** built with PyTorch.

---

## Project Structure

```
snake/
├── game.py          # SnakeGameAI class — core game environment
├── model.py         # Linear_QNet neural network + QTrainer
├── agent.py         # DQN Agent (state, memory, epsilon-greedy)
├── train.py         # Training loop with live matplotlib plot
├── human_play.py    # Play manually with arrow / WASD keys
├── watch.py         # Watch the trained AI play
├── ai_demo.py       # Random-action demo (validates RL API)
└── requirements.txt
```

---

## Quick Start

> **Requires Python 3.10–3.12** (pygame does not yet support 3.14)

```bash
# 1. Create virtual environment with Python 3.12
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3a. Play as a human
python human_play.py

# 3b. Train the AI
python train.py

# 3c. Watch the trained AI play
python watch.py
```

---

## Modes

### 🎮 Human Mode (`human_play.py`)
- Arrow keys or **WASD** to control the snake
- Speed increases as the snake grows
- **Game Over screen** with Restart / Exit buttons after each death
  - `R` = Restart &nbsp;&nbsp; `Q` / `Esc` = Exit

### 🤖 AI Training (`train.py`)
- Trains a Deep Q-Network from scratch
- Live **matplotlib plot** shows score and mean score over time
- Best model auto-saved to `./model/model.pth` on every new high score
- Stops and resumes from checkpoint automatically on restart

### 👁️ Watch Mode (`watch.py`)
- Loads the saved model and plays with no exploration (ε = 0)
- Speed increases as the snake grows (same formula as human mode)
- **Game Over screen** after each episode — Restart or Exit

---

## Speed System

Both human and watch modes use the same dynamic FPS formula:

| Snake Length | Speed |
|---|---|
| 3 (start) | 10 FPS |
| 13 | ~15 FPS |
| 43 | ~30 FPS (max) |

Configurable at the top of `human_play.py` and `watch.py`:
```python
BASE_FPS = 10   # starting speed
MAX_FPS  = 30   # maximum speed
FPS_STEP = 0.5  # FPS gained per extra snake segment
```

---

## AI — Deep Q-Learning (DQN)

### State Vector (11 booleans)
| # | Feature |
|---|---------|
| 0–2 | Danger: straight / right / left |
| 3–6 | Current direction: L / R / U / D |
| 7–10 | Food location: left / right / up / down |

### Neural Network
```
Input (11) → ReLU → Hidden (256) → Output (3 Q-values)
```
Action with highest Q-value is chosen: `[Straight, Right, Left]`

### Reward Structure
| Event | Reward |
|---|---|
| Eat food | `+10.0` |
| Collision / timeout | `−10.0` |
| Each step | `−0.01` |

### Exploration (ε-Greedy)
Random moves for the first ~80 games, then gradually shifts to pure model decisions.

---

## RL Interface (`game.py`)

```python
from game import SnakeGameAI

env = SnakeGameAI(ai_playing=True, render=True)
env.reset()

while True:
    action = [1, 0, 0]  # straight | [0,1,0] right | [0,0,1] left
    reward, game_over, score = env.play_step(action)
    if game_over:
        env.reset()
```

| Method | Returns | Description |
|---|---|---|
| `reset()` | `None` | Restart environment |
| `play_step(action)` | `(reward, game_over, score)` | One RL step |
| `play_human_step(fps)` | `(game_over, score)` | One keyboard step |
| `show_game_over_screen()` | `'restart'` | Shows overlay, waits for input |
| `is_collision(pt=None)` | `bool` | Wall or self collision check |

---

## Grid & Window

| Property | Value |
|---|---|
| Grid | 30 × 25 cells |
| Cell size | 20 × 20 px |
| Window | 600 × 540 px |
| Header | 40 px (score + mode badge) |

---

## Dependencies

| Library | Purpose |
|---|---|
| `pygame` | Game rendering, input |
| `torch` | Neural network, training |
| `matplotlib` | Live training plot |
| `numpy` | State array conversion |
