# 🐍 Snake Game — RL Edition

A modular, OOP-based Snake Game built with **Python + Pygame**, designed as a clean environment for Reinforcement Learning integration.

---

## Project Structure

```
snake/
├── game.py          # SnakeGameAI class — the RL environment
├── human_play.py    # Play manually with arrow / WASD keys
├── ai_demo.py       # Random-action RL demo (validates the API)
└── requirements.txt
```

---

## Quick Start

> **Requires Python 3.10–3.12** (pygame does not yet support 3.14)

```bash
# 1. Create virtual environment
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install dependency
pip install -r requirements.txt

# 3a. Play as a human
python human_play.py

# 3b. Watch a random AI play (tests the RL interface)
python ai_demo.py
```

---

## RL Interface (`game.py`)

### `SnakeGameAI(ai_playing=True, render=True)`

| Parameter    | Type  | Description                                              |
|-------------|-------|----------------------------------------------------------|
| `ai_playing` | bool  | `True` = RL mode (no FPS cap); `False` = human mode     |
| `render`     | bool  | `True` = Pygame window; `False` = headless (fast train)  |

### Methods

| Method                   | Returns                          | Description                              |
|--------------------------|----------------------------------|------------------------------------------|
| `reset()`                | `None`                           | Restart the environment                   |
| `play_step(action)`      | `(reward, game_over, score)`     | Advance one step with an RL action        |
| `play_human_step()`      | `(game_over, score)`             | Advance one step with keyboard input      |
| `is_collision(pt=None)`  | `bool`                           | Check if a point (default: head) collides |

### Action Encoding (Relative Directions)

```python
[1, 0, 0]  # Straight ahead
[0, 1, 0]  # Turn right (clockwise)
[0, 0, 1]  # Turn left  (counter-clockwise)
```

### Reward Structure

| Event          | Reward  |
|----------------|---------|
| Eat food       | `+10.0` |
| Collision / timeout | `-10.0` |
| Each step      | `-0.01` |

### State Accessors

```python
game.head_pos          # namedtuple Point(x, y)
game.body              # list[Point] — snake body (excluding head)
game.current_direction # Direction enum
game.score             # int
game.snake             # full list[Point] including head
```

---

## Integrating Your RL Agent

```python
from game import SnakeGameAI, Point, Direction

env = SnakeGameAI(ai_playing=True, render=True)

for episode in range(1000):
    env.reset()
    while True:
        state  = your_agent.get_state(env)
        action = your_agent.get_action(state)
        reward, game_over, score = env.play_step(action)
        your_agent.train(state, action, reward, game_over)
        if game_over:
            break
```

---

## Grid & Window

| Property    | Value               |
|------------|---------------------|
| Grid size   | 30 × 25 cells       |
| Cell size   | 20 × 20 px          |
| Window      | 600 × 540 px        |
| Header bar  | 40 px (score + mode)|
