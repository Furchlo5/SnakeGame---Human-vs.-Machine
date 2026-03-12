"""
agent.py — Deep Q-Learning Agent
=================================
Handles:
  • State extraction from the game
  • Epsilon-greedy action selection
  • Short-memory (single step) and long-memory (replay batch) training
"""

import random
import numpy as np
from collections import deque

from game import SnakeGameAI, Direction, Point, BLOCK
from model import Linear_QNet, QTrainer

# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
MAX_MEMORY   = 100_000   # max transitions stored in replay buffer
BATCH_SIZE   = 1_000     # transitions sampled for long-memory training
LR           = 1e-3      # Adam learning rate
GAMMA        = 0.9       # discount factor


class Agent:
    """
    Deep Q-Learning agent for SnakeGameAI.

    State vector (11 booleans)
    --------------------------
    [0]  danger straight
    [1]  danger right
    [2]  danger left
    [3]  direction left
    [4]  direction right
    [5]  direction up
    [6]  direction down
    [7]  food left
    [8]  food right
    [9]  food up
    [10] food down
    """

    def __init__(self):
        self.n_games  = 0
        self.epsilon  = 0        # exploration rate (managed dynamically)
        self.memory   = deque(maxlen=MAX_MEMORY)

        self.model   = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=GAMMA)

        # Try loading a pre-existing checkpoint
        self.model.load()

    # ─────────────────────────────────────────
    # State
    # ─────────────────────────────────────────

    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        head = game.head
        dir_ = game.current_direction

        # Four candidate points one step ahead of the head
        pt_l = Point(head.x - BLOCK, head.y)
        pt_r = Point(head.x + BLOCK, head.y)
        pt_u = Point(head.x,         head.y - BLOCK)
        pt_d = Point(head.x,         head.y + BLOCK)

        dir_l = dir_ == Direction.LEFT
        dir_r = dir_ == Direction.RIGHT
        dir_u = dir_ == Direction.UP
        dir_d = dir_ == Direction.DOWN

        def danger(pt: Point) -> bool:
            return game.is_collision(pt)

        state = [
            # --- Danger in relative directions ---
            # Straight
            (dir_r and danger(pt_r)) or
            (dir_l and danger(pt_l)) or
            (dir_u and danger(pt_u)) or
            (dir_d and danger(pt_d)),

            # Right (clockwise turn from current)
            (dir_u and danger(pt_r)) or
            (dir_d and danger(pt_l)) or
            (dir_l and danger(pt_u)) or
            (dir_r and danger(pt_d)),

            # Left (counter-clockwise turn from current)
            (dir_d and danger(pt_r)) or
            (dir_u and danger(pt_l)) or
            (dir_r and danger(pt_u)) or
            (dir_l and danger(pt_d)),

            # --- Current direction (one-hot) ---
            dir_l, dir_r, dir_u, dir_d,

            # --- Food location relative to head ---
            game.food.x < head.x,   # food is to the left
            game.food.x > head.x,   # food is to the right
            game.food.y < head.y,   # food is above
            game.food.y > head.y,   # food is below
        ]

        return np.array(state, dtype=int)

    # ─────────────────────────────────────────
    # Memory
    # ─────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done) -> None:
        """Push a single transition into the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    # ─────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────

    def train_long_memory(self) -> None:
        """
        Sample a random minibatch from replay memory and train on it.
        Called once per episode (game over).
        """
        if len(self.memory) < BATCH_SIZE:
            batch = list(self.memory)          # train on everything we have
        else:
            batch = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done) -> None:
        """Train immediately on the most recent single transition."""
        self.trainer.train_step(state, action, reward, next_state, done)

    # ─────────────────────────────────────────
    # Action selection (ε-greedy)
    # ─────────────────────────────────────────

    def get_action(self, state: np.ndarray) -> list[int]:
        """
        Epsilon-greedy policy:
          - First ~80 games: heavy exploration (random moves)
          - Gradually shifts to model exploitation as n_games grows
        """
        # ε decays from 80 → 0 over the first 80 games
        self.epsilon = max(0, 80 - self.n_games)
        action = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Explore: random action
            idx = random.randint(0, 2)
        else:
            # Exploit: best action from the model
            import torch
            state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_t)
            idx = int(q_values.argmax().item())

        action[idx] = 1
        return action
