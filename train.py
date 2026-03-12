"""
train.py — Training loop for the Snake RL Agent
================================================
Run: python train.py

Controls
--------
- Close the Matplotlib window OR the Pygame window to stop training.
- The best model is auto-saved to ./model/model.pth every time a high score
  is reached.
"""

import matplotlib
matplotlib.use("MacOSX")         # native macOS backend — no Tk/Qt needed
import matplotlib.pyplot as plt
from game  import SnakeGameAI
from agent import Agent

# ──────────────────────────────────────────────
# Live plot helpers
# ──────────────────────────────────────────────

plt.ion()   # interactive mode — updates without blocking

def _setup_plot():
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0f0f19")
    ax.set_facecolor("#0f0f19")
    ax.set_title("Snake — Training Progress", color="white", fontsize=13, pad=10)
    ax.set_xlabel("Games", color="#aaaaaa")
    ax.set_ylabel("Score",  color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    return fig, ax


def plot(scores: list, mean_scores: list, fig, ax) -> None:
    ax.clear()
    ax.set_facecolor("#0f0f19")
    ax.set_title("Snake — Training Progress", color="white", fontsize=13, pad=10)
    ax.set_xlabel("Games",  color="#aaaaaa")
    ax.set_ylabel("Score",  color="#aaaaaa")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    x = list(range(1, len(scores) + 1))
    ax.plot(x, scores,      color="#50c878", linewidth=1.2,
            label="Score",      alpha=0.7)
    ax.plot(x, mean_scores, color="#5599ff", linewidth=2.0,
            label="Mean Score", linestyle="--")

    if scores:
        ax.set_ylim(bottom=0, top=max(max(scores), 1) + 2)
        # Annotate latest values
        ax.annotate(f"{scores[-1]}",
                    xy=(x[-1], scores[-1]),
                    color="#50c878", fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
        ax.annotate(f"{mean_scores[-1]:.1f}",
                    xy=(x[-1], mean_scores[-1]),
                    color="#5599ff", fontsize=8,
                    xytext=(4, -12), textcoords="offset points")

    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9,
              framealpha=0.8)
    fig.canvas.draw()
    fig.canvas.flush_events()


# ──────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────

def train() -> None:
    scores:       list[int]   = []
    mean_scores:  list[float] = []
    total_score:  int         = 0
    record:       int         = 0

    agent = Agent()
    game  = SnakeGameAI(ai_playing=True, render=True)

    fig, ax = _setup_plot()
    plt.show(block=False)

    print("=" * 55)
    print("  Snake RL Training  —  press Ctrl-C to stop")
    print("=" * 55)

    while True:
        # ── Current state ──────────────────────────────────
        state_old = agent.get_state(game)

        # ── Choose action ──────────────────────────────────
        action = agent.get_action(state_old)

        # ── Perform step ──────────────────────────────────
        reward, done, score = game.play_step(action)

        # ── New state ──────────────────────────────────────
        state_new = agent.get_state(game)

        # ── Train on this single step ──────────────────────
        agent.train_short_memory(state_old, action, reward, state_new, done)

        # ── Store in replay buffer ─────────────────────────
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # ── Episode finished: long-memory (replay) train ──
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # ── Record keeping ────────────────────────────
            if score > record:
                record = score
                agent.model.save()

            total_score  += score
            mean_score    = total_score / agent.n_games
            scores.append(score)
            mean_scores.append(round(mean_score, 2))

            print(f"Game {agent.n_games:>4}  |  Score: {score:>3}  |  "
                  f"Record: {record:>3}  |  Mean: {mean_score:>6.2f}  |  "
                  f"ε={agent.epsilon}")

            # ── Update live plot ──────────────────────────
            plot(scores, mean_scores, fig, ax)


if __name__ == "__main__":
    train()
