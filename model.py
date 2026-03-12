"""
model.py — Neural Network and Q-Learning Trainer
=================================================
Linear_QNet : 3-layer fully connected network
QTrainer    : Bellman equation update with Adam + MSE
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """
    Simple 3-layer linear network for Deep Q-Learning.

    Architecture
    ------------
    Input  → 11 neurons  (state vector)
    Hidden → 256 neurons (ReLU)
    Output → 3 neurons   (Q-values for [Straight, Right, Left])
    """

    def __init__(self, input_size: int = 11,
                 hidden_size: int = 256,
                 output_size: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)          # raw Q-values (no activation)

    def save(self, filename: str = "model.pth") -> None:
        """Persist model weights to ./model/<filename>."""
        folder = os.path.join(os.path.dirname(__file__), "model")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)
        torch.save(self.state_dict(), path)
        print(f"[Model] Saved → {path}")

    def load(self, filename: str = "model.pth") -> bool:
        """Load weights if the checkpoint exists. Returns True on success."""
        path = os.path.join(os.path.dirname(__file__), "model", filename)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, weights_only=True))
            self.eval()
            print(f"[Model] Loaded ← {path}")
            return True
        return False


class QTrainer:
    """
    One-step and batch Q-Learning updates.

    Update rule (Bellman equation)
    ------------------------------
    Q_new = reward  +  γ · max( Q(next_state) )
            (only when not game_over; otherwise Q_new = reward)
    """

    def __init__(self, model: Linear_QNet,
                 lr: float = 1e-3,
                 gamma: float = 0.9):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self,
                   state,
                   action,
                   reward,
                   next_state,
                   done) -> None:
        """
        Perform a single gradient update step.

        All inputs can be single samples or batches (numpy arrays / lists).
        """
        state      = torch.tensor(state,      dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action     = torch.tensor(action,     dtype=torch.long)
        reward     = torch.tensor(reward,     dtype=torch.float)

        # Ensure batch dimension exists
        if state.dim() == 1:
            state      = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action     = action.unsqueeze(0)
            reward     = reward.unsqueeze(0)
            done       = (done,)

        # 1. Predicted Q-values for current state
        pred = self.model(state)                      # shape: [B, 3]

        # 2. Compute target Q-values (Bellman)
        target = pred.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i].unsqueeze(0)))
            target[i][torch.argmax(action[i]).item()] = Q_new

        # 3. Gradient step
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
