import torch
from torch import nn


class QNetwork(nn.Module):
    """DQN 中的 Q 网络：输入状态，输出每个动作对应的 Q 值。"""

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 512):
        super().__init__()
        # LunarLander-v3 的状态维度为 8，离散动作数为 4。
        # 网络最后一层输出 4 个 Q 值，分别对应 4 个可选动作。
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
