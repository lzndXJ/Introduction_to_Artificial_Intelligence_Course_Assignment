import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.dqn import QNetwork


class DQNAgent:
    """DQN 智能体：包含策略网络、目标网络和 epsilon-greedy 动作选择。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_dim: int = 512,
    ):
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma

        # policy_net 是训练时直接更新的 Q 网络。
        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        # target_net 用于计算目标 Q 值，定期从 policy_net 同步参数。
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.update_target()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, epsilon: float = 0.0) -> int:
        # epsilon-greedy 策略：以 epsilon 的概率随机探索，否则选择 Q 值最大的动作。
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state = np.asarray(state, dtype=np.float32)
        state_tensor = torch.tensor(state, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def learn(self, replay_buffer, batch_size: int) -> float:
        # 从经验回放池中采样一批 transition，用于一次梯度更新。
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        # 取出 policy_net 对“实际执行动作”的 Q(s, a) 估计。
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            # DQN 目标值：r + gamma * max_a' Q_target(s', a')。
            # 如果 episode 已结束，下一状态不再产生未来奖励，因此乘以 (1 - done)。
            next_q_values = self.target_net(next_states).max(dim=1).values
            target_q_values = rewards + self.gamma * (1.0 - dones) * next_q_values

        # Huber loss 对异常 TD 误差更稳健，常用于 DQN。
        loss = F.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target(self) -> None:
        # 将策略网络参数复制到目标网络。
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 作业中只需要保存模型参数即可，测试时再重新创建同结构网络并加载参数。
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.update_target()
        self.policy_net.eval()
        self.target_net.eval()
