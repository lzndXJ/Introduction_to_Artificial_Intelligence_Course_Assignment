from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """经验回放池：保存智能体与环境交互产生的 transition。"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done) -> None:
        # 一条经验由 (当前状态, 动作, 奖励, 下一状态, 是否结束) 组成。
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        # 随机采样可以打乱样本之间的时间相关性，使 DQN 训练更稳定。
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
