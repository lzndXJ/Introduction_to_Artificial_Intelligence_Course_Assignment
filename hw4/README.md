# 基于 DQN 的 LunarLander-v3 智能体训练

本项目使用 DQN（Deep Q-Network）算法训练 Gymnasium 中的 LunarLander-v3 环境。LunarLander-v3 的状态空间为 8 维连续向量，动作空间为 4 个离散动作，适合使用 DQN 进行训练。

## 1. 环境配置

本项目使用 Anaconda 管理 Python 环境。如果已经有安装好 PyTorch 的 conda 环境，可以直接激活原有环境，例如：

```bash
conda activate torch
```

如果没有合适的环境，可以新建环境：

```bash
conda create -n rl python=3.9.19
conda activate rl
```

安装依赖：

```bash
pip install torch numpy matplotlib tqdm gymnasium[box2d]
```

检查 LunarLander-v3 是否安装成功：

```bash
python -c "import gymnasium as gym; env = gym.make('LunarLander-v3'); print(env.observation_space); print(env.action_space); env.close()"
```

如果输出中包含 `Box(..., (8,), float32)` 和 `Discrete(4)`，说明环境配置成功。

## 2. 项目结构

```text
LunarLander_DQN/          # 项目根目录
├── train.py              # 训练脚本，负责训练 DQN 智能体、保存模型和生成奖励曲线
├── test.py               # 测试脚本，负责加载模型、输出测试得分和显示可视化窗口
├── models/               # 训练后自动生成，用于保存模型文件和奖励曲线
└── src/                  # 核心源代码目录
    ├── dqn.py            # 定义 Q 网络结构，输入 8 维状态，输出 4 个动作 Q 值
    ├── replay_buffer.py  # 实现经验回放池，保存并随机采样训练经验
    └── agent.py          # 实现 DQN 智能体，包括动作选择、网络更新和目标网络同步
```

说明：`models/` 文件夹是训练产物，不是手动编写的代码文件，可以不随代码一起提交。重新运行 `train.py` 时会自动创建该文件夹并保存模型。

## 3. 训练模型

进入项目目录：

```bash
cd LunarLander_DQN
```

开始训练：

```bash
python train.py
```

训练完成后，模型会保存到：

```text
models/dqn_lunarlander_best.pth
```

训练奖励曲线会保存到：

```text
models/reward_curve.png
```

默认每 100 个 episode 还会额外保存一次模型，用于观察不同训练阶段的效果：

```text
models/dqn_lunarlander_ep100.pth
models/dqn_lunarlander_ep200.pth
...
models/dqn_lunarlander_final.pth
```

也可以指定训练轮数：

```bash
python train.py --episodes 1000
```

也可以修改模型保存间隔，例如每 50 个 episode 保存一次：

```bash
python train.py --episodes 800 --save-interval 50
```

## 4. 测试模型

测试训练好的模型：

```bash
python test.py
```

测试指定轮数保存的模型：

```bash
python test.py --model-path models/dqn_lunarlander_ep300.pth
```

显示可视化窗口：

```bash
python test.py --render
```

默认测试 10 个 episode，并输出每个 episode 的奖励和平均奖励。

## 5. 主要参数

常用训练参数如下：

```text
--episodes         训练的 episode 数量
--batch-size       每次从经验回放池中采样的样本数量
--capacity         经验回放池容量
--learning-starts  开始训练前先收集的经验数量
--target-update    目标网络同步间隔
--lr               学习率
--gamma            折扣因子
--epsilon-start    初始探索率
--epsilon-end      最小探索率
--epsilon-decay    探索率衰减系数
--save-interval    每隔多少个 episode 保存一次模型
```
