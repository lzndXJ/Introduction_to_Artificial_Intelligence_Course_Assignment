import argparse
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.agent import DQNAgent
from src.replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN on LunarLander-v3.")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default="models/dqn_lunarlander_best.pth")
    parser.add_argument("--checkpoint-dir", type=str, default="models")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--curve-path", type=str, default="models/reward_curve.png")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    # 固定随机种子，尽量保证每次实验结果可复现。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_reward_curve(rewards, curve_path: str) -> None:
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    # 100 局滑动平均能更清楚地反映训练趋势。
    moving_avg = [
        np.mean(rewards[max(0, i - 99) : i + 1]) for i in range(len(rewards))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode reward", alpha=0.5)
    plt.plot(moving_avg, label="100-episode average")
    plt.axhline(200, color="red", linestyle="--", label="Solved threshold")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training on LunarLander-v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 创建 LunarLander-v3 环境。该环境状态空间为 8 维，动作空间为 4 个离散动作。
    env = gym.make("LunarLander-v3")
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
    )
    replay_buffer = ReplayBuffer(args.capacity)

    rewards = []
    # epsilon 控制探索概率，训练初期较大，后期逐渐降低。
    epsilon = args.epsilon_start
    total_steps = 0
    best_avg_reward = -float("inf")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    progress_bar = tqdm(range(1, args.episodes + 1), desc="Training")
    for episode in progress_bar:
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0

        for _ in range(args.max_steps):
            # 1. 根据 epsilon-greedy 策略选择动作。
            action = agent.select_action(state, epsilon)
            # 2. 执行动作，获得下一状态和奖励。
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. 将交互经验放入经验回放池。
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            # 4. 经验数量足够后，开始从回放池采样并更新 Q 网络。
            if len(replay_buffer) >= args.learning_starts:
                agent.learn(replay_buffer, args.batch_size)

            # 5. 每隔固定步数同步一次目标网络，提高训练稳定性。
            if total_steps % args.target_update == 0:
                agent.update_target()

            if done:
                break

        rewards.append(episode_reward)
        # 每个 episode 结束后降低探索率，但不低于 epsilon_end。
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        avg_reward = float(np.mean(rewards[-100:]))

        # 保存最近 100 局平均得分最高的模型。
        if episode >= 100 and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save(args.model_path)

        # 按间隔保存阶段模型，方便后续比较 ep100、ep200 等不同训练阶段。
        if args.save_interval > 0 and episode % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"dqn_lunarlander_ep{episode}.pth"
            )
            agent.save(checkpoint_path)

        progress_bar.set_postfix(
            reward=f"{episode_reward:.1f}",
            avg100=f"{avg_reward:.1f}",
            epsilon=f"{epsilon:.3f}",
        )

        # LunarLander 通常以最近 100 局平均奖励达到 200 作为解决标准。
        if episode >= 100 and avg_reward >= 200:
            print(f"Solved at episode {episode}, average reward: {avg_reward:.2f}")
            agent.save(args.model_path)
            break

    if not os.path.exists(args.model_path):
        agent.save(args.model_path)

    final_model_path = os.path.join(args.checkpoint_dir, "dqn_lunarlander_final.pth")
    agent.save(final_model_path)

    save_reward_curve(rewards, args.curve_path)
    env.close()
    print(f"Best model saved to {args.model_path}")
    print(f"Final model saved to {final_model_path}")
    print(f"Reward curve saved to {args.curve_path}")


if __name__ == "__main__":
    main()
