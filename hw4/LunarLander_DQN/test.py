import argparse

import gymnasium as gym
import numpy as np
import torch

from src.agent import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained DQN agent.")
    parser.add_argument("--model-path", type=str, default="models/dqn_lunarlander_best.pth")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加上 --render 时会打开可视化窗口，否则只在终端输出测试分数。
    render_mode = "human" if args.render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim, device)
    # 加载训练阶段保存的模型参数。
    agent.load(args.model_path)

    rewards = []
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        done = False

        while not done:
            # 测试阶段不再随机探索，直接选择 Q 值最大的动作。
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode {episode}: reward = {episode_reward:.2f}")

    env.close()
    print(f"Average reward over {args.episodes} episodes: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    main()
