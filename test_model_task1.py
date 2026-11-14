"""Evaluate a Task 1 (CartPole-v1) snapshot over seeds 0..19, as in the Lab 5 evaluation protocol.

Usage:
    python test_model_task1.py --model-path LAB5_115550809_task1.pt
    python test_model_task1.py --model-path LAB5_115550809_task1.pt --render --episodes 3   # watch it play
"""
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    """Same architecture as the CartPole DQN in dqn.py (4 -> 64 -> 64 -> 2)."""

    def __init__(self, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # render_mode="human" opens Gymnasium's pygame window and plays at real-time speed
    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)

    model = DQN(env.action_space.n).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    rewards = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        obs, _ = env.reset(seed=seed)
        done, total_reward = False, 0.0
        while not done:
            state = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        rewards.append(total_reward)
        print(f"seed: {seed}, eval reward: {total_reward:.0f}")

    print(f"Average reward: {np.mean(rewards):.2f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--render", action="store_true", help="Open a window and watch the agent play")
    evaluate(parser.parse_args())
