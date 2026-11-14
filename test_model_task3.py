"""Evaluate a Task 3 (enhanced DQN on ALE/Pong-v5) snapshot over seeds 0..19.

The Q-network is the Task 2 CNN, optionally with a dueling head (detected from the weights);
Double DQN, PER, multi-step returns and soft target updates only change training.

Usage:
    python test_model_task3.py --model_path LAB5_115550809_task3_600000.pt
    python test_model_task3.py --model_path LAB5_115550809_task3_best.pt --env-steps 1234567
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions, dueling=False):
        super(DQN, self).__init__()
        trunk = [
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ]
        self.dueling = dueling
        if dueling:
            # Same layout as dqn.py --dueling: shared conv trunk, then V(s) and A(s, a) streams
            self.network = nn.Sequential(*trunk)
            self.value_head = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, 1))
            self.advantage_head = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions))
        else:
            self.network = nn.Sequential(*trunk, nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions))

    def forward(self, x):
        h = self.network(x / 255.0)
        if not self.dueling:
            return h
        advantage = self.advantage_head(h)
        return self.value_head(h) + advantage - advantage.mean(dim=1, keepdim=True)
class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        stacked = np.stack(self.frames, axis=0)
        return stacked
        
def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    state_dict = torch.load(args.model_path, map_location=device)
    dueling = any(k.startswith("value_head.") for k in state_dict)   # detect the architecture from the weights
    model = DQN(4, num_actions, dueling=dueling).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)

    rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []
        frame_idx = 0

        while not done:
            if args.save_video:
                frames.append(env.render())

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)
            frame_idx += 1

        rewards.append(total_reward)
        print(f"Environment steps: {args.env_steps}, seed: {args.seed + ep}, eval reward: {total_reward:.0f}")
        if args.save_video:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
            with imageio.get_writer(out_path, fps=30) as video:
                for f in frames:
                    video.append_data(f)
            print(f"Saved video → {out_path}")

    print(f"Average reward: {np.mean(rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--model_path", dest="model_path", type=str, required=True,
                        help="Path to trained .pt model, e.g. LAB5_115550809_task3_600000.pt")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="First evaluation seed (episodes use seed, seed+1, ...)")
    parser.add_argument("--save-video", action="store_true", help="Also record an mp4 per episode")
    parser.add_argument("--env-steps", type=str, default=None,
                        help="Training env steps of this snapshot (printed only); read from the file name if omitted")
    args = parser.parse_args()
    if args.env_steps is None:
        # LAB5_<id>_task3_600000.pt -> 600000; LAB5_<id>_task3_best.pt -> "best"
        args.env_steps = os.path.splitext(os.path.basename(args.model_path))[0].split("_")[-1]
    evaluate(args)
