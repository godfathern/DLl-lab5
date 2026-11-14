# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions, atari=False, dueling=False):
        super(DQN, self).__init__()
        # An example:
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.atari = atari
        if atari:
            # Pong: CNN over 4 stacked 84x84 grayscale frames
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        else:
            # CartPole: MLP over the 4-dim state
            self.network = nn.Sequential(
               nn.Linear(4, 64),
               nn.ReLU(),
               nn.Linear(64, 64),
               nn.ReLU(),
               nn.Linear(64, num_actions)
            )
        ########## END OF YOUR CODE ##########

        # Dueling networks: slides 70-71 / DL-7 (Wang et al., ICML 2016)
        # Keep the shared feature layers, replace the single Q output by a value head V(s) and an advantage head A(s, a)
        self.dueling = dueling
        if dueling:
            if atari:
                # Shared conv trunk up to Flatten (3136 features), then two 512-unit streams as in Wang et al.
                self.network = nn.Sequential(*list(self.network.children())[:-3])
                self.value_head = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, 1))
                self.advantage_head = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU(), nn.Linear(512, num_actions))
            else:
                # Shared two hidden layers (64 features), then one linear layer per stream
                self.network = nn.Sequential(*list(self.network.children())[:-1])
                self.value_head = nn.Linear(64, 1)
                self.advantage_head = nn.Linear(64, num_actions)

    def forward(self, x):
        if self.atari:
            x = x / 255.0  # uint8 pixels -> [0, 1]
        if not self.dueling:
            return self.network(x)
        h = self.network(x)
        value = self.value_head(h)            # V(s): how good the state is
        advantage = self.advantage_head(h)    # A(s, a): how much better each action is
        # Q = V + A - mean_a A  (slide 70, the identifiable version)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        # Stores p_i^alpha (already exponentiated) for each slot of the ring buffer
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.eps = eps              # keeps every priority > 0 so no transition is never sampled
        self.max_priority = 1.0     # largest raw priority |delta| + eps seen so far

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error=None):
        ########## YOUR CODE HERE (for Task 3) ##########
        # New transitions have no TD error yet: give them the max priority so each is replayed soon
        priority = self.max_priority if error is None else abs(error) + self.eps
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition          # overwrite the oldest (FIFO ring buffer)
        self.priorities[self.pos] = priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ##########
        return
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        # PER
        # P(i) = p_i^alpha / sum_k p_k^alpha, slide 62 / DL-7.
        n = len(self.buffer)
        scaled = self.priorities[:n].astype(np.float64)
        cumsum = np.cumsum(scaled)
        total = cumsum[-1]
        targets = (np.arange(batch_size) + np.random.rand(batch_size)) * (total / batch_size)
        indices = np.minimum(np.searchsorted(cumsum, targets, side="right"), n - 1)

        # Importance-sampling weights w_i = (N * P(i))^-beta, normalized so the largest is 1
        probs = scaled[indices] / total
        weights = (n * probs) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        ########## END OF YOUR CODE (for Task 3) ##########
        return batch, indices, weights.astype(np.float32)
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        priorities = np.abs(errors) + self.eps
        self.priorities[indices] = priorities ** self.alpha
        self.max_priority = max(self.max_priority, float(priorities.max()))
        ########## END OF YOUR CODE (for Task 3) ##########
        return
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.atari = env_name.startswith("ALE/")
        # Separate frame stacks for the training and evaluation envs so evaluation can't clobber training state
        self.preprocessor = AtariPreprocessor()
        self.test_preprocessor = AtariPreprocessor()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")   # Apple-silicon GPU
        else:
            self.device = torch.device("cpu")
        print("Using device:", self.device)


        self.q_net = DQN(self.num_actions, atari=self.atari, dueling=args.dueling).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, atari=self.atari, dueling=args.dueling).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21 if self.atari else 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        self.eval_episodes = args.eval_episodes
        self.eval_freq = args.eval_freq
        self.train_freq = args.train_freq

        # Task 3 enhancements (all off by default, so Tasks 1-2 behave exactly as before)
        self.double_dqn = args.double_dqn
        self.use_per = args.per
        self.n_step = args.n_step
        self.per_beta_start = args.per_beta_start
        self.per_beta_steps = args.per_beta_steps
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta=args.per_beta_start)
        else:
            self.memory = deque(maxlen=args.memory_size)
        self.n_step_buffer = deque(maxlen=self.n_step)   # last n raw transitions of the current episode

        # Snapshots at fixed environment steps (Task 3 grading) and an optional step budget
        self.snapshot_steps = {int(s) for s in args.snapshot_steps.split(",") if s.strip()}
        self.max_env_steps = args.max_env_steps
        os.makedirs(self.save_dir, exist_ok=True)

    def push_memory(self, transition):
        if self.use_per:
            self.memory.add(transition)       # new transitions get max priority
        else:
            self.memory.append(transition)

    def n_step_transition(self):
        # Slide 66-69 / DL-7
        # Turn the n_step_buffer into one transition starting at its oldest entry:
        # (s_t, a_t, R = sum_k gamma^k r_{t+k}, s_{t+k}, done_{t+k}, gamma^k), k = len(buffer).
        # Episodes are flushed at their end, so a terminal step can only be the last entry.
        ret = sum((self.gamma ** k) * t[2] for k, t in enumerate(self.n_step_buffer))
        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        next_state, terminated = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        return (state, action, ret, next_state, terminated, self.gamma ** len(self.n_step_buffer))

    def store_transition(self, state, action, reward, next_state, terminated, episode_end):
        # With n_step = 1 this stores (s, a, r, s', terminated, gamma): plain 1-step DQN
        self.n_step_buffer.append((state, action, reward, next_state, terminated))
        if len(self.n_step_buffer) == self.n_step:
            self.push_memory(self.n_step_transition())
        if episode_end:
            # The last n-1 starting points never collect n rewards: store them with shorter returns
            if len(self.n_step_buffer) == self.n_step:
                self.n_step_buffer.popleft()          # already stored just above
            while self.n_step_buffer:
                self.push_memory(self.n_step_transition())
                self.n_step_buffer.popleft()

    def to_state(self, obs, reset, preprocessor):
        # Atari: grayscale/resize/stack frames (uint8, shape 4x84x84). CartPole: raw 4-dim state.
        if not self.atari:
            return obs
        return preprocessor.reset(obs) if reset else preprocessor.step(obs)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def save_snapshot(self):
        path = os.path.join(self.save_dir, f"snapshot_{self.env_count}.pt")
        torch.save(self.q_net.state_dict(), path)
        print(f"Saved snapshot at {self.env_count} env steps to {path}")

    def run(self, episodes=1000):
        budget_reached = False
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = self.to_state(obs, True, self.preprocessor)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps and not budget_reached:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self.to_state(next_obs, False, self.preprocessor)
                # Store `terminated`, not `done`: hitting the 500-step time limit (truncated)
                # is not a real terminal state, so its target must still bootstrap from s'
                episode_end = done or step_count + 1 >= self.max_episode_steps
                self.store_transition(state, action, reward, next_state, terminated, episode_end)

                if self.env_count % self.train_freq == 0:
                    for _ in range(self.train_per_step):
                        self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count in self.snapshot_steps:
                    self.save_snapshot()
                if self.max_env_steps and self.env_count >= self.max_env_steps:
                    budget_reached = True

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed
                    if self.use_per:
                        wandb.log({"Env Step Count": self.env_count, "PER Beta": self.memory.beta})
                    ########## END OF YOUR CODE ##########
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % self.eval_freq == 0 or budget_reached:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

            if budget_reached:
                print(f"Reached the budget of {self.max_env_steps} env steps, stopping.")
                break

    def evaluate(self):
        # Average over several greedy episodes so "best model" isn't picked from one lucky run
        episode_rewards = []
        for _ in range(self.eval_episodes):
            obs, _ = self.test_env.reset()
            state = self.to_state(obs, True, self.test_preprocessor)
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self.to_state(next_obs, False, self.test_preprocessor)
            episode_rewards.append(total_reward)

        return float(np.mean(episode_rewards))


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        # group all states together, all actions togther and so on
        if self.use_per:
            # Anneal beta linearly to 1 so the IS correction becomes exact late in training
            progress = min(1.0, self.env_count / self.per_beta_steps)
            self.memory.beta = self.per_beta_start + (1.0 - self.per_beta_start) * progress
            batch, indices, weights = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
        # discounts = gamma^k, where k is the number of rewards summed in the (multi-step) return
        states, actions, rewards, next_states, dones, discounts = zip(*batch)

        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        
        #for a batch of 32, the network outputs shape [32, 2] - one value for pushing left and one for pushing right
        # Move raw arrays (uint8 for Atari) to the device first, then convert: 4x less data to copy
        states = torch.from_numpy(np.array(states)).to(self.device).float()
        next_states = torch.from_numpy(np.array(next_states)).to(self.device).float()
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        discounts = torch.tensor(discounts, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Double DQN : slides 34-35-67 / DL-7
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN (handout eq. 2): online net selects a', target net evaluates it
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(dim=1).values
            # Bellman target (DL-7 / slide 55); discounts = gamma for 1-step, gamma^n for n-step returns
            targets = rewards + discounts * next_q_values * (1 - dones)

        td_errors = targets - q_values
        if self.use_per:
            # PER : slide 68 / DL-7 + handout (IS weights, Schaul et al. 2016)
            # IS-weighted squared Bellman error, then refresh the sampled transitions' priorities
            weights = torch.from_numpy(weights).to(self.device)
            loss = (weights * td_errors.pow(2)).mean()
            self.memory.update_priorities(indices, td_errors.detach().abs().cpu().numpy())
        else:
            # DQN loss (handout eq. 1): squared Bellman error between prediction and frozen target
            loss = nn.functional.mse_loss(q_values, targets)

        # Gradient update on q_net only (target_net is never stepped)
        self.optimizer.zero_grad()   # clear gradients from the previous update
        loss.backward()              # backprop: d(loss)/d(weights)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)  # guard against exploding updates
        self.optimizer.step()        # Adam moves the weights
        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1", help='"CartPole-v1" (Task 1) or "ALE/Pong-v5" (Tasks 2-3)')
    parser.add_argument("--wandb-project", type=str, default=None, help="Defaults to a per-environment project")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5, help="Greedy episodes averaged per evaluation")
    parser.add_argument("--train-freq", type=int, default=1, help="Run training every N environment steps")
    parser.add_argument("--eval-freq", type=int, default=20, help="Evaluate every N training episodes")
    # Task 3: enhancements
    parser.add_argument("--double-dqn", action="store_true", help="Use the Double DQN target")
    parser.add_argument("--per", action="store_true", help="Use prioritized experience replay")
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-steps", type=int, default=1000000, help="Env steps to anneal beta to 1")
    parser.add_argument("--n-step", type=int, default=1, help="Multi-step return length (1 = vanilla)")
    parser.add_argument("--dueling", action="store_true", help="Dueling network head: Q = V + A - mean(A)")
    parser.add_argument("--snapshot-steps", type=str, default="", help='e.g. "600000,1000000,1500000,2000000,2500000"')
    parser.add_argument("--max-env-steps", type=int, default=0, help="Stop after this many env steps (0 = no limit)")
    args = parser.parse_args()

    project = args.wandb_project or ("DLP-Lab5-DQN-Pong" if args.env.startswith("ALE/") else "DLP-Lab5-DQN-CartPole")
    wandb.init(project=project, name=args.wandb_run_name, config=vars(args), save_code=True)
    agent = DQNAgent(env_name=args.env, args=args)
    agent.run(episodes=args.episodes)
