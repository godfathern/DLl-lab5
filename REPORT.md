# Lab 5: Value-Based Reinforcement Learning

**Course:** 535528 Deep Learning, Fall 2026
**Student:** Phan Tung Bui — Student ID: `115550809`

> Sections marked **TODO** are still to be filled in (Task 3 has not been trained yet).
> Replace every `screenshots/...png` placeholder with your own image (create a `screenshots/` folder next to this file, or change the paths).

---

## 1. Introduction

This lab implements Deep Q-Networks (DQN) and evaluates them on two environments of increasing difficulty:

| Task | Environment | Observation | Q-network | Goal |
|---|---|---|---|---|
| 1 | CartPole-v1 | 4-dim state vector | MLP | average score ≥ 480 over 20 episodes |
| 2 | ALE/Pong-v5 | stacked 84×84 grayscale frames | CNN | average score ≥ 19 over 20 episodes |
| 3 | ALE/Pong-v5 | same as Task 2 | CNN + Double DQN, PER, multi-step return | reach score 19 in as few environment steps as possible |

**Main findings so far**

- **Task 1:** Vanilla DQN with a 4 → 64 → 64 → 2 MLP solves CartPole. The best snapshot (saved at ≈ 95.7k environment steps) scores **500.00 on all 20 evaluation seeds (0–19)**, the maximum possible.
- **Hyperparameters matter a lot.** With the same algorithm, configurations ranged from never learning at all to solving the task reliably (Section 5.1). The learning rate was the most important setting: 1e-3 learned fast but kept collapsing, 1e-4 was stable but too slow, and 5e-4 worked best.
- **Time-limit truncation must not be treated as termination.** CartPole cuts episodes off at 500 steps; storing that as a terminal transition is a known source of biased targets, so only real terminations zero out the bootstrap term (Section 2.1).
- **Task 2:** Vanilla DQN with the Nature-DQN CNN learns Pong from pixels. The 3-game evaluation average first reached 19 at ≈ 0.95M environment steps. The submitted snapshot (≈ 3.09M steps) averages **19.10 over seeds 0–19**.
- **Model selection needs a reliable evaluation.** On Pong, the "best" model chosen during training from a 3-game average scored only 17.40 on the 20-seed test, while an earlier checkpoint scored 19.10 (Section 4.2).
- **TODO:** Task 3 findings.

**Organization.** Section 2 describes the implementation (Bellman error, Double DQN, PER, multi-step return, W&B logging). Section 3 lists the evaluation commands and screenshots. Section 4 shows training curves and analysis. Section 5 covers additional experiments (bonus), including the CartPole hyperparameter study.

---

## 2. Implementation

### 2.1 Vanilla DQN and the Bellman error (Task 1)

**Concept.** DQN approximates the action-value function Q(s, a), the expected discounted return from taking action *a* in state *s* and acting greedily afterwards. It minimizes the squared Bellman error (handout eq. 1):

$$
L_{\text{DQN}}(\theta) = \frac{1}{2}\sum_{(s,a,r,s')\in D}\Big(r + \gamma\max_{a'}Q(s',a';\bar\theta) - Q(s,a;\theta)\Big)^2
$$

Two techniques stabilize training (DL-7, slides 52–55):

- **Experience replay.** Transitions (s, a, r, s′, terminated) are stored in a FIFO buffer and mini-batches are sampled uniformly. This breaks the correlation between consecutive samples and reuses data.
- **Target network.** The target uses a separate, periodically synced copy of the network (θ̄), so the regression target does not move every time θ is updated.

**Q-network (CartPole).** A fully connected network: 4 → 64 → 64 → 2, ReLU activations, no activation on the output layer (Q-values are unbounded). It has 4,610 parameters.

```python
self.network = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, num_actions)
)
```

**Observation handling.** CartPole's observation is already a compact 4-dim state containing velocities, so the `AtariPreprocessor` (grayscale / resize / frame stack) is bypassed and the raw observation is used as the state.

**Replay buffer.** `collections.deque(maxlen=memory_size)`; mini-batches are drawn with `random.sample(self.memory, batch_size)`.

**Computing the Bellman error.** For a sampled batch, the prediction is the Q-value of the action actually taken, and the target is computed with the frozen target network without tracking gradients:

```python
q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

with torch.no_grad():
    next_q_values = self.target_net(next_states).max(dim=1).values
    targets = rewards + self.gamma * next_q_values * (1 - dones)

loss = nn.functional.mse_loss(q_values, targets)
self.optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
self.optimizer.step()
```

The Bellman error is `targets - q_values`. The `(1 - dones)` factor removes the bootstrap term for terminal transitions. Gradient-norm clipping (max 10) guards against occasional very large updates. Every `target_update_frequency` updates, the target network is synced with `target_net.load_state_dict(q_net.state_dict())`.

**Terminated vs. truncated.** Gymnasium returns two flags: `terminated` (the pole fell, a true terminal state) and `truncated` (the 500-step time limit). The episode loop ends on either, but the replay buffer stores only `terminated`:

```python
done = terminated or truncated          # ends the episode loop
self.memory.append((state, action, reward, next_state, terminated))  # bootstrap flag
```

If truncation were stored as terminal, a well-balanced state at step 500 would get target `r` instead of `r + γ max Q(s′)`. Since the state does not include the time step, the network would receive contradictory targets for near-identical states.

**ε-greedy exploration.** With probability ε a random action is taken, otherwise argmax Q. ε is multiplied by `epsilon_decay` after every update, from 1.0 down to `epsilon_min = 0.05`. With decay 0.9995, ε reaches 0.05 after ≈ 6,000 updates.

**Evaluation and model selection.** Every 20 training episodes, the greedy policy (ε = 0) is evaluated over 5 episodes and the **average** is used. The model is saved as `best_model.pt` whenever this average improves. Averaging several episodes avoids saving a model that scored 500 once by luck.

### 2.2 Visual observations on Atari (Task 2)

The same agent class handles both tasks: `--env ALE/Pong-v5` sets `self.atari = True`, which switches the network to a CNN and turns the preprocessor on. The training loop, replay buffer, target network and loss are unchanged from Task 1.

**Environment.** `ALE/Pong-v5` from `ale-py` (Arcade Learning Environment, which emulates the Atari 2600 and runs the original Pong ROM). The v5 defaults are kept: frame skip 4 (each action is repeated for 4 emulator frames) and sticky actions with probability 0.25 (the emulator repeats the previous action 25% of the time, so the agent cannot memorize a fixed action sequence). The action space is Pong's minimal set of 6 actions: NOOP, FIRE, RIGHT (paddle up), LEFT (paddle down), RIGHTFIRE and LEFTFIRE. An episode is one full game to 21 points, so returns range from −21 to +21.

**Preprocessing (`AtariPreprocessor`).** Each 210×160×3 RGB frame is:

1. converted to grayscale (color carries no information in Pong),
2. resized to 84×84 with area interpolation,
3. stacked with the previous 3 frames, giving a state of shape 4×84×84.

A single frame shows where the ball is but not where it is going. Stacking 4 frames lets the network infer the ball's direction and speed. On reset, the first frame is repeated 4 times. Training and evaluation use separate preprocessor instances so that evaluation episodes do not corrupt the training frame stack.

**Memory.** States are stored in the replay buffer as `uint8` (28 KB per state) and converted to float only when a mini-batch is sampled. With 100k transitions (each holding s and s′) the buffer uses about 5.6 GB, compared with about 22 GB as float32.

**CNN architecture.** The convolutional network from the Nature DQN paper (Mnih et al., 2015). Pixel values are divided by 255 inside `forward()` so the input is in [0, 1]:

```python
self.network = nn.Sequential(
    nn.Conv2d(4, 32, kernel_size=8, stride=4),   # 4×84×84  -> 32×20×20
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 32×20×20 -> 64×9×9
    nn.ReLU(),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 64×9×9   -> 64×7×7
    nn.ReLU(),
    nn.Flatten(),                                # 3136
    nn.Linear(64 * 7 * 7, 512),
    nn.ReLU(),
    nn.Linear(512, num_actions)                  # 6 Q-values
)

def forward(self, x):
    if self.atari:
        x = x / 255.0
    return self.network(x)
```

| Layer | Output shape | Parameters |
|---|---|---|
| Conv 8×8, stride 4, 32 filters | 32×20×20 | 8,224 |
| Conv 4×4, stride 2, 64 filters | 64×9×9 | 32,832 |
| Conv 3×3, stride 1, 64 filters | 64×7×7 | 36,928 |
| Linear 3136 → 512 | 512 | 1,606,144 |
| Linear 512 → 6 | 6 | 3,078 |
| **Total** | | **1,687,206** (6.75 MB) |

All Conv and Linear layers use Kaiming-uniform initialization (suited to ReLU) with zero biases.

**Other choices.**

- **Rewards** are used as given (no clipping or scaling needed, since Pong only gives −1, 0 or +1 per step).
- **Terminal flag.** As in Task 1, only `terminated` (game over) zeroes the bootstrap term.
- **Exploration.** ε decays from 1.0 by ×0.999988 per update and reaches the floor of 0.02 after ≈ 326k updates (≈ 376k environment steps).
- **Evaluation during training.** Every 20 games, 3 greedy games (ε = 0) are played on a separate environment and their average is logged as `Eval Reward`. A checkpoint is also saved every 100 games.
- **Device.** Training ran on an Apple-silicon GPU (PyTorch `mps` backend). The code picks `cuda`, then `mps`, then `cpu`.

### 2.3 Double DQN (Task 3) — TODO

**Concept.** The max in the DQN target uses the same network to select and evaluate the next action, which overestimates Q-values. Double DQN selects with the online network and evaluates with the target network (handout eq. 2):

$$
y = r + \gamma\, Q\big(s', \arg\max_{a'} Q(s',a';\theta);\ \bar\theta\big)
$$

**Implementation:** **TODO** (code snippet)

### 2.4 Prioritized Experience Replay (Task 3) — TODO

**Concept.** Transition *i* gets priority p_i = |δ_i| + ε and is sampled with probability P(i) = p_i^α / Σ_k p_k^α. Importance-sampling weights w_i = (N · P(i))^(−β), normalized by their maximum, correct the bias.

**Implementation:** **TODO** (data structure, how priorities are updated, β schedule)

### 2.5 Multi-step return (Task 3) — TODO

**Concept.** Replace the 1-step target with the n-step return:

$$
R_t^{(n)} = \sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n},a')
$$

**Implementation:** **TODO** (n, how the n-step buffer handles episode ends)

### 2.6 Weights & Biases tracking

Training is logged with `wandb.log(...)`:

| Metric | When | Meaning |
|---|---|---|
| `Total Reward` | every episode | training-episode return (includes ε-random actions) |
| `Eval Reward` | every 20 episodes | average greedy return over `--eval-episodes` games (5 for CartPole, 3 for Pong) |
| `Epsilon` | every episode and every 1,000 steps | current exploration rate |
| `Env Step Count` | with every log | total environment steps so far (**x-axis of all plots**) |
| `Update Count` | with every log | number of gradient updates |

All plots use `Env Step Count` as the x-axis, as required.

---

## 3. Evaluation Results (Snapshots)

### 3.1 Task 1 — CartPole-v1

**Command**

```bash
python test_model_task1.py --model-path LAB5_115550809_task1.pt
```

`test_model_task1.py` loads the snapshot, runs 20 greedy episodes with seeds 0–19 and prints the average reward.

**Result: average reward 500.00 over 20 episodes** (every seed scored 500).

![Task 1 evaluation output (seeds 0–19)](screenshots/task1_eval.png)

**How this snapshot was obtained.** The template's default hyperparameters did not get there: with ε decay 0.999999 and a 50k-step replay warm-up, the agent stays almost fully random for the whole CartPole training budget. Reaching the maximum score of 500 took a systematic search:

- **8 configurations** (A–H) in two rounds, varying the learning rate (1e-4 to 1e-3), target-network update period (200 to 2,000 updates), batch size, replay warm-up and ε decay;
- a **fix to the replay targets**, so that time-limit truncation at step 500 is not stored as a terminal state (Section 2.1);
- **averaged evaluation during training** (5 greedy episodes instead of 1), so `best_model.pt` is not chosen from a single lucky 500;
- the best snapshots of the round-2 candidates **re-checked on seeds 0–19**. Only configuration E (lr 5e-4, target update 500, batch 64) was both stable and perfect, and it was then retrained with W&B logging as the final run.

Results ranged from runs that never learned (best eval 14) to runs that hit 500 but repeatedly collapsed. The full comparison is in **Section 5.1**, and the final training curve is in **Section 4.1**.

### 3.2 Task 2 — Pong-v5

**Command**

```bash
python test_model.py --model-path LAB5_115550809_task2.pt --env-steps 3092538
```

`test_model.py` builds the same CNN and preprocessing as training, plays 20 greedy games with seeds 0–19 (`env.reset(seed=seed + ep)`), and prints each score and the average. `--env-steps` is only printed, to label the snapshot. Adding `--save-video` also writes one mp4 per game to `./eval_videos/` (used for the demo video).

**Snapshot:** `LAB5_115550809_task2.pt`, the checkpoint saved after training game 1400, at **3,092,538 environment steps**.

**Result: average reward 19.10 over 20 games** (minimum 16, maximum 21; 13 of 20 games scored 19 or more).

![Task 2 evaluation output (seeds 0–19)](screenshots/task2_eval.png)

### 3.3 Task 3 — Enhanced DQN on Pong-v5 — TODO

> **Provisional:** these are the run 1 snapshots (Double DQN + PER + 3-step, Section 4.3). Run 2 (Section 4.3) scored lower at every snapshot, so run 1 is kept unless a later run does better.

| Snapshot | Env steps | Average score (20 episodes) | Screenshot |
|---|---|---|---|
| `LAB5_115550809_task3_600000.pt` | 600k | 13.05 | ![](screenshots/task3_600k.png) |
| `LAB5_115550809_task3_1000000.pt` | 1M | 11.05 | ![](screenshots/task3_1M.png) |
| `LAB5_115550809_task3_1500000.pt` | 1.5M | 13.50 | ![](screenshots/task3_1.5M.png) |
| `LAB5_115550809_task3_2000000.pt` | 2M | 13.70 | ![](screenshots/task3_2M.png) |
| `LAB5_115550809_task3_2500000.pt` | 2.5M | 13.55 | ![](screenshots/task3_2.5M.png) |
| `LAB5_115550809_task3_best.pt` | ≈ 2.36M | 15.25 | ![](screenshots/task3_best.png) |

Command: `python test_model_task3.py --model-path LAB5_115550809_task3_600000.pt` (same for each file; seeds 0–19).

**First time reaching score 19:** not reached by any run 1 snapshot within 2.5M steps (best: 15.25 on seeds 0–19).

---

## 4. Analysis and Discussion

### 4.1 Task 1 training curve

W&B run: `task1-final` (project `DLP-Lab5-DQN-CartPole`).

![Task 1: Eval Reward vs Env Step Count](screenshots/task1_wandb_eval.png)

**Hyperparameters (final run)**

| Hyperparameter | Value |
|---|---|
| Learning rate (Adam) | 5e-4 |
| Discount factor γ | 0.99 |
| Batch size | 64 |
| Replay memory size | 50,000 |
| Replay start size | 1,000 |
| ε start / min / decay (per update) | 1.0 / 0.05 / 0.9995 |
| Target network update | every 500 updates |
| Updates per environment step | 1 |
| Training episodes | 800 |
| Evaluation | 5 greedy episodes every 20 training episodes |
| Loss | MSE, gradient-norm clip 10 |

**Learning progress (average of 5 evaluation episodes)**

| Env steps | Eval reward | Phase |
|---|---|---|
| 0 – 2k | ≈ 9 | random play, replay buffer warming up |
| ≈ 7k | 278 | fast initial learning |
| 10k – 72k | 113 – 328 | plateau |
| ≈ 86k | 476 | breakthrough |
| **≈ 96k – 127k** | **500** | solved (best model saved at ≈ 95.7k) |
| ≈ 132k – 142k | 160 – 343 | temporary performance drop |
| ≈ 150k – 160k | 500 | recovered |

**Discussion.**

- The agent learns a decent policy within ≈ 7k steps but then plateaus for ≈ 65k steps before solving the task. During the plateau the agent typically balances for a few hundred steps and then drifts off the track.
- After solving, performance dropped temporarily around 132k–142k steps. DQN keeps updating its weights after reaching the optimum; once the replay buffer is dominated by "balanced" states, the network gets fewer examples of recovering from bad states and can partially forget them. This is why the best snapshot is kept rather than the final one.

### 4.2 Task 2 training curve

W&B run: `task2-vanilla` (project `DLP-Lab5-DQN-Pong`). The run was stopped manually after 2,700 games (≈ 5.91M environment steps).

![Task 2: Eval Reward vs Env Step Count](screenshots/task2_wandb_eval.png)

**Hyperparameters**

| Hyperparameter | Value |
|---|---|
| Learning rate (Adam) | 1e-4 |
| Discount factor γ | 0.99 |
| Batch size | 32 |
| Replay memory size | 100,000 |
| Replay start size | 50,000 |
| ε start / min / decay (per update) | 1.0 / 0.02 / 0.999988 |
| Target network update | every 1,000 updates |
| Updates per environment step | 1 |
| Evaluation | 3 greedy games every 20 training games |
| Loss | MSE, gradient-norm clip 10 |

These are close to the standard Nature-DQN / PyTorch-tutorial settings for Pong. The main change from the provided defaults is a faster ε decay (0.999988 instead of 0.999999), so exploration reaches its floor after ≈ 0.38M steps instead of several million.

**Learning progress (3-game evaluation average)**

| Env steps | Eval reward | Phase |
|---|---|---|
| 0 – 56k | −21 | warm-up: random play fills the replay buffer (no updates before 50k) |
| ≈ 0.1M – 0.42M | −17 → −4 | learns to track and return the ball; games get longer |
| ≈ 0.49M – 0.90M | 8 → 16 | starts winning points consistently |
| **≈ 0.95M** | **19.33** | first evaluation at 19 or above |
| 1M – 5.9M | mostly 15 – 20, dips to ≈ 10 | plateau; best 3-game averages 20.33 at 3.13M steps |

**Model selection.** `best_model.pt` was saved at the highest 3-game average (20.33 at ≈ 3.13M steps). On the 20-seed test, however, it scored only **17.40**. Three games are too few to tell a 17-point policy from a 20-point one. I therefore re-evaluated the saved checkpoints on seeds 0–19:

| Checkpoint | Env steps | 20-seed average |
|---|---|---|
| **ep1400** | **3.09M** | **19.10** |
| best_model (3-game max) | 3.13M | 17.40 |
| ep1600 | ≈ 3.54M | 18.25 |
| ep1800 | ≈ 3.97M | 17.65 |
| ep2000 | ≈ 4.40M | 18.60 |
| ep2100 | ≈ 4.62M | 17.05 |
| ep2200 | ≈ 4.83M | 17.50 |
| ep2300 | ≈ 5.05M | 17.95 |
| ep2400 | ≈ 5.26M | 18.85 |

The ep1400 checkpoint was evaluated twice and gave 19.10 both times, so it is submitted as `LAB5_115550809_task2.pt`.

**Discussion.**

- **Sparse reward, slow start.** Pong gives a reward only when a point ends. Early on, the random agent loses almost every point, and the network first has to learn the link between paddle position and not losing. The large jump between ≈ 0.42M and 0.49M steps (−4 → +8) is where the agent starts returning the ball reliably and winning rallies.
- **The plateau is noisy, not flat.** After ≈ 1M steps the policy keeps changing with every update. Different checkpoints from the plateau score anywhere from 17 to 19 on the same 20 seeds, and single evaluations dip to ≈ 10. Vanilla DQN has no mechanism to lock in a good policy, which is why checkpoints and careful selection matter.
- **Training reward is lower than evaluation reward.** Training games still use ε = 0.02 random actions on top of 25% sticky actions, so training returns stayed around 11–14 while greedy evaluation reached 17–20.
- **The margin is small.** 19.10 is only just above the threshold. The game is stochastic (sticky actions), and GPU kernels on another device (for example CUDA instead of MPS) may produce slightly different Q-values, so a re-evaluation elsewhere could land slightly lower.

### 4.3 Task 3 training curve and sample efficiency — TODO

![Task 3 vs Task 2: Eval Reward vs Env Step Count](screenshots/task3_wandb_eval.png)

#### Run 1: Double DQN + PER + 3-step return on top of the Task 2 setup

W&B run: `task3-full`. I first added the three required techniques to the Task 2 agent without retuning anything else, to see what they give on their own.

| Setting | Run 1 | Task 2 (vanilla) |
|---|---|---|
| Double DQN target | yes | no |
| PER (α / β start → 1 over) | 0.6 / 0.4 → 1 over 1M steps | uniform |
| Multi-step return | n = 3 | n = 1 |
| Learning rate, batch, memory | 1e-4, 32, 100k | same |
| Loss | IS-weighted squared TD error, gradient-norm clip 10 | MSE, gradient-norm clip 10 |
| Target network | hard copy every 1,000 updates | same |
| Replay start size | 20,000 | 50,000 |
| ε decay (per update) / min | 0.99998 / 0.02 | 0.999988 / 0.02 |
| Evaluation during training | 5 games every 10 games | 3 games every 20 games |
| Budget | stopped at 2.5M env steps | 5.9M env steps |

**Result: no snapshot reached 19.** All snapshots were evaluated with `test_model_task3.py` on seeds 0–19:

| Snapshot | 20-seed average | Min / max | Games ≥ 19 |
|---|---|---|---|
| 600k | 13.05 | 8 / 19 | 1 |
| 1M | 11.05 | 3 / 17 | 0 |
| 1.5M | 13.50 | 4 / 21 | 3 |
| 2M | 13.70 | −1 / 20 | 1 |
| 2.5M | 13.55 | −1 / 20 | 4 |
| best (17.6 on 5 games) | 15.25 | 10 / 20 | 2 |

**Comparison with vanilla DQN at the same budget.** Task 2 did not save checkpoints at exactly these step counts, so the nearest Task 2 checkpoints were evaluated with the same script and seeds:

| Env steps | Vanilla (Task 2), 20-seed average | Run 1, 20-seed average |
|---|---|---|
| ≈ 0.6M | 13.80 (ep300, 0.66M) | 13.05 |
| ≈ 1M | 17.60 (ep400, 0.90M) / 16.35 (ep500, 1.12M) | 11.05 |
| ≈ 2.5M | 15.55 (ep1100, 2.44M) / 16.25 (ep1200, 2.66M) | 13.55 |

**Observation.** The naïve combination learned about as fast as vanilla DQN up to ≈ 0.6M steps, then stopped improving at ≈ 11–15 while vanilla kept climbing to 16–18. At ≈ 1M steps it was about 5 points behind. So adding the three techniques on their own did **not** improve sample efficiency in this run; it made the plateau lower.

**Likely causes.**

- **PER feeds the largest errors back on purpose.** Transitions with a large TD error are replayed most often. With a squared loss, those transitions also produce the largest gradients. The IS weights and the gradient clip only partly offset this, so updates stay noisy.
- **The 3-step target is noisier.** It sums three rewards, and in Pong (25% sticky actions) the ball can still be deflected unexpectedly. Its variance adds to the noise above.
- **The hard target copy makes the target jump every 1,000 updates.** Combined with the two points above, the online network keeps chasing a moving, noisy target.
- **Other confounds.** Run 1 also used a smaller replay start size and a faster ε decay than Task 2, and each method was trained only once. Pong results vary between training seeds, so differences of 2–3 points may partly be chance. The ≈ 5-point gap at 1M steps is less likely to be chance alone.

**What I changed for run 2.** DL-7 lists two further ideas that target these issues: a **dueling network head** (slides 70–71), which learns V(s) once per state instead of six separate Q-values, and a **soft target update** (slide 55, "updated only periodically or softly"), which moves the target network a little every update instead of jumping every 1,000 updates. Run 2 keeps every setting of run 1 and adds `--dueling --tau 0.001`.

#### Run 2: + dueling head and soft target update

W&B run: `task3-run2-dueling-soft`. Same settings as run 1, plus `--dueling --tau 0.001`.

**Implementation.**

- **Dueling head.** The convolutional trunk is shared. After `Flatten`, the network splits into a value stream (3136 → 512 → 1) and an advantage stream (3136 → 512 → 6), combined as Q(s, a) = V(s) + A(s, a) − mean_a A(s, a) (slide 70, the identifiable form). This raises the parameter count from 1.69M to 3.29M. `test_model_task3.py` detects the dueling layout from the saved weights.
- **Soft target update.** After every gradient step, w̄ ← τ·w + (1 − τ)·w̄ with τ = 0.001, instead of copying w into w̄ every 1,000 updates.

**Result: worse than run 1 at every snapshot.**

| Snapshot | Run 1, 20-seed average | Run 2, 20-seed average | Run 2 min / max | Run 2 games ≥ 19 |
|---|---|---|---|---|
| 600k | **13.05** | 7.30 | −1 / 16 | 0 |
| 1M | **11.05** | 6.10 | −5 / 17 | 0 |
| 1.5M | **13.50** | 9.80 | −4 / 18 | 0 |
| 2M | **13.70** | 11.50 | −13 / 20 | 2 |
| 2.5M | **13.55** | 12.05 | 3 / 19 | 1 |
| best | **15.25** (≈ 2.36M) | 9.10 (≈ 1.43M, 16.6 on 5 games) | −6 / 17 | 0 |

The training evaluations (5-game averages) agree:

| Env steps | Run 1 | Run 2 |
|---|---|---|
| 0 – 0.5M | −11.1 | −9.1 |
| 0.5 – 1.0M | 11.1 | 7.6 |
| 1.0 – 1.5M | 10.7 | 8.6 |
| 1.5 – 2.0M | 13.5 | 12.0 |
| 2.0 – 2.5M | 14.2 | 12.1 |

Run 2 was slightly ahead during the first ≈ 0.4M steps, then fell behind once the agent started winning points, and never caught up.

**Why it did not help (likely).**

- **τ = 0.001 makes the target slower, not just smoother.** The target is an exponential average of past networks with a time constant of ≈ 1/τ = 1,000 updates, so it always lags behind the online network. A hard copy every 1,000 updates has the same average delay but catches up completely at each copy. With a slow target, the 3-step and PER updates spend longer fitting stale values.
- **Twice the parameters, same data and learning rate.** The two 512-unit streams double the size of the network. With the same learning rate and number of updates, the larger head may simply need more steps.
- **Single run.** As with run 1, each configuration was trained once, so part of the gap may be chance. But the gap is consistent across all six snapshots and the training curve.

**Conclusion.** Adding the two DL-7 ideas on top did not fix the plateau of run 1; it lowered it further. Run 1 therefore remains the submitted Task 3 model. Both enhanced runs stay below vanilla DQN at the same budget, which suggests the problem comes from a component they share. The ablation study below tests each technique separately to find it.

#### Ablation study: each technique on its own

To find which part of run 1 caused the plateau, I trained three more agents, each with **only one** technique switched on. Everything else is identical to run 1 (lr 1e-4, batch 32, memory 100k, replay start 20k, ε decay 0.99998 → 0.02, hard target copy every 1,000 updates, MSE-type loss with gradient clip 10). Each run stopped at 1M env steps, and its 600k and 1M snapshots were evaluated on seeds 0–19 with `test_model_task3.py`.

| W&B run | Double DQN | PER (α 0.6, β 0.4 → 1) | 3-step return |
|---|---|---|---|
| `task3-ablation-ddqn` | ✓ | – | – |
| `task3-ablation-per` | – | ✓ | – |
| `task3-ablation-nstep3` | – | – | ✓ |
| `task3-full` (run 1) | ✓ | ✓ | ✓ |

**Snapshot scores (20-seed average):**

| Variant | 600k | 1M | 1M min / max | 1M games ≥ 19 |
|---|---|---|---|---|
| Vanilla DQN (Task 2, nearest checkpoints) | 13.80 (0.66M) | 17.60 (0.90M) / 16.35 (1.12M) | 12 / 21 (0.90M) | 7 (0.90M) |
| Double DQN only | 2.30 | **16.55** | 12 / 21 | 3 |
| PER only | 8.80 | 15.05 | 7 / 21 | 4 |
| 3-step return only | 8.55 | 12.40 | 2 / 19 | 1 |
| All three (run 1) | 13.05 | 11.05 | 3 / 17 | 0 |

**Training curves (5-game evaluations, averaged per 100k steps):**

| Env steps | DDQN only | PER only | 3-step only | All three (run 1) |
|---|---|---|---|---|
| 0.1M | −12.0 | −14.3 | −11.2 | −12.0 |
| 0.2M | −11.8 | −10.1 | **−2.3** | −9.2 |
| 0.3M | −7.9 | −2.7 | **7.1** | −3.5 |
| 0.4M | −1.8 | 0.6 | **11.6** | 2.3 |
| 0.5M | 3.6 | 7.5 | **10.2** | 8.4 |
| 0.6M | 7.1 | 6.6 | **11.0** | 10.0 |
| 0.7M | 12.1 | **12.4** | 9.4 | 9.4 |
| 0.8M | 13.4 | 11.7 | 10.7 | **15.1** |
| 0.9M | **15.6** | 14.3 | 12.1 | 12.4 |
| 1.0M | **17.0** | 13.8 | 16.0 | 9.3 |

None of the five enhanced runs had a 5-game evaluation of 19 or more within 1M steps (2.5M for runs 1 and 2).

**Findings.**

- **The 3-step return learns fastest early, then levels off lower.** With the reward propagated three steps per update, it reaches 11.6 at 0.4M while the other variants are still around 0. After that it stays around 10–12, and its 1M snapshot (12.40) is the weakest single technique. Run 1 has the same shape: an early lead from the 3-step return, then a plateau. So the multi-step return is the most likely cause of run 1's low ceiling.
  - A plausible reason: the 3-step target sums rewards from actions stored in the replay buffer, which were chosen by an *older* policy. Without an off-policy correction, these targets are biased toward that older behavior. Early on this hardly matters and the faster propagation dominates. Once the policy is good, the bias holds it back.
- **Double DQN starts slowly but ends strongest.** It is the slowest variant up to ≈ 0.5M, then climbs steadily to 16.55 at 1M, level with vanilla DQN. This fits its purpose: it corrects overestimated Q-values, which improves stability later in training rather than early speed.
- **PER is in between.** It is close to Double DQN at 0.5–0.7M and ends at 15.05. On its own it does not cause a plateau.
- **Combining them did not add up.** All three together kept the 3-step return's early speed (best of all variants at 0.6M, 13.05) but also its ceiling, and ended below every single technique at 1M (11.05). PER may make this worse, since it replays the transitions with the largest (and here, most biased) n-step errors more often.

**Limitations.**

- Each variant was trained **once**, and single snapshots are noisy. For example, the Double DQN 600k snapshot scored 2.30 on 20 seeds while its training evaluations around 0.6M averaged 7.1. Conclusions are about clear trends, not small gaps.
- The vanilla baseline is the Task 2 run, which used a replay start of 50k and a slower ε decay (0.999988). A "no enhancement" run with exactly the ablation settings would be a cleaner baseline.
- The ablation stops at 1M steps, so it says nothing about which variant would reach 19 first after that.

#### Sample efficiency with and without the enhancements

| Variant | Env steps to reach score 19 | Best 20-seed score within budget |
|---|---|---|
| Vanilla DQN (Task 2) | ≈ 0.95M (first 3-game eval ≥ 19); 3.09M on 20 seeds | 19.10 (3.09M snapshot) |
| Double DQN only | not reached within 1M | 16.55 (1M) |
| PER only | not reached within 1M | 15.05 (1M) |
| 3-step return only | not reached within 1M | 12.40 (1M) |
| All three (Task 3, run 1) | not reached on 20 seeds within 2.5M | 15.25 (best snapshot, ≈ 2.36M) |
| All three + dueling + soft target (run 2) | not reached on 20 seeds within 2.5M | 12.05 (2.5M) |

In this setup the enhancements **did not improve sample efficiency** over vanilla DQN past the first ≈ 0.6M steps. The 3-step return gave a clear early speed-up (score ≈ 11 by 0.4M steps versus ≈ 0 for the others), but every enhanced variant ended at or below vanilla DQN at 1M steps. The ablation points to the uncorrected multi-step return as the component that limits the final score, and to Double DQN as the one that helps most later in training.

---

## 5. Additional Analysis (Bonus)

### 5.1 Hyperparameter study on CartPole

The template's default hyperparameters suit Pong rather than CartPole: ε decay 0.999999 keeps exploration near 100% for millions of updates, and a 50k replay start delays learning by about half of what CartPole needs. To reach the 480 target efficiently, I ran a small hyperparameter study. Before the final run, eight configurations were compared, each for a single seed with W&B disabled. The "best eval" column is the best 5-episode average reached during training; the last column is the best snapshot evaluated on seeds 0–19.

**Round 1** (600 episodes; truncation stored as terminal)

| Run | lr | Target update | Batch | Replay start | ε decay | Best eval | Notes |
|---|---|---|---|---|---|---|---|
| A | 5e-4 | 500 | 32 | 500 | 0.999 | 309 | drifted down to ≈ 150, collapsed near the end |
| B | 2.5e-4 | 1000 | 64 | 1000 | 0.9995 | 272 | stayed around 160–270 |
| C | 1e-4 | 1000 | 64 | 1000 | 0.9995 | 322 | slid down to ≈ 170 |
| D | 5e-4 | 2000 | 64 | 1000 | 0.9995 | 14 | did not learn within 7k steps |

**Round 2** (800 episodes; truncation not treated as terminal; batch 64, replay start 1000, ε decay 0.9995)

| Run | lr | Target update | Best eval | Behavior | 20-seed score of best model |
|---|---|---|---|---|---|
| **E** | 5e-4 | 500 | 500 | 500 at every evaluation from ≈ 94k to 160k steps | **500.00** |
| F | 1e-3 | 200 | 500 | reached 500 by ≈ 52k steps but repeatedly collapsed (500 → 78 → 500 → 36 …) | 500.00 |
| G | 2.5e-4 | 1000 | 246 | stayed around 130–250 | 190.60 |
| H | 1e-4 | 1000 | 355 | steady around 200–350 | 344.25 |

**Observations.**

- **Learning rate.** A high learning rate (1e-3) learns fastest but is unstable; low ones (1e-4, 2.5e-4) are stable but too slow for the training budget. 5e-4 was the best compromise.
- **Target update frequency.** Very frequent syncs (every 200 updates, run F) contributed to oscillations (a moving target). Very infrequent syncs (every 2,000 updates, run D) slowed early learning.
- **Truncation handling.** Runs B→G and C→H differ only in the truncation fix and gave similar results within this budget, so this single-seed experiment does not show a clear empirical effect. The fix is kept because it makes the targets correct.
- **Limitation.** Every configuration was run with one seed, so differences of a few dozen reward points are within run-to-run noise.

### 5.2 Other training strategies — TODO (optional)

Ideas: Huber vs. MSE loss, soft target updates, learning-rate schedules.

---

## Appendix: How to Reproduce

```bash
pip install -r requirements.txt
wandb login
```

**Task 1 training**

```bash
python dqn.py --wandb-run-name task1-final --lr 5e-4 --target-update-frequency 500 \
  --batch-size 64 --replay-start-size 1000 --epsilon-decay 0.9995 \
  --memory-size 50000 --episodes 800 --eval-episodes 5 --save-dir ./results/task1
```

**Task 1 evaluation**

```bash
python test_model_task1.py --model-path LAB5_115550809_task1.pt
```

**Task 2 training**

```bash
python dqn.py --env ALE/Pong-v5 --wandb-run-name task2-vanilla --save-dir ./results/task2 \
  --lr 1e-4 --batch-size 32 --memory-size 100000 --replay-start-size 50000 \
  --epsilon-decay 0.999988 --epsilon-min 0.02 --target-update-frequency 1000 \
  --episodes 3000 --eval-episodes 3
```

(Stopped manually after 2,700 games.)

**Task 2 evaluation and demo video**

```bash
python test_model.py --model-path LAB5_115550809_task2.pt --env-steps 3092538
python test_model.py --model-path LAB5_115550809_task2.pt --episodes 3 --save-video --output-dir ./eval_videos
```

**Task 3:** TODO
