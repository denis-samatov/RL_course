# 🎯 Session 10: Deep Q-Network (DQN)

> **Theory:** [note_10_deep_q_network.md](../../notes/md/note_10_deep_q_network.md)  
> **Algorithm:** Deep Q-Network with Experience Replay and a Target Network

---

## 📖 Overview

A full implementation of **Deep Q-Network (DQN)** — the first successful application of deep neural networks to Reinforcement Learning. DQN solves the problem of scaling tabular Q-Learning to large state spaces.

**Environment:** LunarLander-v2  
**Action type:** Discrete (4 actions)  
**State:** Continuous (8-dimensional)  
**Goal:** Land the lunar module between the flags while minimizing fuel use

---

## 🎯 Implementation features

### The DQN algorithm

- ✅ **Deep Q-Network** — an MLP approximating the Q-function
- ✅ **Experience Replay** — a buffer for storing and reusing transitions
- ✅ **Target Network** — stabilizes training via periodic weight copying
- ✅ **ε-greedy exploration** — balances exploration and exploitation
- ✅ **Gradient clipping** — prevents gradient explosion

### Architecture

```text
State (8) → FC(128) → ReLU → FC(128) → ReLU → Q-values(4)
```

### Key formulas

**TD target:**

```text
y = r + γ * max_a' Q_target(s', a')  (if not done)
y = r                                 (if done)
```

**Loss (MSE):**

```text
L(θ) = E[(y - Q_θ(s, a))²]
```

**Target network update:**

```text
θ_target ← θ  (every C steps)
```

---

## 🚀 Quick start

### Installing dependencies

```bash
pip install gymnasium[box2d] torch numpy matplotlib tqdm
```

### Basic run

```bash
cd code/10_deep_q_network
python dqn_algorithm.py
```

### Running with parameters

```bash
# Train for 80k steps
python dqn_algorithm.py --total-steps 80000

# With a different learning rate
python dqn_algorithm.py --lr 5e-4

# With a larger buffer
python dqn_algorithm.py --buffer-size 100000
```

### All parameters

| Parameter | Default | Description |
|----------|--------------|----------|
| `--total-steps` | 50000 | Total number of training steps |
| `--lr` | 1e-3 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--epsilon-start` | 1.0 | Starting epsilon |
| `--epsilon-end` | 0.05 | Final epsilon |
| `--epsilon-decay` | 0.997 | Exponential decay |
| `--buffer-size` | 50000 | Replay buffer size |
| `--batch-size` | 64 | Mini-batch size |
| `--target-update` | 1000 | Target network update frequency |
| `--warmup-steps` | 2000 | Steps before training starts |
| `--seed` | 42 | Random seed |

---

## 📊 Expected results

### Convergence

**Typical training curve:**

```text
Steps 0-10000:   Reward ~ -200 to 0    (hard landings)
Steps 10000-25000: Reward ~ 0 to 150    (learning a soft landing)
Steps 25000-40000: Reward ~ 150 to 220  (stable landing)
Steps 40000-50000: Reward ~ 220 to 260  (near-optimal strategy)
```

**Solved criterion:** Average reward >= 200 over 100 consecutive episodes

### Sample output

```text
==========================================================
DQN Training on LunarLander-v2
==========================================================
Total steps: 50000
Learning rate: 0.001
Buffer size: 50000
Batch size: 64
Target update: 1000
Environment: LunarLander-v2
==========================================================
Training: 100%|████████| 50000/50000 [14:37<00:00, step=3425, reward=236.5, epsilon=0.05]

Evaluating trained policy...
Evaluation over 100 episodes: 212.7 ± 34.8
✓ Environment SOLVED! (Average reward >= 200)
Model saved to dqn_lunarlander.pt
```

---

## 📈 Visualization

After training, `dqn_training.png` is generated automatically:

- **Left panel:** Rewards per episode with a rolling average
- **Right panel:** Epsilon decay schedule
- **Red line:** Solved threshold (reward = 200)

---

## 🔬 Experiments

### 1. Effect of the target network

```bash
# With a target network (stable training)
python dqn_algorithm.py --target-update 1000

# Without a target network (unstable)
python dqn_algorithm.py --target-update 1
```

**Expected result:** Convergence is 30-40% more stable with a target network

### 2. Effect of the replay buffer

```bash
# Large buffer (better for stability)
python dqn_algorithm.py --buffer-size 100000

# Small buffer (faster, but less stable)
python dqn_algorithm.py --buffer-size 10000
```

### 3. Different epsilon schedules

```bash
# Fast decay (faster exploitation)
python dqn_algorithm.py --epsilon-decay 0.995

# Slow decay (more exploration)
python dqn_algorithm.py --epsilon-decay 0.999
```

---

## 🧪 Connection to the theory

This code implements the concepts from **note_10_deep_q_network.md**:

| Concept | Implementation in code |
|-----------|-------------------|
| Deep Q-Network | `DQN` class — an MLP for Q(s,a) |
| Experience Replay | `ReplayBuffer` — stores and samples transitions |
| Target Network | `target_net.load_state_dict(...)` every N steps |
| TD target | `compute_td_target()` — r + γ * max Q_target |
| ε-greedy | `epsilon_schedule()` + `select_action()` |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_()` |

**Code:**

- `dqn_algorithm.py` — the full DQN implementation for LunarLander-v2
- `homework.ipynb` — hands-on exercises
- `homework_solution.ipynb` — annotated solutions

**References:**

- Mnih et al. (2015): "Human-level control through deep reinforcement learning" (Nature DQN)
- Van Hasselt et al. (2016): "Deep Reinforcement Learning with Double Q-learning" (Double DQN)
- Wang et al. (2016): "Dueling Network Architectures" (Dueling DQN)

---

## 🐛 Troubleshooting

### Problem: Not converging after 50000 steps

**Solution:**

```bash
# Increase the number of steps
python dqn_algorithm.py --total-steps 80000

# Or lower the learning rate
python dqn_algorithm.py --lr 5e-4 --total-steps 20000
```

### Problem: Unstable training

**Solution:**

```bash
# Increase how often the target network updates
python dqn_algorithm.py --target-update 500

# Increase the buffer size
python dqn_algorithm.py --buffer-size 100000
```

### Problem: Training is too slow

**Solution:**

```bash
# Increase the learning rate
python dqn_algorithm.py --lr 2e-3

# Decrease the batch size (more updates)
python dqn_algorithm.py --batch-size 32
```

---

## 📊 Benchmark

**System:** MacBook Pro M2, 16GB RAM  
**Training time:** ~12-15 minutes (50000 steps)  
**Memory:** ~300-400 MB  
**Solved after:** ~40000-50000 steps

---

## 🎓 Homework

1. **Run the base training** and reach reward >= 200
2. **Compare** DQN with/without the target network (plot the curves)
3. **Experiment** with the replay buffer size (10K, 50K, 100K)
4. **Implement** Double DQN (see homework.ipynb)
5. **Add** Prioritized Experience Replay or a Dueling Network

---

**Author:** Denis Samatov, TPU / 2025  
**Course link:** Session 10 — Deep Q-Network
