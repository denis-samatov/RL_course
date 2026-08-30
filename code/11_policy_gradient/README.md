# Policy Gradient (REINFORCE) on LunarLander-v2

## 📘 Description

An implementation of the **REINFORCE** algorithm (Monte Carlo Policy Gradient) for the lunar-lander task. Demonstrates **policy-based methods**, which directly learn a stochastic policy via gradient ascent on expected reward.

**Environment:** LunarLander-v2  
**Action type:** Discrete (4 actions)  
**State:** Continuous (8-dimensional)  
**Goal:** Land the lunar module between the flags while minimizing fuel use

---

## 🎯 Implementation features

### The REINFORCE algorithm
- ✅ **Pure policy gradient** with a Monte Carlo return estimate
- ✅ **Baseline (value function)** to reduce gradient variance
- ✅ **Entropy regularization** to maintain exploration
- ✅ **Gradient clipping** for training stability
- ✅ **Advantage normalization** to improve convergence

### Architecture
```
State (8) → FC(128) → ReLU → FC(128) → ReLU → Logits(4) → Softmax → Policy
State (8) → FC(128) → ReLU → FC(128) → ReLU → Value(1)
```

### Key formulas

**Policy Gradient Theorem:**
```
∇_θ J(θ) = E_τ [ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t - V(s_t)) ]
```

**Advantage (with a baseline):**
```
A_t = G_t - V_φ(s_t)
where G_t = Σ_{k=t}^T γ^(k-t) r_{k+1}
```

---

## 🚀 Quick start

### Installing dependencies

```bash
# From the repo root
pip install -r requirements.txt

# Additionally, for LunarLander
pip install gymnasium[box2d]
```

### Basic run

```bash
cd code/11_policy_gradient
python lunarlander_reinforce.py
```

### Running with parameters

```bash
# Train for 3000 episodes with video recording
python lunarlander_reinforce.py --episodes 3000 --record-video

# Without a baseline (pure REINFORCE)
python lunarlander_reinforce.py --baseline False --episodes 2000

# With increased entropy for exploration
python lunarlander_reinforce.py --entropy 0.05

# Change the learning rate
python lunarlander_reinforce.py --lr 1e-3
```

### All parameters

| Parameter | Default | Description |
|----------|--------------|----------|
| `--episodes` | 2000 | Number of training episodes |
| `--lr` | 3e-4 | Learning rate for the policy |
| `--baseline` | True | Use the value baseline |
| `--entropy` | 0.01 | Entropy regularization coefficient |
| `--seed` | 42 | Random seed for reproducibility |
| `--record-video` | False | Record video of the best episodes |

---

## 📊 Expected results

### Convergence

**Typical training curve:**
```
Episode 0-500:    Reward ~ -300 to -100 (random actions)
Episode 500-1000: Reward ~ -100 to 0    (learning a soft landing)
Episode 1000-1500: Reward ~ 0 to 150   (stable landing)
Episode 1500-2000: Reward ~ 150 to 250 (near-optimal strategy)
```

**Solved criterion:** Average reward >= 200 over 100 consecutive episodes

### Sample output

```
==========================================================
REINFORCE on LunarLander-v2
==========================================================
Episodes: 2000
Learning rate: 0.0003
Baseline: True
Entropy coefficient: 0.01
Seed: 42
==========================================================
Training REINFORCE: 100%|████████| 2000/2000 [12:34<00:00, reward=-45.2, avg_100=178.3, length=234]

Evaluating trained policy...
Evaluation over 100 episodes: 203.45 ± 38.22
✓ Environment SOLVED! (Average reward >= 200)
Model saved to lunarlander_reinforce.pt
```

---

## 📈 Visualization

After training, `lunarlander_reinforce_training.png` is generated automatically:

- **Left panel:** Rewards per episode with a rolling average
- **Right panel:** Episode lengths
- **Red line:** Solved threshold (reward = 200)

---

## 🎥 Video recording

```bash
python lunarlander_reinforce.py --record-video
```

Videos are saved to `videos/lunarlander/`:
- The 5 best episodes after training
- Format: MP4
- FPS: 30

---

## 🔬 Experiments

### 1. Effect of the baseline

```bash
# With a baseline (low variance)
python lunarlander_reinforce.py --baseline --episodes 1500

# Without a baseline (high variance)
python lunarlander_reinforce.py --episodes 1500
```

**Expected result:** Convergence is ~30-40% faster with a baseline

### 2. Effect of entropy

```bash
# Low entropy (faster convergence, risk of a local minimum)
python lunarlander_reinforce.py --entropy 0.001

# High entropy (slower, but better exploration)
python lunarlander_reinforce.py --entropy 0.05
```

### 3. Different learning rates

```bash
python lunarlander_reinforce.py --lr 1e-3  # Faster, but less stable
python lunarlander_reinforce.py --lr 1e-4  # Slower, but more stable
python lunarlander_reinforce.py --lr 5e-4  # A middle ground
```

---

## 🧪 Connection to the theory

This code implements the concepts from **note_11_policy_gradients_reinforce.md**:

| Concept | Implementation in code |
|-----------|-------------------|
| Policy Gradient Theorem | `train_episode()` → gradient computation |
| REINFORCE | The full Monte Carlo return, `compute_returns()` |
| Baseline | `ValueNetwork` and subtracting `values.detach()` |
| Advantage | `advantages = returns - values.detach()` |
| Entropy regularization | `entropy_loss = -entropies.mean()` |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_()` |

---

## 📚 Further materials

**Theory:**
- `/notes/md/note_11_policy_gradients_reinforce.md` — Policy Gradients and REINFORCE
- `/notes/md/note_04_policy_vs_value_methods.md` — Policy-Based vs Value-Based methods

**Code:**
- `PolicyNetwork` — the discrete stochastic policy
- `ValueNetwork` — the baseline for variance reduction
- `compute_returns()` — the Monte Carlo return estimate

**References:**
- Williams (1992): "Simple Statistical Gradient-Following Algorithms"
- Sutton & Barto (2020): Chapter 13 - Policy Gradient Methods
- Schulman et al. (2015): "High-Dimensional Continuous Control Using GAE"

---

## 🐛 Troubleshooting

### Problem: Not converging after 2000 episodes

**Solution:**
```bash
# Increase the number of episodes
python lunarlander_reinforce.py --episodes 3000

# Or lower the learning rate
python lunarlander_reinforce.py --lr 1e-4 --episodes 2000
```

### Problem: Reward variance is too high

**Solution:**
```bash
# Make sure the baseline is enabled
python lunarlander_reinforce.py --baseline

# Increase the network size
# (change hidden_sizes in the code)
```

### Problem: ImportError for box2d

**Solution:**
```bash
pip install gymnasium[box2d]
# or
pip install box2d-py swig
```

---

## 📊 Benchmark

**System:** MacBook Pro M2, 16GB RAM  
**Training time:** ~12-15 minutes (2000 episodes)  
**Memory:** ~200-300 MB  
**Solved after:** ~1500-1800 episodes (with a baseline)

---

## 🎓 Homework

1. **Run the base training** and reach reward >= 200
2. **Compare** REINFORCE with/without a baseline (plot the curves)
3. **Experiment** with the entropy coefficient (0.001, 0.01, 0.05)
4. **Implement** n-step returns instead of the full MC return
5. **Try** a different environment (CartPole-v1)

---

**Author:** Denis Samatov, TPU / 2025  
**Course link:** Session 11 — Policy Gradient Methods
