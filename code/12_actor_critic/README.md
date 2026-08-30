# A2C (Advantage Actor-Critic) on Pendulum-v1

## 📘 Description

An implementation of the **A2C** (Advantage Actor-Critic) algorithm for the inverted-pendulum balancing task, with a **continuous action space**. Demonstrates the power of Actor-Critic methods for continuous control.

**Environment:** Pendulum-v1  
**Action type:** Continuous (torque ∈ [-2, 2])  
**State:** Continuous (3-dimensional: cos(θ), sin(θ), θ̇)  
**Goal:** Swing the pendulum upright and hold it there

---

## 🎯 Implementation features

### The A2C algorithm
- ✅ **Actor-Critic architecture** with a shared backbone
- ✅ **Gaussian policy** for continuous actions
- ✅ **TD-error (advantage)** to reduce variance
- ✅ **Separate optimizers** for the Actor and Critic (best practice)
- ✅ **Entropy regularization** for exploration
- ✅ **Online updates** after every step
- ✅ **Gradient clipping** for stability

### Architecture

```
                      State (3)
                          ↓
        ┌─────────────────────────────────┐
        │   Shared Backbone                │
        │   FC(256) → ReLU → FC(256) → ReLU│
        └──────────────┬──────────────────┘
                       ↓
           ┌───────────┴───────────┐
           ↓                       ↓
       Actor Head              Critic Head
       ↓                       ↓
   Mean(1), LogStd(1)      Value(1)
       ↓
   N(μ, σ) → Action
```

### Key formulas

**TD error (Advantage):**
```
δ_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)
```

**Actor update (Policy Gradient):**
```
∇_θ J(θ) = E [ ∇_θ log π_θ(a_t|s_t) · δ_t + β · H(π_θ) ]
```

**Critic update (TD Learning):**
```
L(φ) = (δ_t)²
```

**Gaussian Policy:**
```
π_θ(a|s) = N(μ_θ(s), σ_θ(s))
log π_θ(a|s) = -½[(a - μ)/σ]² - log σ - ½log(2π)
```

---

## 🚀 Quick start

### Installing dependencies

```bash
# From the repo root
pip install -r requirements.txt
```

### Basic run

```bash
cd code/12_actor_critic
python pendulum_a2c.py
```

### Running with parameters

```bash
# Train for 1500 episodes with video recording
python pendulum_a2c.py --episodes 1500 --record-video

# With different learning rates for the Actor and Critic
python pendulum_a2c.py --lr-actor 5e-4 --lr-critic 1e-3

# With a different entropy coefficient
python pendulum_a2c.py --entropy 0.01

# A different random seed
python pendulum_a2c.py --seed 123
```

### All parameters

| Parameter | Default | Description |
|----------|--------------|----------|
| `--episodes` | 1000 | Number of training episodes |
| `--lr-actor` | 3e-4 | Learning rate for the Actor |
| `--lr-critic` | 1e-3 | Learning rate for the Critic |
| `--entropy` | 0.001 | Entropy regularization coefficient |
| `--seed` | 42 | Random seed for reproducibility |
| `--record-video` | False | Record video of the evaluation episodes |

---

## 📊 Expected results

### Convergence

**Typical training curve:**
```
Episode 0-200:    Reward ~ -1400 to -800  (random actions)
Episode 200-400:  Reward ~ -800 to -400   (learning to swing up)
Episode 400-600:  Reward ~ -400 to -200   (holding upright)
Episode 600-1000: Reward ~ -200 to -150   (near-optimal control)
```

**Solved criterion:** Average reward >= -200 over 100 consecutive episodes

**Note:** Rewards in Pendulum-v1 are negative (penalties for deviation); the goal is to maximize them (get as close to 0 as possible).

### Sample output

```
============================================================
A2C on Pendulum-v1 (Continuous Control)
============================================================
Episodes: 1000
Actor LR: 0.0003
Critic LR: 0.001
Entropy coefficient: 0.001
Seed: 42
============================================================
Training A2C: 100%|████████| 1000/1000 [05:23<00:00, reward=-156.3, avg_100=-178.4, actor_loss=0.234, critic_loss=12.456]

Evaluating trained policy...
Evaluation over 100 episodes: -165.23 ± 45.67
✓ Environment SOLVED! (Average reward >= -200)
Model saved to pendulum_a2c.pt
```

---

## 📈 Visualization

After training, `pendulum_a2c_training.png` is generated automatically, with 4 panels:

1. **Training Rewards** — rewards per episode
2. **Episode Lengths** — duration (always 200 for Pendulum)
3. **Actor Loss** — the policy loss
4. **Critic Loss** — the value function loss

---

## 🎥 Video recording

```bash
python pendulum_a2c.py --record-video
```

Videos are saved to `videos/pendulum/`:
- 5 evaluation episodes after training
- Show the trained agent's behavior
- Format: MP4, FPS: 30

---

## 🔬 Experiments

### 1. Comparing learning rates

```bash
# High Actor LR (can be unstable)
python pendulum_a2c.py --lr-actor 1e-3 --lr-critic 1e-3

# Low Actor LR (stable, but slow)
python pendulum_a2c.py --lr-actor 1e-4 --lr-critic 1e-3

# Balanced (recommended)
python pendulum_a2c.py --lr-actor 3e-4 --lr-critic 1e-3
```

**Rule of thumb:** The Critic usually needs to learn faster than the Actor (lr_critic > lr_actor)

### 2. Effect of entropy

```bash
# No entropy (deterministic policy)
python pendulum_a2c.py --entropy 0.0

# Low entropy (recommended for continuous control)
python pendulum_a2c.py --entropy 0.001

# High entropy (more exploration, but slower)
python pendulum_a2c.py --entropy 0.01
```

### 3. Training length

```bash
# Short training
python pendulum_a2c.py --episodes 500

# Standard
python pendulum_a2c.py --episodes 1000

# Long (for stabilization)
python pendulum_a2c.py --episodes 2000
```

---

## 🧪 Connection to the theory

This code implements the concepts from **note_12_actor_critic_a2c.md**:

| Concept | Implementation in code |
|-----------|-------------------|
| Actor-Critic | `ActorCriticNetwork` with two heads |
| Gaussian Policy | `Normal(mean, std)` for continuous actions |
| TD error (Advantage) | `td_target - value` |
| Shared Backbone | `self.shared` layers for the Actor and Critic |
| Separate optimizers | `actor_optimizer`, `critic_optimizer` |
| Entropy regularization | `entropy_coef * entropy` |
| Gradient clipping | `clip_grad_norm_()` |
| Online updates | Updated after every step, in `train_step()` |

---

## 🆚 Comparison with REINFORCE

| Property | REINFORCE | A2C |
|----------------|-----------|-----|
| Updates | After each episode | After each step |
| Baseline | Optional value function | Mandatory Critic |
| Variance | High | Medium |
| Sample efficiency | Low | Medium |
| Convergence speed | Slow | Faster |
| Continuous actions | ✓ | ✓ |
| Stability | Low | Higher |

**Takeaway:** A2C is more sample-efficient and stable thanks to online updates and TD-learning.

---

## 📚 Further materials

**Theory:**
- `/notes/md/note_12_actor_critic_a2c.md` — Actor-Critic methods and A2C
- `/notes/md/note_11_policy_gradients_reinforce.md` — Policy Gradients (for comparison)
- `/notes/md/note_08_monte_carlo_vs_td.md` — TD-learning (the basis for the Critic)

**Code:**
- `ActorCriticNetwork` — the architecture with a shared backbone
- `get_action()` — sampling from the Gaussian policy
- `train_step()` — the online A2C update

**References:**
- Mnih et al. (2016): "Asynchronous Methods for Deep RL" (A3C)
- Schulman et al. (2015): "High-Dimensional Continuous Control Using GAE"
- Sutton & Barto (2020): Chapter 13 - Policy Gradient Methods

---

## 🐛 Troubleshooting

### Problem: Actor/Critic losses keep growing

**Cause:** Learning rates too high, or unstable gradients

**Solution:**
```bash
# Lower the learning rates
python pendulum_a2c.py --lr-actor 1e-4 --lr-critic 5e-4

# Or increase gradient clipping (change config.gradient_clip in the code)
```

### Problem: The policy becomes deterministic too quickly

**Cause:** Low entropy, or log_std drops too fast

**Solution:**
```bash
# Increase the entropy coefficient
python pendulum_a2c.py --entropy 0.01

# Or change log_std_min in the code (e.g. -10 instead of -20)
```

### Problem: Not converging to -200

**Cause:** Not enough episodes, or suboptimal hyperparameters

**Solution:**
```bash
# Increase the number of episodes
python pendulum_a2c.py --episodes 1500

# Or balance the learning rates
python pendulum_a2c.py --lr-actor 5e-4 --lr-critic 1e-3 --episodes 1200
```

---

## 📊 Benchmark

**System:** MacBook Pro M2, 16GB RAM  
**Training time:** ~5-6 minutes (1000 episodes)  
**Memory:** ~150-200 MB  
**Solved after:** ~600-800 episodes (with optimal hyperparameters)

---

## 🎯 Advanced improvements

### 1. GAE (Generalized Advantage Estimation)

The current implementation uses a 1-step TD error. To improve it, you can implement GAE:

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

### 2. PPO (Proximal Policy Optimization)

The next step in the progression is to add clipping for more stable updates:

```python
ratio = torch.exp(new_log_prob - old_log_prob)
clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()
```

### 3. Batch updates

Instead of updating after every step, accumulate several transitions first:

```python
# Collect N steps
# Then update all at once
```

---

## 🎓 Homework

1. **Run the base training** and reach reward >= -200
2. **Compare** different learning rates (plot the Actor/Critic loss curves)
3. **Experiment** with the entropy coefficient (0.0, 0.001, 0.01)
4. **Implement** GAE instead of the 1-step TD error
5. **Try** a different continuous environment (MountainCarContinuous-v0)
6. **Compare** A2C against REINFORCE on the same environment

---

## 🔗 Related implementations

- **REINFORCE on LunarLander:** `/code/11_policy_gradient/` — for comparison
- **Q-Learning on CartPole:** `/code/09_q_learning_bellman/` — a value-based approach
- **MC vs TD on FrozenLake:** `/code/08_mc_vs_td/` — TD-learning basics

---

**Author:** Denis Samatov, TPU / 2025  
**Course link:** Session 12 — Actor-Critic and A2C  
**Next steps:** PPO, SAC, DDPG, TD3
