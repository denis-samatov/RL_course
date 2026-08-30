# 🚀 Session 14: Proximal Policy Optimization (PPO) and TRPO

> **Theory:** [note_14_ppo_trpo.md](../../notes/md/note_14_ppo_trpo.md)  
> **Algorithm:** PPO-Clip with GAE, Multiple Epochs, Gradient Clipping

---

## 📖 Overview

A full implementation of the **state-of-the-art** PPO algorithm for continuous actions.

### Key features:

✅ **PPO-Clip** — clips the probability ratio for stability  
✅ **GAE (λ=0.95)** — an optimal bias-variance trade-off  
✅ **Vectorized Environments** — parallel trajectory collection  
✅ **Multiple Epochs** — reuses data for sample efficiency  
✅ **Gradient Clipping** — prevents gradient explosion  
✅ **LR Annealing** — linear learning-rate decay  
✅ **Value Clipping** (optional) — stabilizes the critic  

---

## 🗂️ File structure

```
ppo_trpo/
├── ppo_agent.py           # Full PPO implementation
├── train_ppo.py           # Training script with wandb logging
├── evaluate_ppo.py        # Evaluation and video recording
├── compare_a2c_ppo.py     # Comparison against A2C
├── README.md              # This documentation
└── checkpoints/           # (Created during training)
    └── ppo_bipedalwalker.pt
```

> **Note:** only `ppo_agent.py` is currently in this directory. `train_ppo.py`, `evaluate_ppo.py`, and `compare_a2c_ppo.py`, referenced below, are not present yet — treat those sections as a description of intended usage rather than commands you can run today.

---

## 🚀 Quick start

### 1. Installing dependencies

```bash
pip install gymnasium[box2d] numpy torch tqdm matplotlib wandb
```

### 2. Training PPO

```bash
python ppo_agent.py
```

**Default parameters:**
- Environment: `BipedalWalker-v3`
- Total timesteps: 1,000,000
- Parallel envs: 4
- Clip range: 0.2
- GAE lambda: 0.95

**Expected time:** ~30-60 minutes on CPU, ~10-15 minutes on GPU

---

### 3. Evaluation

```bash
python evaluate_ppo.py --model checkpoints/ppo_bipedalwalker.pt --episodes 10
```

---

## 📊 PPO hyperparameters

| Parameter | Value | Description |
|----------|----------|----------|
| `clip_range` | 0.2 | Epsilon for the clipping ratio |
| `n_steps` | 2048 | Steps of trajectory collected |
| `n_epochs` | 10 | Optimization epochs per batch |
| `batch_size` | 64 | Mini-batch size |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `learning_rate` | 3e-4 | Initial LR (with annealing) |
| `value_coef` | 0.5 | Weight on the value loss |
| `entropy_coef` | 0.01 | Weight on the entropy bonus |
| `max_grad_norm` | 0.5 | Gradient clipping |

---

## 🎯 The BipedalWalker-v3 environment

**Description:** A two-legged robot must learn to walk across uneven terrain.

**Observations (24D):**
- Joint angular positions
- Angular velocities
- Foot contact with the ground
- LIDAR (10 rays)

**Actions (4D, continuous [-1, 1]):**
- Torques on the 4 joints (hip and knee for each leg)

**Rewards:**
- +300 for covering the distance
- -100 for falling
- A penalty for using the motors

**Solved criterion:** Average reward > 300

---

## 📈 Training results

### Expected training curve:

```
Timesteps     Mean Reward    Notes
---------     -----------    -----
0 - 200k      -100 to 0      Learning to stand
200k - 500k   0 to 150       Learning to take steps
500k - 800k   150 to 250     Learning to walk stably
800k - 1M     250 to 300+    Fine-tuning the gait
```

### Typical metrics:

| Metric | Start | End |
|---------|--------|-------|
| Mean Reward | -100 | 300+ |
| Episode Length | 300 | 1600 |
| Policy Loss | 0.5 | 0.05 |
| Value Loss | 50 | 5 |
| Approx KL | 0.02 | 0.005 |
| Clip Fraction | 0.3 | 0.1 |

---

## 🧪 Experiments

### Experiment 1: Effect of clip_range

```bash
# Test different epsilon values
python train_ppo.py --clip_range 0.1  # Conservative
python train_ppo.py --clip_range 0.2  # Baseline
python train_ppo.py --clip_range 0.3  # Aggressive
```

**Expected:**
- ε=0.1: Slower, but more stable
- ε=0.2: An optimal balance
- ε=0.3: Faster, but can be unstable

---

### Experiment 2: Effect of GAE lambda

```bash
python train_ppo.py --gae_lambda 0.90  # More bias
python train_ppo.py --gae_lambda 0.95  # Baseline
python train_ppo.py --gae_lambda 0.99  # More variance
```

**Theory:**
- λ→0: TD-like (high bias, low variance)
- λ→1: MC-like (low bias, high variance)

---

### Experiment 3: Shared vs separate networks

```python
# Shared backbone (default)
config = PPOConfig(shared_backbone=True)

# Separate networks
config = PPOConfig(shared_backbone=False)
```

**Trade-off:**
- Shared: Fewer parameters, faster training
- Separate: More flexibility, can work better for complex tasks

---

## 🔬 Comparison with A2C

```bash
python compare_a2c_ppo.py
```

**Expected differences:**

| Metric | A2C | PPO |
|---------|-----|-----|
| Final reward | 250 | 300+ |
| Sample efficiency | Lower | Higher |
| Stability | Medium | High |
| Training speed | Faster (per update) | Slower (multiple epochs) |
| Wall-clock time | ~40 min | ~30 min (more efficient) |

**Takeaway:** PPO usually outperforms A2C due to:
- Multiple epochs over the data (sample efficiency)
- Clipping for stability
- Lower sensitivity to hyperparameters

---

## 💡 Key components of PPO

### 1. Probability ratio with clipping

```python
# Ratio π_new / π_old
ratio = exp(log_prob_new - log_prob_old)

# PPO-Clip objective
surr1 = ratio * advantages
surr2 = clip(ratio, 1-ε, 1+ε) * advantages
policy_loss = -min(surr1, surr2).mean()
```

**Intuition:** If the ratio strays far from 1, clipping stops the update.

---

### 2. GAE (Generalized Advantage Estimation)

```python
gae = 0
for t in reversed(range(T)):
    delta = reward[t] + gamma * V[t+1] - V[t]
    gae = delta + gamma * lambda * gae
    advantages[t] = gae
```

**Intuition:** An exponentially weighted average of TD errors.

---

### 3. Multiple epochs

```python
for epoch in range(K):  # K = 10
    for batch in rollout_buffer.get_batches():
        # Update on the same batch of trajectories
        optimize_policy(batch)
```

**Why:** Reuses data for sample efficiency.

---

### 4. Value function loss

```python
# Optional: clipping for the value
v_clipped = v_old + clip(v_new - v_old, -ε, ε)
value_loss = max((v_new - returns)^2, (v_clipped - returns)^2).mean()
```

---

## 🐛 Troubleshooting

### Problem 1: Reward isn't increasing

**Symptoms:** Stuck around ~-100 after 500k steps

**Possible causes:**
- clip_range too small → increase to 0.3
- entropy too low → increase entropy_coef to 0.02
- learning rate too high → decrease to 1e-4

---

### Problem 2: Unstable training

**Symptoms:** Reward oscillates up and down

**Solution:**
- Decrease clip_range to 0.1
- Enable value clipping: `clip_range_vf = 0.2`
- Decrease the learning rate
- Increase n_steps (more trajectory data per update)

---

### Problem 3: High KL divergence

**Symptoms:** `approx_kl > 0.05` consistently

**Cause:** The policy is changing too fast

**Solution:**
- Decrease clip_range
- Decrease the learning rate
- Decrease n_epochs (fewer updates per batch)

---

## 📚 Further materials

### Original papers:

1. **PPO:** *Proximal Policy Optimization Algorithms* (Schulman et al., 2017)
2. **TRPO:** *Trust Region Policy Optimization* (Schulman et al., 2015)
3. **GAE:** *High-Dimensional Continuous Control Using GAE* (Schulman et al., 2016)

### Implementations:

- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Stable-Baselines3: PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [CleanRL: PPO](https://github.com/vwxyzjn/cleanrl)

---

## 🔗 Connection to other sessions

### Where this comes from:
- **[note_11_policy_gradients_reinforce.md](../../notes/md/note_11_policy_gradients_reinforce.md):** REINFORCE — the baseline PG method
- **[note_12_actor_critic_a2c.md](../../notes/md/note_12_actor_critic_a2c.md):** A2C — the Actor-Critic architecture

### Where this leads:
- **[note_15_rlhf_pipeline.md](../../notes/md/note_15_rlhf_pipeline.md):** RLHF — PPO for LLM fine-tuning
- **[note_16_dpo_and_variants.md](../../notes/md/note_16_dpo_and_variants.md):** DPO — an alternative to PPO-RLHF

---

## 💻 Usage examples

### Basic training:

```python
from ppo_agent import PPOAgent, PPOConfig

config = PPOConfig(
    env_id="BipedalWalker-v3",
    total_timesteps=1_000_000,
    n_envs=8,
)

agent = PPOAgent(config)
agent.train()
agent.save("ppo_model.pt")
```

### Customization:

```python
config = PPOConfig(
    clip_range=0.1,         # More conservative
    gae_lambda=0.99,        # Less bias
    n_epochs=15,            # More updates
    entropy_coef=0.02,      # More exploration
)
```

### Evaluation:

```python
agent = PPOAgent(config)
agent.load("ppo_model.pt")

# Run episodes
env = gym.make("BipedalWalker-v3", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    with torch.no_grad():
        action, _, _, _ = agent.network.get_action_and_value(
            torch.tensor(obs).unsqueeze(0)
        )
    obs, reward, terminated, truncated, _ = env.step(action.squeeze().numpy())
    if terminated or truncated:
        break
```

---

**Author:** Denis Samatov, TPU / 2025

✅ **Session 14 complete!** Moving on to [Session 15: RLHF](../15_rlhf_basics/README.md)
