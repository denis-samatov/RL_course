# 🤖 Session 15: RLHF — Reinforcement Learning from Human Feedback

> **Theory:** [note_15_rlhf_pipeline.md](../../notes/md/note_15_rlhf_pipeline.md)  
> **Pipeline:** SFT → Reward Model → PPO with a KL penalty

---

## 📖 Overview

A simplified demonstration of the RLHF pipeline on a toy task.

**Features:**
- A 3-stage pipeline (SFT → RM → PPO)
- A preference-based reward model
- A KL penalty against the reference policy
- Reward-hacking monitoring

---

## 🗂️ Structure

```
rlhf_basics/
├── simple_text_env.py      # A toy environment for text generation
├── sft_model.py            # Supervised Fine-Tuning
├── reward_model.py         # Reward Model training
├── ppo_rlhf.py             # PPO with a KL penalty
├── generate_data.py        # Preference data generation
├── train_pipeline.py       # The full pipeline
├── README.md               # This documentation
└── data/                   # (created when run)
    ├── sft_data.json
    └── preferences.json
```

---

## 🚀 Quick start

```bash
# 1. Generate data
python generate_data.py

# 2. The full RLHF pipeline
python train_pipeline.py
```

**Output:**
- An SFT model checkpoint
- A Reward Model checkpoint
- The RLHF-aligned model
- Comparison plots

---

## 📊 Experiments

### Experiment 1: Effect of β (the KL penalty)

```bash
python ppo_rlhf.py --beta 0.001  # Weak control
python ppo_rlhf.py --beta 0.01   # Baseline
python ppo_rlhf.py --beta 0.1    # Strong control
```

**Expected:**
- β=0.001: High reward, but reward hacking
- β=0.01: A balance
- β=0.1: Stable, but slow improvement

---

### Experiment 2: Reward hacking without KL

```bash
python ppo_rlhf.py --beta 0.0  # No KL penalty
```

**Expected:** The model finds adversarial examples with high reward, but meaningless output.

---

## 💡 Key components

### 1. Reward Model loss

```python
# Bradley-Terry Model
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

loss = -log σ(r(x, y_w) - r(x, y_l))
```

### 2. PPO with a KL penalty

```python
total_reward = rm_reward - β * KL(π_θ || π_SFT)
```

### 3. Metrics to monitor

- RM Reward (should increase)
- KL Divergence (should stay < 10)
- Response Length (shouldn't blow up)
- Perplexity (should stay reasonable)

---

**Author:** Denis Samatov, TPU / 2025

✅ **Session 15 complete!** Moving on to [Session 16: DPO](../16_dpo/README.md)
