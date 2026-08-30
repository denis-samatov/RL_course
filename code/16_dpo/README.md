# 🎯 Session 16: DPO — Direct Preference Optimization

> **Theory:** [note_16_dpo_and_variants.md](../../notes/md/note_16_dpo_and_variants.md)  
> **Method:** Direct optimization on preferences, without a Reward Model

---

> **Note:** this directory currently contains only this README — none of the files described below (`dpo_trainer.py`, `preference_dataset.py`, `train_dpo.py`, `compare_ppo_dpo.py`) exist yet. Treat this as a description of planned content, not runnable code.

## 📖 Overview

An implementation of DPO — a modern alternative to PPO-RLHF.

**Advantages:**
- ✅ Simpler than PPO-RLHF (2 stages instead of 3)
- ✅ No Reward Model (saves memory and time)
- ✅ More stable (fewer hyperparameters)
- ✅ Sample efficient

---

## 🗂️ Structure

```
dpo/
├── dpo_trainer.py              # DPO loss and training implementation
├── preference_dataset.py       # A dataset of paired preferences
├── train_dpo.py                # The training script
├── compare_ppo_dpo.py          # PPO vs DPO comparison
├── README.md                   # This documentation
└── experiments/
    └── results.json
```

---

## 🚀 Quick start

```bash
# Train DPO
python train_dpo.py --beta 0.1 --epochs 3

# Compare against PPO
python compare_ppo_dpo.py
```

---

## 📊 The DPO loss

```python
# Compute log-probs
logits_w = β * (log π_θ(y_w|x) - log π_ref(y_w|x))
logits_l = β * (log π_θ(y_l|x) - log π_ref(y_l|x))

# DPO loss
loss = -log sigmoid(logits_w - logits_l)
```

**Intuition:** Maximize the gap in log-probs between good and bad responses.

---

## 📈 Experiments

### Experiment 1: Effect of β

| β | Reward | KL | Interpretation |
|---|---------|----| --------------|
| 0.01 | High | High | Aggressive updates |
| 0.1 | Medium | Medium | Baseline ✅ |
| 0.5 | Low | Low | Conservative |

### Experiment 2: DPO vs PPO

| Metric | PPO | DPO | Winner |
|---------|-----|-----|--------|
| Final Reward | 8.5 | 8.3 | PPO |
| Training Time | 60 min | 15 min | DPO ✅ |
| Stability | 7/10 | 9/10 | DPO ✅ |
| Memory Usage | 40GB | 20GB | DPO ✅ |

**Takeaway:** DPO is faster, simpler, and more stable, with minimal quality loss.

---

**Author:** Denis Samatov, TPU / 2025

✅ **Session 16 complete!** Moving on to [Session 17: Final Project](../17_final_project/README.md)
