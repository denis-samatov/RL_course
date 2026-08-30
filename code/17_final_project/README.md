# 🏆 Session 17: Final Project — Mini-RLHF Pipeline

> **Theory:** [note_17_final_project.md](../../notes/md/note_17_final_project.md)  
> **Goal:** Integrate all the components into a complete RLHF pipeline

---

> **Note:** this directory currently contains only this README — none of the files, notebooks, or results described below exist yet. This reads as the final-project brief/assignment (with example target metrics), not a report of completed work.

## 📖 Project overview

Building an **end-to-end system** for LLM alignment:

1. **Supervised Fine-Tuning (SFT)** — adapting to instructions
2. **Reward Model (RM)** — training on preferences
3. **RL Fine-Tuning** — PPO or DPO for optimization
4. **Evaluation** — a comprehensive assessment of quality and safety

---

## 🗂️ Project structure

```
final_project/
├── data/
│   ├── sft_demonstrations.json      # SFT data
│   ├── preferences.json             # Paired preferences
│   └── test_prompts.json            # Evaluation
├── src/
│   ├── train_sft.py                 # Supervised Fine-Tuning
│   ├── train_rm.py                  # Reward Model
│   ├── train_ppo.py                 # PPO-RLHF
│   ├── train_dpo.py                 # DPO
│   └── evaluation/
│       ├── metrics.py               # Reward, KL, Perplexity
│       ├── safety_checks.py         # Toxicity, Bias
│       └── compare_models.py        # A/B testing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_rm_training.ipynb
│   ├── 04_rlhf_comparison.ipynb     # PPO vs DPO
│   └── 05_final_analysis.ipynb      # Final report
├── models/                           # (Created during training)
│   ├── sft_model/
│   ├── reward_model/
│   ├── ppo_model/
│   └── dpo_model/
├── scripts/
│   ├── train_full_pipeline.sh       # The full pipeline
│   └── run_evaluation.sh            # Evaluation
└── README.md                         # This documentation
```

---

## 🚀 Quick start

### 1. The full pipeline (automated)

```bash
bash scripts/train_full_pipeline.sh
```

**Stages:**
1. SFT training (~10 min)
2. RM training (~5 min)
3. PPO fine-tuning (~30 min)
4. DPO fine-tuning (~10 min)
5. Evaluation (~5 min)

**Total:** ~60 minutes on GPU

---

### 2. Running it step by step

```bash
# Stage 1: SFT
python src/train_sft.py --data data/sft_demonstrations.json

# Stage 2: RM
python src/train_rm.py --data data/preferences.json --base models/sft_model

# Stage 3: PPO-RLHF
python src/train_ppo.py --rm models/reward_model --ref models/sft_model --beta 0.01

# Stage 4: DPO (alternative)
python src/train_dpo.py --data data/preferences.json --ref models/sft_model --beta 0.1

# Stage 5: Evaluation
python src/evaluation/compare_models.py --models models/sft_model models/ppo_model models/dpo_model
```

---

## 📊 Evaluation metrics

### Automated metrics

```python
from evaluation.metrics import evaluate_model

metrics = evaluate_model(model, test_prompts)
print(f"RM Reward: {metrics['reward']:.2f}")
print(f"KL Divergence: {metrics['kl']:.3f}")
print(f"Perplexity: {metrics['perplexity']:.2f}")
print(f"Avg Length: {metrics['length']:.1f}")
```

### Manual evaluation (Win Rate)

```bash
python src/evaluation/human_eval.py --model_a models/sft_model --model_b models/ppo_model
```

**Example results:**
- Model A wins: 35%
- Model B wins: 55%
- Tie: 10%

**Win Rate (B vs A):** 55%

---

## 🎯 Target metrics (success criteria)

| Metric | Baseline (SFT) | Target (RLHF) | Achieved? |
|---------|----------------|---------------|-------------|
| RM Reward | 5.0 | > 7.0 | ✅ 7.8 |
| KL Divergence | 0.0 | < 5.0 | ✅ 2.3 |
| Perplexity | 12.5 | < 15.0 | ✅ 13.2 |
| Win Rate vs SFT | — | > 60% | ✅ 68% |
| Toxicity | 0.05 | < 0.10 | ✅ 0.07 |

---

## 🧪 Experiments

### Experiment 1: PPO vs DPO

```bash
python src/evaluation/compare_methods.py
```

**Example results:**

| Method | Reward | KL | Training Time | Memory |
|--------|--------|----| -------------|--------|
| SFT | 5.0 | 0.0 | — | — |
| PPO | 7.8 | 2.3 | 30 min | 40GB |
| DPO | 7.5 | 1.8 | 10 min | 20GB |

**Takeaway:** DPO reaches 96% of PPO's quality in 33% of the time and 50% of the memory.

---

### Experiment 2: Effect of β (the KL penalty)

| β | RM Reward | KL | Quality (Human) |
|---|-----------|----| ---------------|
| 0.001 | 9.2 | 8.5 | Poor (reward hacking) |
| 0.01 | 7.8 | 2.3 | Good ✅ |
| 0.1 | 6.2 | 0.5 | Okay (too conservative) |

---

### Experiment 3: Safety testing

```bash
python src/evaluation/safety_checks.py --model models/ppo_model
```

**Example results:**

| Metric | Score | Pass? |
|--------|-------|-------|
| Toxicity (avg) | 0.07 | ✅ < 0.10 |
| Bias (gender) | 0.52 | ✅ ≈ 0.50 |
| Refusal Rate (harmful) | 94% | ✅ > 90% |
| Hallucination Rate | 8% | ✅ < 10% |

---

## 📚 Notebooks

### 01_data_exploration.ipynb

- Analysis of the SFT data
- Distribution of prompt/response lengths
- Frequent tokens

### 02_sft_training.ipynb

- SFT loss curves
- Sample generations before/after SFT

### 03_rm_training.ipynb

- RM accuracy on validation
- Examples of high/low reward responses

### 04_rlhf_comparison.ipynb

- PPO vs DPO comparison
- Evolution of metrics (reward, KL, perplexity)
- A/B testing results

### 05_final_analysis.ipynb

- Final metrics for all models
- Safety analysis
- Best/worst examples
- Recommendations

---

## 💡 Key project takeaways

1. **RLHF significantly improves quality** (68% Win Rate vs SFT)
2. **The KL penalty is critical** for preventing reward hacking
3. **DPO is an effective alternative to PPO** (96% quality, 33% of the time)
4. **Safety testing is mandatory** before deployment
5. **Human evaluation is irreplaceable** for the final assessment

---

## 🐛 Troubleshooting

### Problem: Out of Memory

**Solution:**
- Decrease the batch size
- Use gradient accumulation
- Try LoRA/QLoRA

### Problem: Reward hacking

**Symptoms:** RM reward increases, but human eval drops

**Solution:**
- Increase β (the KL penalty)
- Early stopping based on human eval
- Refresh the RM with new data

### Problem: PPO converges slowly

**Solution:**
- Check the advantage normalization
- Decrease clip_range
- Increase the batch size

---

## 📝 Final checklist

### Required components

- [x] SFT trained and saved
- [x] RM trained on preferences
- [x] PPO fine-tuning with a KL penalty
- [x] DPO fine-tuning (alternative)
- [x] Automated metrics (reward, KL, perplexity)
- [x] Human evaluation (win rate)
- [x] Safety testing (toxicity, bias)

### Documentation

- [x] A README with instructions
- [x] Notebooks with analysis
- [x] Training plots
- [x] Sample generations
- [x] A final report

---

## 🎓 Congratulations on completing the course!

You've gone through the full journey from MDP fundamentals to modern RLHF. You can now:

✅ Implement any RL algorithm from scratch  
✅ Apply RLHF to LLM alignment  
✅ Evaluate the quality and safety of AI systems  
✅ Read and implement research papers  

---

**Next steps:**

1. Apply RLHF to a real task
2. Study advanced topics (multi-agent, model-based RL)
3. Participate in RL competitions
4. Contribute to open-source RL libraries

---

**Author:** Denis Samatov, TPU / 2025  
**Contact:** [GitHub](https://github.com/denissamatov)

🚀 **Good luck applying Reinforcement Learning!**
