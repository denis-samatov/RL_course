# Theoretical Note #17

## Topic: Final Project — Course Wrap-Up and Directions for Further Study

> **Final session:** Integrating everything covered in the course

---

## 1. What we covered: the full path from MDP to RLHF

### Foundational concepts (Sessions 1-4)

| Session | Topic | Key concepts |
|---------|------|-------------------|
| 1 | Introduction to RL | Agent-Environment, Reward, MDP |
| 2 | The MDP framework | States, actions, transitions, Bellman |
| 3 | Exploration vs Exploitation | ε-greedy, Softmax, UCB |
| 4 | Policy vs Value-Based | Two approaches to solving RL |

### Classical methods (Sessions 5-10)

| Session | Topic | Algorithms |
|---------|------|-----------|
| 5 | Deep RL | Neural-network approximation |
| 6 | Value functions | V(s) and Q(s,a) |
| 7 | The Bellman equation | Recursive updates |
| 8 | MC vs TD | Two ways of estimating V |
| 9 | Q-Learning | Off-policy TD control |
| 10 | DQN | Deep Q-Network and its extensions |

### Policy Gradient (Sessions 11-12)

| Session | Topic | Methods |
|---------|------|--------|
| 11 | Policy Gradients | REINFORCE, Baseline, Entropy |
| 12 | Actor-Critic | A2C, GAE, Continuous Control |

### Advanced RL (Sessions 13-14)

| Session | Topic | Methods |
|---------|------|--------|
| 13 | Dynamic Programming | Policy/Value Iteration, GPI |
| 14 | PPO and TRPO | Trust regions, Clipping |

### RLHF (Sessions 15-16)

| Session | Topic | Methods |
|---------|------|--------|
| 15 | RLHF | The SFT → RM → PPO pipeline |
| 16 | DPO | Direct Preference Optimization |

---

## 2. Final project: a mini RLHF pipeline

### Project goal

Build an **end-to-end RLHF system** on a simplified task:

1. A pretrained base model
2. Supervised Fine-Tuning on demonstrations
3. A Reward Model trained on preferences
4. RL fine-tuning (PPO or DPO)
5. Comprehensive evaluation

### Project components

```
final_project/
├── data/
│   ├── base_prompts.json        # Training prompts
│   ├── sft_demonstrations.json  # High-quality responses
│   ├── preferences.json         # Paired preferences
│   └── test_set.json            # Evaluation prompts
├── models/
│   ├── base_model/              # The starting model
│   ├── sft_model/                # After SFT
│   ├── reward_model/            # The trained RM
│   ├── ppo_model/                # After PPO-RLHF
│   └── dpo_model/                # After DPO
├── src/
│   ├── train_sft.py
│   ├── train_rm.py
│   ├── train_ppo.py
│   ├── train_dpo.py
│   └── evaluation/
│       ├── metrics.py           # Reward, KL, Perplexity
│       ├── safety_checks.py     # Toxicity, Bias
│       └── compare_models.py    # A/B testing
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_sft_training.ipynb
│   ├── 03_rm_training.ipynb
│   ├── 04_rlhf_comparison.ipynb
│   └── 05_final_analysis.ipynb
└── README.md
```

---

## 3. Evaluation metrics

### Automatic metrics

| Metric | Description | Code |
|---------|----------|-----|
| **RM Reward** | The average reward from the RM | `r_φ(x, y).mean()` |
| **KL Divergence** | Distance from SFT | `KL(π_θ \|\| π_SFT)` |
| **Perplexity** | Language quality | `exp(-log_prob.mean())` |
| **Length** | The average response length | `len(tokens).mean()` |
| **Diversity** | Unique n-grams | `unique_ngrams / total_ngrams` |

### Manual evaluation

**Win Rate:** For every prompt, a human picks the better response:

$$
\text{Win Rate} = \frac{\# \text{times model wins}}{\# \text{total comparisons}}
$$

**Typical comparisons:**
- SFT vs PPO-RLHF
- SFT vs DPO
- PPO vs DPO

---

### Safety metrics

| Metric | Tool | Threshold |
|---------|-----------|-----------|
| **Toxicity** | Perspective API | < 0.1 |
| **Bias** | Gender/Race templates | Balanced |
| **Refusal Rate** | On harmful prompts | > 90% |
| **Hallucination** | Fact-checking | < 10% |

---

## 4. Safety in RL

### Types of problems

1. **Reward hacking**
   - The model exploits weaknesses in the RM
   - Example: repeating words to inflate length

2. **Goal misgeneralization**
   - The model generalizes the wrong pattern
   - Example: learns to copy style rather than meaning

3. **Distributional shift**
   - Degrades on out-of-distribution prompts
   - Example: good on polite requests, poor on rude ones

4. **Value alignment**
   - The model doesn't share human values
   - Example: assists with dangerous actions

---

### Prevention methods

| Method | Description | Effectiveness |
|-------|----------|---------------|
| **KL Penalty** | $\beta D_{KL}(\pi_\theta \|\| \pi_{SFT})$ | ✅✅✅ Critically important |
| **Adversarial Testing** | Red-teaming with harmful prompts | ✅✅ Very useful |
| **Constitutional AI** | The model critiques its own responses | ✅✅ Helps, but expensive |
| **Human-in-the-Loop** | Regular human review | ✅✅✅ Necessary |
| **Multiple RMs** | Different reward models for different goals | ✅ Improves robustness |
| **Filtered Training Data** | Removing toxic examples | ✅ A baseline |

---

## 5. Ethics and social aspects

### Bias in the data

**The problem:** Datasets reflect societal biases (gender, race, etc.)

**Example:**
```
Prompt: "The doctor said..."
Biased: "...he will see you now"
Balanced: "...they will see you now"
```

**Solutions:**
- Balanced annotators (diversity)
- Explicit de-biasing in the SFT data
- Monitoring bias metrics

---

### Annotator representativeness

**The problem:** If every annotator comes from a single demographic → the preferences aren't universal.

**The solution:**
- Hire annotators from different countries/cultures
- Account for disagreement between annotators
- Multiple reward models for different audiences

---

### Long-term effects

**Questions:**
- How does RLHF shape society's perception of AI?
- Does it make models "too aligned" (sycophantic)?
- Does it centralize control over alignment?

---

## 6. Directions for further study

### Multi-Agent RL

- Agents interact with each other
- Nash equilibrium, coordination
- Applications: games, negotiation, multi-robot systems

### Model-Based RL

- Learning a model of the environment $P(s'|s,a)$
- Planning with a learned model (Dreamer, MuZero)
- Sample efficiency

### Offline RL

- Training on a fixed dataset (no interaction)
- Conservative Q-Learning, IQL
- Applications: medicine, finance

### Meta-RL and Few-Shot RL

- Learning "to learn"
- Fast adaptation to new tasks
- MAML, RL²

### Hierarchical RL

- Decomposition into sub-tasks
- Options, HAM
- Long-horizon tasks

### Safe RL and Constrained RL

- Explicit constraints (safety, fairness)
- CPO (Constrained Policy Optimization)
- Critical for real-world deployment

### RL for robotics

- Sim-to-real transfer
- Imitation learning + RL
- Continuous control

### Advanced RLHF

- Constitutional AI (Anthropic)
- Debate (OpenAI)
- Recursive Reward Modeling
- Scalable Oversight

---

## 7. A summary table of algorithms

| Algorithm | Type | On/Off Policy | Continuous | Sample Efficiency | Stability | When to use |
|----------|-----|---------------|------------|-------------------|--------------|-------------------|
| **Q-Learning** | Value-based | Off | ❌ | Low | Moderate | Discrete, small |
| **DQN** | Value-based | Off | ❌ | Moderate | Moderate | Discrete, large |
| **REINFORCE** | Policy | On | ✅ | Low | Low | Simple tasks |
| **A2C** | Actor-Critic | On | ✅ | Moderate | Moderate | General purpose |
| **PPO** | Actor-Critic | On | ✅ | High | High | State-of-the-art |
| **SAC** | Actor-Critic | Off | ✅ | Very high | High | Robotics |
| **TD3** | Actor-Critic | Off | ✅ | Very high | High | Continuous control |

---

## 8. Practical recommendations for real-world RL

### 1. Start simple

- Tabular Q-Learning to build understanding
- DQN for discrete actions
- PPO for continuous control

### 2. Tune hyperparameters

**Critical ones:**
- Learning rate (typically 1e-4 to 3e-4)
- Discount $\gamma$ (0.99 for episodic, 0.95 for continuing)
- Exploration (ε decay, entropy bonus)

### 3. Monitor metrics

- Episode return (should trend upward)
- Episode length (may vary)
- Loss (should stabilize)
- Gradient norm (shouldn't explode)

### 4. Use proven libraries

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — PyTorch, PPO/DQN/SAC
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — minimalist implementations
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html) — scalable RL
- [Hugging Face TRL](https://github.com/huggingface/trl) — RLHF for LLMs

### 5. Reproducibility

```python
# Seed everything
np.random.seed(42)
torch.manual_seed(42)
env.reset(seed=42)
env.action_space.seed(42)
```

---

## 9. What we learned (Learning Outcomes)

After completing this course, you can:

✅ Formalize a task as an MDP  
✅ Choose the right algorithm for a task  
✅ Implement DQN, REINFORCE, A2C, PPO from scratch  
✅ Understand the trade-offs: sample efficiency vs stability  
✅ Apply RLHF for LLM alignment  
✅ Evaluate the quality and safety of RL systems  
✅ Debug and tune RL algorithms  
✅ Read research papers and implement new methods  

---

## 10. Final project checklist

### Technical components

- [ ] The SFT pipeline is implemented
- [ ] A Reward Model is trained on preferences
- [ ] PPO with a KL penalty is implemented
- [ ] (Optional) DPO is implemented
- [ ] Every metric is logged (RM reward, KL, perplexity)

### Experiments

- [ ] Ablation: the effect of β (the KL penalty)
- [ ] Comparison: SFT vs PPO vs DPO
- [ ] Safety testing: adversarial prompts
- [ ] Human evaluation: win rate

### Documentation

- [ ] A README with instructions
- [ ] A description of the datasets
- [ ] Training curves
- [ ] Examples of the best/worst responses
- [ ] An analysis of the limitations

---

## 11. Course summary

**The path we took:**

```
MDP → Bellman → Q-Learning → DQN → Policy Gradient → 
Actor-Critic → PPO → RLHF → DPO
```

**Key lessons:**

1. **RL = iterative improvement** through trial and error
2. **Trade-offs are everywhere:** bias-variance, exploration-exploitation, sample efficiency-stability
3. **Engineering matters:** the right tricks (GAE, clipping, normalization) are critical
4. **Safety is a priority:** KL penalties, testing, monitoring

---

## 12. Conclusion

Reinforcement Learning is a **powerful tool** for solving sequential decision-making problems.

From **games** (AlphaGo, Dota 2) to **robotics** and **language models** (ChatGPT) — RL continues to transform AI.

**Next steps:**

1. Implement the final project
2. Study advanced topics (multi-agent, model-based)
3. Read new papers (arxiv.org/list/cs.LG/recent)
4. Participate in competitions (Kaggle RL, NeurIPS competitions)
5. Apply what you've learned to real-world tasks

---

**🎓 Congratulations on completing the course!**

**Author:** Denis Samatov, TPU / 2025  
**Contact:** [GitHub](https://github.com/denissamatov) · [Email](mailto:denissamatov@example.com)

---

## 📚 Complete list of recommended reading

### Books

1. **Sutton & Barto** — *Reinforcement Learning: An Introduction (2nd ed.)*
2. **Maxim Lapan** — *Deep Reinforcement Learning Hands-On*
3. **Andrea Lonza** — *Reinforcement Learning Algorithms with Python*
4. **Csaba Szepesvári** — *Algorithms for Reinforcement Learning*

### Online courses

1. **David Silver's RL Course** (DeepMind/UCL)
2. **CS285: Deep RL** (UC Berkeley, Sergey Levine)
3. **OpenAI Spinning Up** (a practical RL guide)

### Research papers (must-read)

1. DQN: *Playing Atari with Deep RL* (Mnih et al., 2013)
2. PPO: *Proximal Policy Optimization* (Schulman et al., 2017)
3. RLHF: *InstructGPT* (Ouyang et al., 2022)
4. DPO: *Direct Preference Optimization* (Rafailov et al., 2023)

---

✅ **Course complete!** Best of luck applying Reinforcement Learning! 🚀
