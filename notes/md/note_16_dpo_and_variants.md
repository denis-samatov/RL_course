# Theoretical Note #16

## Topic: DPO — Direct Preference Optimization

> **Related to:** [note_15_rlhf_pipeline.md](note_15_rlhf_pipeline.md) — The RLHF pipeline

---

## 1. Motivation: simplifying RLHF

**PPO-RLHF's problems:**

1. **Complexity:** 4 models (Actor, Critic, Reference, RM)
2. **Instability:** Requires careful hyperparameter tuning
3. **Computational cost:** Enormous resources
4. **Reward hacking:** The RM can be fooled

**DPO's idea:** Can we train **directly on preferences**, bypassing the Reward Model and RL entirely?

---

## 2. DPO's theoretical basis

### The optimal policy in RLHF

In RLHF we solve:

$$
\max_\pi \mathbb{E}_{x, y \sim \pi} \left[ r(x, y) - \beta D_{KL}(\pi \| \pi_{ref}) \right]
$$

**Theorem (Rafailov et al., 2023):** The optimal policy has a **closed-form solution**:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta}r(x,y)\right)
$$

where $Z(x)$ is the partition function.

### Solving for the reward:

$$
r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)
$$

### Substituting into Bradley-Terry:

$$
P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))
$$

$$
= \sigma\left( \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)} \right)
$$

**The key observation:** $Z(x)$ **cancels out**! We can optimize the policy directly.

---

## 3. The DPO loss

**The final loss function:**

$$
L_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

**Simplified:**

$$
L_{DPO} = -\mathbb{E} \left[ \log \sigma \left( \beta \left[ \log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x) \right] - \beta \left[ \log \pi_{ref}(y_w|x) - \log \pi_{ref}(y_l|x) \right] \right) \right]
$$

**Intuition:**
- We raise $\log \pi_\theta(y_w|x)$ (the good response)
- We lower $\log \pi_\theta(y_l|x)$ (the bad response)
- We stay controlled via $\pi_{ref}$ (an implicit KL penalty!)

---

## 4. DPO vs PPO-RLHF

| Aspect | PPO-RLHF | DPO |
|--------|----------|-----|
| **Stages** | 3 (SFT → RM → PPO) | 2 (SFT → DPO) |
| **Models in memory** | 4 (Actor, Critic, RM, Ref) | 2 (Policy, Reference) |
| **Reward model** | Needed (trained separately) | Not needed |
| **RL algorithm** | PPO (complex) | No RL (just supervised) |
| **Implementation complexity** | High | Low |
| **Stability** | Moderate (needs tuning) | High |
| **Sample efficiency** | Moderate | High |
| **Flexibility** | High (the reward can be changed) | Low (a fixed dataset) |
| **Online data collection** | ✅ Yes | ❌ No |

---

## 5. DPO's advantages

### 1. Simplicity
- Trains like supervised classification
- No complex RL loop
- Fewer hyperparameters

### 2. Stability
- No reward hacking (no separate RM)
- An implicit KL penalty is built into the formula
- No need to fine-tune PPO

### 3. Efficiency
- Less memory (2 models instead of 4)
- Trains faster (a single pass over the preference data)
- Sample efficient

### 4. Interpretability
- A direct connection to the preferences
- A clear loss function

---

## 6. DPO's drawbacks

### 1. Offline only
- Requires a fixed preference dataset
- Can't collect new data online
- Limited by the SFT model's quality

### 2. Less flexible
- Can't combine multiple reward functions
- Hard to add constraints (safety, length)

### 3. Dependence on the reference policy
- Quality depends on $\pi_{ref}$ (usually SFT)
- If SFT is poor → DPO will be poor too

---

## 7. The DPO algorithm

```python
# 1. Initialization
π_θ = copy(π_SFT)  # The policy being trained
π_ref = π_SFT      # Reference (frozen)
β = 0.1            # Temperature

# 2. Training
for epoch in range(N):
    for (x, y_w, y_l) in preference_dataset:
        # Compute log-probs
        log_prob_w = π_θ.log_prob(y_w | x)
        log_prob_l = π_θ.log_prob(y_l | x)
        
        # Reference log-probs (frozen)
        with torch.no_grad():
            log_prob_w_ref = π_ref.log_prob(y_w | x)
            log_prob_l_ref = π_ref.log_prob(y_l | x)
        
        # DPO loss
        logits_w = β * (log_prob_w - log_prob_w_ref)
        logits_l = β * (log_prob_l - log_prob_l_ref)
        
        loss = -log_sigmoid(logits_w - logits_l)
        
        # Update (IMPORTANT: zero the gradients before backward!)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 8. DPO variants

### IPO (Identity Preference Optimization)

Uses MSE instead of log-sigmoid:

$$
L_{IPO} = \mathbb{E} \left[ \left( \frac{1}{2\beta} \left( r_w - r_l \right) - 1 \right)^2 \right]
$$

**Advantage:** Less sensitive to outliers.

### KTO (Kahneman-Tversky Optimization)

Accounts for the **asymmetry** in human preferences (loss aversion):

$$
L_{KTO} = \lambda_{pos} \mathbb{E}[loss_{pos}] + \lambda_{neg} \mathbb{E}[loss_{neg}]
$$

where $\lambda_{pos} < \lambda_{neg}$ (people react more strongly to something bad).

### ORPO (Odds Ratio Preference Optimization)

**Merges SFT and DPO into a single stage:**

$$
L_{ORPO} = L_{SFT} + \lambda L_{DPO}
$$

**Advantage:** Saves a full training stage.

---

## 9. When to use DPO vs PPO

### Use DPO if:

✅ You have a fixed preference dataset  
✅ You need simplicity and stability  
✅ Compute resources are limited  
✅ Online data collection isn't required  

### Use PPO if:

✅ You need flexibility in the reward (multiple objectives)  
✅ You can collect data online  
✅ You need a complex, composable reward function  
✅ You have the resources to tune and debug  

---

## 10. Practical recommendations for DPO

### 1. Preference data quality

- Preferences must be **clear and consistent**
- Annotator agreement rate > 70%
- Balanced coverage of different prompt types

### 2. Choosing β

| β | Effect |
|---|--------|
| 0.01 | Weak control, a strong update |
| 0.1 | Baseline (recommended) |
| 0.5 | Strong control, a conservative update |

### 3. Learning rate

- Lower than for SFT: `1e-6` to `5e-6`
- Linear warmup for the first 10% of steps

### 4. Batch size

- Larger = more stable: 32-128 pairs
- Gradient accumulation if memory is tight

---

## 11. Results in industry

### Zephyr-7B (Hugging Face)

- DPO on 60K preference pairs
- Outperforms Llama-2-70B-Chat on MT-Bench
- Trained in **a few hours** on 8× A100s

### Mistral-7B-Instruct

- Uses DPO for alignment
- State-of-the-art among 7B models

### Intel Neural Chat

- DPO + distillation
- Efficient alignment for edge deployment

---

## 12. Summary

| Concept | Description |
|-----------|----------|
| **The DPO loss** | Direct optimization on preferences, no RM |
| **The closed-form solution** | The optimal policy expressed via the reward |
| **Implicit KL** | Control via the reference policy is built in |
| **Supervised learning** | Trains like classification, not RL |
| **2 stages** | SFT → DPO (simpler than SFT → RM → PPO) |

**Key takeaways:**

1. **DPO is a simpler alternative to RLHF**
2. **It's theoretically grounded** via a closed-form solution
3. **It's practically effective** for offline preference optimization
4. **The trade-off:** simplicity vs flexibility

---

## 13. Hands-on assignment

`code/16_dpo/` implements DPO and compares it against PPO on the same dataset.

**Experiments:**
- The effect of β
- Sample efficiency: DPO vs PPO
- Quality: win rate against a baseline

---

**Next:** [note_17_final_project.md](note_17_final_project.md) — Final project and safety
