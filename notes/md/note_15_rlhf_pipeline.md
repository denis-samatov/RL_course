# Theoretical Note #15

## Topic: RLHF — Reinforcement Learning from Human Feedback

> **Related to:** [note_14_ppo_trpo.md](note_14_ppo_trpo.md) — PPO · [note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md) — Policy Gradient

---

## 1. Motivation: aligning LLMs with human values

**The problem:** Large Language Models (LLMs), trained on huge text corpora, can:

- Generate toxic content
- Give harmful advice
- Fail to follow user instructions
- "Hallucinate" (make up facts)

**Why isn't Supervised Fine-Tuning (SFT) enough?**

- Collecting thousands of hand-written response examples is expensive
- Hard to cover every possible scenario
- The "correct" response is subjective (tone, style, verbosity)

**The solution: RLHF** — train the model on human **preferences**, rather than absolute answers.

---

## 2. RLHF's history

| Year | Model | Contribution |
|-----|--------|-------|
| 2017 | Deep RL from Human Preferences | The first RLHF work, on Atari |
| 2020 | GPT-3 | Showed LLM potential, but with no alignment |
| 2022 | **InstructGPT** | The first large-scale RLHF for language → ChatGPT |
| 2023 | GPT-4, Claude, Llama 2 | RLHF becomes standard |

**Key insight:** It's easier for a person to **compare** two responses than to write a perfect one.

---

## 3. The three-stage RLHF pipeline

```
Pretrained LLM
      ↓
[1] Supervised Fine-Tuning (SFT)
      ↓
[2] Reward Model Training (RM)
      ↓
[3] RL Fine-Tuning (PPO)
      ↓
Aligned LLM
```

---

### Stage 1: Supervised Fine-Tuning (SFT)

**Goal:** Adapt a pretrained LLM to the "instruction → response" format.

**Data:**
- Pairs $(x, y)$: a prompt + a high-quality response
- Typically 10K-50K examples (human-collected)

**Training:** The standard language-modeling loss

$$
L_{SFT}(\theta) = -\sum_{t=1}^T \log p_\theta(y_t \mid y_{<t}, x)
$$

**Example:**

```
x (prompt): "Explain quantum computing to a 5-year-old"
y (response): "Imagine you have a magic coin..."
```

**Result:** $\pi_{SFT}$ — a model that can follow instructions, but isn't perfect.

---

### Stage 2: Reward Model Training (RM)

**Goal:** Train a model that predicts a response's "quality" from a human's perspective.

**Data: preferences**

For a single prompt $x$, we collect 4-9 responses from $\pi_{SFT}$, and a human ranks them:

$$
(x, y_w, y_l) \quad \text{where } y_w \succ y_l
$$

- $y_w$ — the "winning" response (preferred)
- $y_l$ — the "losing" response

Typically 50K-100K such pairs.

**The reward model:** A scalar function $r_\phi(x, y) \in \mathbb{R}$

Initialized from $\pi_{SFT}$, with the last layer replaced by a linear scalar head.

**Loss: the Bradley-Terry model**

The probability that $y_w$ is preferred over $y_l$:

$$
P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))
$$

where $\sigma$ is the sigmoid function.

**Loss:**

$$
L_{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

**Intuition:** Maximize the reward gap between good and bad responses.

---

### Stage 3: RL Fine-Tuning (PPO)

**Goal:** Optimize the policy $\pi_\theta$ to maximize the RM's reward, **BUT** without straying too far from $\pi_{SFT}$.

**The objective function:**

$$
\max_\theta \mathbb{E}_{x \sim D_{prompts}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{SFT}) \right]
$$

where:
- $r_\phi(x, y)$ is the RM's reward
- $\beta$ is the KL-penalty coefficient (typically 0.01-0.1)
- $D_{KL}$ is the KL divergence from the reference policy $\pi_{SFT}$

**The KL penalty:**

$$
D_{KL}(\pi_\theta \| \pi_{SFT}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{SFT}(y|x)} \right]
$$

**Why the KL penalty?**

1. **Prevents reward hacking** (the model finding loopholes in the RM)
2. **Preserves language ability** (doesn't forget grammar, facts)
3. **Stabilizes training**

---

### PPO for LLMs

We use **PPO** (see [note_14_ppo_trpo.md](note_14_ppo_trpo.md)) with a few modifications:

**The algorithm:**

```
for iteration in range(N):
    # 1. Sample prompts
    prompts = sample_batch(D_prompts)
    
    # 2. Generate responses with π_θ
    responses = π_θ.generate(prompts)
    
    # 3. Compute rewards
    rewards = r_φ(prompts, responses)
    
    # 4. Compute the KL penalty
    kl_penalty = KL(π_θ || π_SFT)
    
    # 5. Compute advantages via GAE
    advantages = compute_gae(rewards - β * kl_penalty, V_critic)
    
    # 6. PPO update (multiple epochs)
    for epoch in range(K):
        for batch in minibatches:
            # PPO-Clip loss
            policy_loss = ppo_clip_loss(batch, advantages)
            
            # Value loss
            value_loss = (V(s) - returns)^2
            
            # Update
            loss = policy_loss + c1 * value_loss
            optimizer.step()
```

**Differences from ordinary PPO:**

- **4 models held in memory at once:**
  1. Actor $\pi_\theta$ (being trained)
  2. Critic $V_\psi$ (being trained)
  3. Reference $\pi_{SFT}$ (frozen)
  4. Reward Model $r_\phi$ (frozen)
  
- **Enormous compute requirements:** GPT-4 was trained on thousands of GPUs

---

## 4. The KL penalty: a critically important component

### Without a KL penalty: catastrophe

```
Iteration 1: reward = 5.0, text = "Great answer!"
Iteration 10: reward = 8.0, text = "!!!!!!!!!!!!!!!!!!!"
Iteration 50: reward = 15.0, text = "kdfj;alskdjf;laksdjf"
```

**What happens:** The model finds **adversarial examples** that get a high reward from the RM but are meaningless.

### With a KL penalty: stability

$$
\text{Total Reward} = r_\phi(x, y) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{SFT})
$$

If the model strays too far from $\pi_{SFT}$ → a large KL penalty → low total reward.

**Typical values of $\beta$:**

| $\beta$ | Effect |
|---------|--------|
| 0.001 | Weak control, risk of reward hacking |
| 0.01 | A good balance (standard) |
| 0.1 | Strong control, slow improvement |

---

## 5. RLHF's problems and challenges

### 1. Reward hacking

**The problem:** The RM is trained on a limited dataset → the model finds out-of-distribution examples with a high reward.

**Example:**
- The RM learned that longer responses are better
- The model generates wordy, but useless, responses

**Solutions:**
- The KL penalty (the primary mechanism)
- Adversarial testing
- Iteratively updating the RM

---

### 2. Distribution shift

**The problem:** The RM is trained on $\pi_{SFT}$'s responses, but evaluates $\pi_\theta$'s responses.

Once $\pi_\theta \neq \pi_{SFT}$ → the RM has to extrapolate → unreliable rewards.

**Solutions:**
- The KL penalty limits the shift
- Iterative RLHF (periodically update the RM)

---

### 3. Reward over-optimization

**Observation:** With too much training, quality **gets worse**, even as the reward keeps rising!

```
RM Reward: ↑↑↑↑↑↑
Human Eval: ↑↑↑↓↓↓ (Goodhart's Law)
```

**Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure."

**Solutions:**
- Early stopping based on human evaluation
- Regular metric monitoring

---

### 4. Computational cost

**Requirements:**

- 4 models held in memory (Actor, Critic, Reference, RM)
- For a 7B model: ~40-60GB of VRAM
- For GPT-3.5/4: thousands of GPUs, weeks of training

**Solutions:**
- LoRA, QLoRA for parameter-efficient fine-tuning
- Distillation
- DPO (see [note_16_dpo_and_variants.md](note_16_dpo_and_variants.md)) — a more efficient alternative

---

## 6. RLHF evaluation metrics

| Metric | Description | How it's measured |
|---------|----------|--------------|
| **RM Reward** | The reward from the trained RM | Automatically |
| **KL Divergence** | Distance from $\pi_{SFT}$ | $D_{KL}(\pi_\theta \| \pi_{SFT})$ |
| **Human Eval** | Manual quality assessment | Win rate against a baseline |
| **Perplexity** | Preservation of language ability | On held-out data |
| **Safety Metrics** | Toxicity, bias, harmful content | Perspective API, red teaming |
| **Task Performance** | Accuracy on downstream tasks | Benchmarks (MMLU, HumanEval) |

**The key trade-off:**

$$
\text{RM Reward} \uparrow \quad \leftrightarrow \quad \text{KL Divergence} \uparrow
$$

You need to find a balance by tuning $\beta$.

---

## 7. RLHF variants and extensions

### Constitutional AI (Anthropic)

- The model critiques and improves its own responses
- Less reliance on human annotation

### RLAIF (RL from AI Feedback)

- Uses a strong LLM (GPT-4) instead of people to generate preferences
- Cheaper and scales faster

### Iterative RLHF

```
Round 1: SFT → RM1 → PPO1 → Model v1
Round 2: Collect new preferences on Model v1 → RM2 → PPO2 → Model v2
...
```

### Multi-objective RLHF

$$
\max_\theta \mathbb{E} \left[ \alpha_1 r_{\text{helpful}} + \alpha_2 r_{\text{harmless}} + \alpha_3 r_{\text{honest}} - \beta D_{KL} \right]
$$

Trains multiple reward models for different objectives.

---

## 8. Practical tips for RLHF

### 1. Quality > quantity for preferences

- 10K high-quality pairs beat 100K noisy ones
- Annotator agreement matters (agreement rate > 70%)

### 2. A balanced dataset

- Cover different prompt types (questions, instructions, creative writing)
- Avoid strong biases in the preferences

### 3. KL warm-up

```python
# Gradually increase β
β = β_min + (β_max - β_min) * min(1.0, iteration / warmup_steps)
```

### 4. Monitor the KL

If $D_{KL} > 10$: the model has strayed too far → increase $\beta$ or stop early.

### 5. Length normalization

$$
r_{\text{normalized}} = \frac{r_\phi(x, y)}{\text{length}(y)}
$$

Prevents a bias toward long responses.

---

## 9. RLHF in industry

### OpenAI: InstructGPT → ChatGPT → GPT-4

- **InstructGPT (2022):** The first large-scale RLHF applied to GPT-3
- **ChatGPT (Nov 2022):** RLHF applied to a conversational model
- **GPT-4 (Mar 2023):** RLHF scaled to a multimodal model

**Key insights:**
- RLHF is critical for "human-like" behavior
- A KL penalty against SFT is absolutely necessary

### Anthropic: Claude

- Constitutional AI + RLHF
- An emphasis on harmlessness and honesty

### Meta: Llama 2-Chat

- An open-source model trained with RLHF
- The process is described in detail in the paper

---

## 10. Summary

| Concept | Description |
|-----------|----------|
| **The RLHF pipeline** | SFT → RM → PPO, in three stages |
| **The reward model** | Trained on pairwise preferences (Bradley-Terry) |
| **The KL constraint** | $\beta D_{KL}(\pi_\theta \| \pi_{SFT})$ prevents reward hacking |
| **PPO for LLMs** | 4 models in memory, enormous compute |
| **Reward hacking** | The main problem, addressed by the KL penalty |
| **Goodhart's Law** | Over-optimizing the RM degrades actual quality |

**Key takeaways:**

1. **RLHF is the industry standard** for LLM alignment
2. **The KL penalty is critically important** for stability
3. **It's computationally expensive**, but necessary for quality
4. **Alternatives (DPO) are actively being developed**

---

## 11. Connections to earlier sessions

- **[note_14_ppo_trpo.md](note_14_ppo_trpo.md):** PPO — the RL algorithm behind RLHF
- **[note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md):** Policy Gradient — the foundation PPO builds on
- **[note_12_actor_critic_a2c.md](note_12_actor_critic_a2c.md):** Actor-Critic — the architecture used for LLM RL

---

## 12. Further reading

**Key papers:**

- *Training language models to follow instructions with human feedback* (InstructGPT, 2022)
- *Constitutional AI: Harmlessness from AI Feedback* (Anthropic, 2022)
- *Llama 2: Open Foundation and Fine-Tuned Chat Models* (Meta, 2023)

**Libraries:**

- [Hugging Face TRL](https://github.com/huggingface/trl) — RLHF for transformers
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeed) — Efficient RLHF

---

## 13. Hands-on assignment

`code/15_rlhf_basics/` implements a simplified RLHF pipeline on a toy task.

**Experiments:**
- The effect of $\beta$ on quality and KL
- Reward hacking with no KL penalty
- Comparing an SFT baseline vs RLHF

---

**Next:** [note_16_dpo_and_variants.md](note_16_dpo_and_variants.md) — DPO: Direct Preference Optimization
