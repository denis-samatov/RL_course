# Theoretical Note #14

## Topic: Proximal Policy Optimization (PPO) and Trust Region Policy Optimization (TRPO)

> **Related to:** [note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md) — Policy Gradient, REINFORCE · [note_12_actor_critic_a2c.md](note_12_actor_critic_a2c.md) — Actor-Critic, A2C

---

## 1. Motivation: the problems with Policy Gradient

Recall **REINFORCE** and **A2C** ([note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md), [note_12_actor_critic_a2c.md](note_12_actor_critic_a2c.md)):

- ✅ Can train on continuous actions
- ✅ Optimize the policy directly
- ❌ **High gradient variance** → slow learning
- ❌ **Sensitivity to the learning rate** → easy to wreck the policy
- ❌ **Sample inefficiency** → need many trajectories

### The main problem: catastrophic forgetting

Large policy updates can trigger **policy collapse**:

```
Episode 100: Reward = 200 (a good policy)
   ↓ [A large gradient step]
Episode 101: Reward = -50 (the policy is wrecked)
   ↓ [No way to recover]
Episode 200: Reward = -50 (stuck in a bad local minimum)
```

**Intuition:**

> "Too large a policy update can push the agent outside its 'trust region', from which it can't recover."

---

## 2. Trust region methods: the core idea

**The solution:** Limit **how much the policy changes** at every step.

### KL divergence as a distance metric

We use **KL divergence** to measure the "distance" between the old and new policies:

$$
D_{KL}(\pi_{\text{old}} \| \pi_{\text{new}}) = \mathbb{E}_{s \sim \rho_{\pi_{\text{old}}}} \left[ D_{KL}(\pi_{\text{old}}(\cdot|s) \| \pi_{\text{new}}(\cdot|s)) \right]
$$

For discrete actions:

$$
D_{KL}(\pi_{\text{old}} \| \pi_{\text{new}}) = \sum_a \pi_{\text{old}}(a|s) \log \frac{\pi_{\text{old}}(a|s)}{\pi_{\text{new}}(a|s)}
$$

**Properties of KL divergence:**

- $D_{KL} \geq 0$, with equality only when $\pi_{\text{old}} = \pi_{\text{new}}$
- Asymmetric: $D_{KL}(p\|q) \neq D_{KL}(q\|p)$
- A measure of "informational distance"

---

### The trust region constraint

Idea: maximize the objective subject to a **constraint on the KL divergence**:

$$
\max_\theta \mathbb{E}_{\pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s,a) \right]
$$

$$
\text{subject to } \mathbb{E}_{s \sim \rho_{\pi_{\theta_{\text{old}}}}} \left[ D_{KL}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s)) \right] \leq \delta
$$

where:
- $\delta$ is the maximum allowed KL divergence (typically 0.01-0.05)
- $A^{\pi}(s,a)$ is the advantage function

**Intuition:**

> "Improve the policy, but don't stray too far from the current version."

---

## 3. TRPO: Trust Region Policy Optimization

**TRPO** (Schulman et al., 2015) was the first successful implementation of the trust-region idea.

### The surrogate objective

Instead of directly maximizing $J(\theta)$, we use the **Conservative Policy Iteration (CPI)** objective:

$$
L^{CPI}(\theta) = \mathbb{E}_{t} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t \right] = \mathbb{E}_t \left[ r_t(\theta) \hat{A}_t \right]
$$

where the **probability ratio** is:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

**Why does this work?**

- At $\theta = \theta_{\text{old}}$: $r_t = 1$, and the gradients match the policy gradient
- It lets us **reuse old trajectories** (off-policy)

---

### Constrained optimization

TRPO solves:

$$
\max_\theta L^{CPI}(\theta) \quad \text{s.t.} \quad \bar{D}_{KL}(\theta_{\text{old}}, \theta) \leq \delta
$$

where $\bar{D}_{KL}$ is the average KL divergence over states.

**Solved via conjugate gradients:**

1. Approximate the KL constraint locally (second order):
   $$
   \bar{D}_{KL} \approx \frac{1}{2} (\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}})
   $$
   where $F$ is the Fisher Information Matrix

2. Solve for $F^{-1} g$ via **conjugate gradient**

3. A **line search** to guarantee improvement and satisfy the constraint

---

### TRPO's problems

| Problem | Description |
|----------|-------------|
| **Implementation complexity** | Conjugate gradient, line search, Hessian-vector products |
| **Computational cost** | Repeated computation of KL and its derivatives |
| **Sensitivity to hyperparameters** | Backtracking line-search coefficients |
| **Trouble with RNNs/Transformers** | The Fisher matrix becomes enormous |

**Conclusion:**

> TRPO is theoretically elegant, but **practically complex**. We need a simpler alternative!

---

## 4. PPO: Proximal Policy Optimization

**PPO** (Schulman et al., 2017) is a simplified version of TRPO that became an **industry standard**.

### Two versions of PPO

1. **PPO-Clip** (used more often)
2. **PPO-Penalty** (an adaptive KL penalty)

---

### PPO-Clip

**Idea:** Instead of a constraint, **clip** the surrogate objective.

**The objective function:**

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]
$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio
- $\epsilon$ is the clip range (typically 0.1 or 0.2)
- $\text{clip}(x, a, b) = \max(a, \min(b, x))$

**Intuition:**

$$
\text{clip}(r_t, 1-\epsilon, 1+\epsilon) \in [0.8, 1.2] \quad (\text{for } \epsilon=0.2)
$$

- If $r_t$ is close to 1 → this behaves like a regular policy gradient
- If $r_t$ is far from 1 → we clip it, so it can't stray too far

---

### Breaking PPO-Clip down by case

**Case 1:** $\hat{A}_t > 0$ (a good action, we want to raise its probability)

$$
L^{CLIP} = \min(r_t \hat{A}_t, \, (1+\epsilon) \hat{A}_t)
$$

- If $r_t < 1+\epsilon$: use $r_t \hat{A}_t$ (keep increasing)
- If $r_t > 1+\epsilon$: use $(1+\epsilon) \hat{A}_t$ (stop the increase)

**Case 2:** $\hat{A}_t < 0$ (a bad action, we want to lower its probability)

$$
L^{CLIP} = \max(r_t \hat{A}_t, \, (1-\epsilon) \hat{A}_t)
$$

- If $r_t > 1-\epsilon$: use $r_t \hat{A}_t$ (keep decreasing)
- If $r_t < 1-\epsilon$: use $(1-\epsilon) \hat{A}_t$ (stop the decrease)

**Visually:**

```
For A > 0:
  L(r) = min(r*A, (1+ε)*A)
       |     /-------- (clipped)
       |    /
       |   /
       |  /
  -----+--------
     1-ε  1  1+ε   r

For A < 0:
       |
  -----+--------
       |\
       | \
       |  \  (clipped)
       |   \----
              r
```

---

### PPO-Penalty

An alternative approach: an **adaptive KL penalty** instead of clipping.

**The objective function:**

$$
L^{KLPEN}(\theta) = \mathbb{E}_t \left[ r_t(\theta)\hat{A}_t - \beta \cdot D_{KL}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \right]
$$

where $\beta$ is the penalty coefficient, **adaptively adjusted**:

```python
if d_kl < target_kl / 1.5:
    beta = beta / 2  # Reduce the penalty
elif d_kl > target_kl * 1.5:
    beta = beta * 2  # Increase the penalty
```

**Comparison:**

| Method | Advantages | Drawbacks |
|-------|--------------|------------|
| **PPO-Clip** | Simple, no $\beta$ hyperparameter | Less interpretable |
| **PPO-Penalty** | Explicitly controls KL | Needs $\beta$ to be adapted |

**In practice:** PPO-Clip is used more often, for its simplicity.

---

## 5. Generalized Advantage Estimation (GAE)

**The problem:** The advantage $A(s,a)$ has a bias-variance trade-off:

- **TD error** ($\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$): low variance, high bias
- **Monte Carlo** ($\hat{A}_t = G_t - V(s_t)$): unbiased, high variance

**GAE** (Schulman et al., 2016) is an exponentially weighted average of TD errors.

### The GAE formula

$$
\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

**Expanded:**

$$
\hat{A}_t^{GAE} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots
$$

**The parameter $\lambda \in [0,1]$:**

- $\lambda = 0$: $\hat{A}_t = \delta_t$ (TD, high bias, low variance)
- $\lambda = 1$: $\hat{A}_t = G_t - V(s_t)$ (MC, low bias, high variance)
- $\lambda \in (0,1)$: a **trade-off** (usually 0.95-0.99)

**A practical implementation:**

```python
def compute_gae(rewards, values, dones, gamma=0.99, lambda_gae=0.95):
    """
    Computes GAE advantages.
    
    Args:
        rewards: List[float] — rewards, length T
        values: List[float] — V(s), length T+1 (including the bootstrap V(s_T+1))
        dones: List[bool] — termination flags, length T
        gamma: Discount factor
        lambda_gae: the GAE lambda parameter
        
    Returns:
        advantages: List[float] of length T
    """
    advantages = []
    gae = 0.0
    
    # Iterate in reverse order
    for t in reversed(range(len(rewards))):
        # TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t+1] * (1 - int(dones[t])) - values[t]
        
        # GAE: A_t = δ_t + (γλ) δ_{t+1} + (γλ)^2 δ_{t+2} + ...
        gae = delta + gamma * lambda_gae * (1 - int(dones[t])) * gae
        
        advantages.insert(0, gae)
    
    return advantages

# Usage example:
# Assume values contains V(s_0), ..., V(s_T), V(s_{T+1})
advantages = compute_gae(rewards, values, dones, gamma=0.99, lambda_gae=0.95)
```

---

## 6. The PPO agent's architecture

### Actor and Critic networks

**Two options:**

1. **A shared backbone** (more economical):
   ```
   Input (state) → [Shared Layers] → Actor Head
                                   → Critic Head
   ```

2. **Separate networks** (more flexible):
   ```
   Input (state) → [Actor Layers] → π(a|s)
   Input (state) → [Critic Layers] → V(s)
   ```

**A PyTorch example:**

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Actor head
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, state):
        features = self.shared(state)
        # Actor output (for continuous actions)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)
        # Critic output
        value = self.critic(features)
        return action_mean, action_std, value
```

---

### Discrete vs continuous actions

**Discrete actions:**

```python
# The actor outputs logits for each action
logits = self.actor(state)
dist = Categorical(logits=logits)
action = dist.sample()
log_prob = dist.log_prob(action)
```

**Continuous actions (Gaussian policy):**

```python
# The actor outputs a mean and std
mean, std = self.actor(state)
dist = Normal(mean, std)
action = dist.sample()
log_prob = dist.log_prob(action).sum(dim=-1)
```

---

## 7. The full PPO algorithm

**Pseudocode:**

```
for iteration in range(N):
    # 1. Collect trajectories with the current policy π_old
    trajectories = collect_rollouts(env, π_old, n_steps)
    
    # 2. Compute advantages via GAE
    advantages = compute_gae(trajectories, V, γ, λ)
    
    # 3. Normalize the advantages (optional)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 4. Several epochs of optimization on the collected data
    for epoch in range(K):  # K = 3-10
        for batch in mini_batches(trajectories):
            # Compute the current log_probs and values
            log_probs_new, values_new = π_θ(batch)
            
            # Probability ratio
            ratio = exp(log_probs_new - batch.log_probs_old)
            
            # PPO-Clip loss
            surr1 = ratio * batch.advantages
            surr2 = clip(ratio, 1-ε, 1+ε) * batch.advantages
            policy_loss = -min(surr1, surr2).mean()
            
            # Value loss (MSE)
            value_loss = (values_new - batch.returns)^2.mean()
            
            # Entropy bonus (for exploration)
            entropy = π_θ.entropy().mean()
            
            # The total loss
            loss = policy_loss + c1 * value_loss - c2 * entropy
            
            # Parameter update
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters, max_norm=0.5)
            optimizer.step()
```

**Key hyperparameters:**

| Parameter | Typical value | Description |
|----------|-------------------|----------|
| `ε` (clip_range) | 0.1 - 0.2 | The clipping range for the ratio |
| `K` (epochs) | 3 - 10 | Optimization epochs per batch |
| `γ` (gamma) | 0.99 | Discount factor |
| `λ` (lambda_gae) | 0.95 | The GAE lambda |
| `c1` (value_coef) | 0.5 - 1.0 | The value loss's weight |
| `c2` (entropy_coef) | 0.01 | The entropy bonus's weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `lr` | 3e-4 | The learning rate |
| `batch_size` | 64 - 256 | The mini-batch size |
| `n_steps` | 2048 | Steps collected per rollout |

---

## 8. Practical PPO tricks

### 1. Advantage normalization

```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Why:**
- Stabilizes training
- Makes the loss's scale independent of the reward scale

---

### 2. Value function clipping

**The problem:** The value function can change too drastically.

**The solution:** Clip the value loss too:

```python
v_pred_clipped = v_old + torch.clamp(v_pred - v_old, -ε, ε)
value_loss = torch.max(
    (v_pred - returns) ** 2,
    (v_pred_clipped - returns) ** 2
).mean()
```

---

### 3. Learning rate annealing

```python
# Linearly decay the LR
lr_new = lr_init * (1 - iteration / max_iterations)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr_new
```

**Why:** Early iterations take large steps for exploration; later ones take small steps for fine-tuning.

---

### 4. Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
```

**Why:** Prevents gradient explosion.

---

### 5. Multiple environments (a vectorized env)

```python
from gymnasium.vector import SyncVectorEnv

envs = SyncVectorEnv([make_env() for _ in range(8)])
```

**Advantages:**
- Parallel trajectory collection
- Less correlation between samples
- Faster training

---

### 6. State normalization (running mean/std)

```python
# Updating the running statistics
running_mean = alpha * obs + (1 - alpha) * running_mean
running_std = alpha * abs(obs - running_mean) + (1 - alpha) * running_std

# Normalizing
obs_normalized = (obs - running_mean) / (running_std + 1e-8)
```

**Why:** Many environments have observations on very different scales → normalization stabilizes training.

---

## 9. Comparing methods

| Method | Year | Complexity | Stability | Sample Efficiency | Popularity |
|-------|-----|-----------|--------------|-------------------|--------------|
| **REINFORCE** | 1992 | Low | Low | Low | 🔵 Educational |
| **A2C** | 2016 | Moderate | Moderate | Moderate | 🟢 Practical |
| **TRPO** | 2015 | High | High | Moderate | 🟡 Historical |
| **PPO** | 2017 | Moderate | High | High | 🟢🟢 State-of-the-art |

**When to use PPO:**

✅ Continuous actions (robotics, control)  
✅ Discrete actions with a large action space  
✅ Stability and reproducibility matter  
✅ Limited compute resources  

**When NOT to use PPO:**

❌ A very small dataset (offline RL is a better fit)  
❌ Maximal sample efficiency is required (SAC, TD3 are better)  
❌ A simple discrete environment (DQN is enough)  

---

## 10. PPO in industry

### OpenAI Five (Dota 2)

- Trained with PPO
- 256 GPUs, 128,000 CPU cores
- 10 months of in-game time
- Beat professional players

### ChatGPT RLHF

- PPO for fine-tuning on human preferences
- A key component of InstructGPT → ChatGPT
- A KL penalty against the SFT model for stability

### DeepMind's AlphaStar (StarCraft II)

- Uses a variant of PPO
- Scales to tens of thousands of games in parallel

---

## 11. Summary

| Concept | Description |
|-----------|----------|
| **Trust Region** | Limiting policy change via KL divergence |
| **TRPO** | Constrained optimization with conjugate gradient |
| **PPO-Clip** | Clipping the probability ratio to limit updates |
| **PPO-Penalty** | An adaptive KL penalty in the objective |
| **GAE** | A bias-variance trade-off for advantage estimation |
| **Multiple Epochs** | Reusing trajectories for sample efficiency |

**Key takeaways:**

1. **PPO solves Policy Gradient's instability problem** via clipping or a KL penalty
2. **GAE provides an optimal bias-variance trade-off** for the advantage
3. **PPO is the industry standard** for continuous control and RLHF
4. **Simplicity of implementation** + **stability** + **sample efficiency** = PPO's success

---

## 12. Connections to earlier sessions

- **[note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md):** REINFORCE — the baseline for PG methods
- **[note_12_actor_critic_a2c.md](note_12_actor_critic_a2c.md):** A2C — the Actor-Critic architecture PPO builds on
- **[note_13_dynamic_programming.md](note_13_dynamic_programming.md):** GPI — the concept of iterative policy improvement

---

## 13. Further reading

Recommended sources:

- **Original papers:**
  - TRPO: *Trust Region Policy Optimization* (Schulman et al., 2015)
  - PPO: *Proximal Policy Optimization Algorithms* (Schulman et al., 2017)
  - GAE: *High-Dimensional Continuous Control Using GAE* (Schulman et al., 2016)

- **Implementations:**
  - [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
  - [Stable-Baselines3: PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

- **Video:**
  - *Lecture on TRPO/PPO* by John Schulman (UC Berkeley)

---

## 14. Hands-on assignment

`code/14_ppo_trpo/` contains a full PPO agent for `BipedalWalker-v3`.

**Experiments:**
- The effect of the clip range $\epsilon$ (0.1, 0.2, 0.3)
- The effect of GAE $\lambda$ (0.9, 0.95, 0.99)
- Shared vs separate networks for Actor/Critic
- Comparison against A2C on the same environment

---

**Next:** [note_15_rlhf_pipeline.md](note_15_rlhf_pipeline.md) — RLHF: Reinforcement Learning from Human Feedback
