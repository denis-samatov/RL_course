# Theoretical Note #12

## Topic: Actor-Critic Methods and A2C

> **Related to:** [note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md) — Policy Gradients and REINFORCE · [note_09_q_learning.md](note_09_q_learning.md) — Q-Learning · [note_10_deep_q_network.md](note_10_deep_q_network.md) — Deep Q-Network

---

## 1. Motivation: combining Policy-Based and Value-Based approaches

In earlier notes we studied two main approaches:

| Approach | What it learns | Advantages | Drawbacks |
|--------|----------|--------------|------------|
| **Value-Based** (DQN) | The Q-function $Q(s,a)$ | Low variance, sample-efficient | Discrete actions only, deterministic policy |
| **Policy-Based** (REINFORCE) | The policy $\pi_\theta(a\|s)$ | Continuous actions, stochastic policy | High variance, slow convergence |

**The Actor-Critic idea:** combine both approaches to get the advantages of each while offsetting their drawbacks.

---

## 2. The Actor-Critic architecture

### Two components

1. **Actor** — the policy $\pi_\theta(a|s)$, which chooses actions.
2. **Critic** — a value function $V_\phi(s)$ or $Q_\phi(s,a)$, which evaluates the quality of actions.

### How they interact

```
        ┌─────────────────────────┐
        │      Environment (Env)   │
        └───────────┬─────────────┘
                    │ state, reward
                    ↓
        ┌─────────────────────────┐
        │   Actor (π_θ)           │ ← Chooses the action
        │   "What should I do?"   │
        └───────────┬─────────────┘
                    │ action
                    ↓
        ┌─────────────────────────┐
        │   Critic (V_φ or Q_φ)   │ ← Evaluates the action
        │   "How good was that?"  │
        └─────────────────────────┘
                    │
                    ↓ TD error / Advantage
        ┌─────────────────────────┐
        │   Parameter update       │
        │   θ ← θ + α∇J           │
        │   φ ← φ - α∇L           │
        └─────────────────────────┘
```

**Intuition:**

> The Actor learns to make decisions, and the Critic learns to evaluate them. The Critic helps the Actor understand which actions are good and which aren't.

---

## 3. Mathematical formalization

### The Actor (policy) update

We use Policy Gradient, but instead of the full return $G_t$ we use the **Critic's estimate**:

$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t \right]
$$

where $\delta_t$ is the **TD error** (temporal difference error):

$$
\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

### The Critic (value function) update

We minimize the squared error between the prediction and the target:

$$
L(\phi) = \mathbb{E} \left[ \big(r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)\big)^2 \right]
$$

The gradient:

$$
\nabla_\phi L(\phi) = - \mathbb{E} \left[ \delta_t \cdot \nabla_\phi V_\phi(s_t) \right]
$$

---

## 4. Why is Actor-Critic better than REINFORCE?

### Comparing the gradients

| Method | Gradient | Estimate |
|-------|----------|--------|
| **REINFORCE** | $\nabla \log \pi \cdot G_t$ | The full return $G_t$ (high variance) |
| **Actor-Critic** | $\nabla \log \pi \cdot \delta_t$ | The TD error $\delta_t$ (low variance) |

### The advantages of the TD error

1. **Lower variance** — $\delta_t$ is based on a single step, not the whole episode.
2. **Online learning** — parameters can be updated after every step, without waiting for the episode to end.
3. **Bootstrapping** — we use the estimate $V_\phi(s_{t+1})$ instead of the full return.

---

## 5. Advantage Actor-Critic (A2C): an improved version

### The problem with the TD error

The TD error $\delta_t = r + \gamma V(s') - V(s)$ can be biased if $V_\phi$ hasn't been well trained yet.

### The solution: the Advantage Function

Instead of the TD error, we use the **advantage function**:

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

Intuition:

> "How much **better than average** was action $a_t$ for state $s_t$?"

### Estimating Advantage via the TD error

Since $Q(s,a) = r + \gamma V(s')$, we can write:

$$
A_t \approx \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

This is the **one-step advantage estimate**.

---

## 6. Generalized Advantage Estimation (GAE)

To further reduce variance and bias, a weighted combination of n-step advantages, called **GAE**, is used.

### N-step advantage

$$
A_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i+1} + \gamma^n V(s_{t+n}) - V(s_t)
$$

### The GAE formula

$$
A_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where $\lambda \in [0,1]$ is the trade-off parameter:

* $\lambda = 0$ → 1-step TD (low variance, high bias).
* $\lambda = 1$ → Monte Carlo (high variance, low bias).

**Typical value:** $\lambda = 0.95$.

---

## 7. The A2C algorithm (Advantage Actor-Critic)

### Pseudocode

1. Initialize:
   * The Actor $\pi_\theta$ (a policy network).
   * The Critic $V_\phi$ (a value-function network).

2. For every step $t = 1, 2, \dots$:
   1. Observe state $s_t$.
   2. Choose action $a_t \sim \pi_\theta(\cdot|s_t)$.
   3. Take $a_t$, get $r_{t+1}, s_{t+1}$.
   4. Compute the TD error:
      $$
      \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
      $$
   5. Update the Actor:
      $$
      \theta \leftarrow \theta + \alpha_\theta \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t
      $$
   6. Update the Critic:
      $$
      \phi \leftarrow \phi - \alpha_\phi \cdot \nabla_\phi \big(V_\phi(s_t) - (r_{t+1} + \gamma V_\phi(s_{t+1}))\big)^2
      $$

---

## 8. Neural network architectures in A2C

### Separate networks

```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value
```

### A shared backbone

A more efficient variant is to use shared layers for feature extraction:

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # The shared encoder
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor head
        self.actor = nn.Linear(64, action_dim)
        
        # Critic head
        self.critic = nn.Linear(64, 1)
    
    def forward(self, state):
        features = self.shared(state)
        policy = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return policy, value
```

---

## 9. A hands-on example: A2C on CartPole

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Setup
env = gym.make('CartPole-v1')
model = ActorCritic(state_dim=4, action_dim=2)

# For finer control you could use separate optimizers:
# optimizer_actor = optim.Adam(model.actor.parameters(), lr=3e-4)
# optimizer_critic = optim.Adam(model.critic.parameters(), lr=1e-3)
# Here, for simplicity, we use a single one:
optimizer = optim.Adam(model.parameters(), lr=0.001)
gamma = 0.99

for episode in range(1000):
    state, _ = env.reset()
    done = False
    episode_reward = 0
    
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Forward pass
        policy, value = model(state_tensor)
        
        # Choosing the action
        action = torch.multinomial(policy, 1).item()
        
        # A step in the environment
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        episode_reward += reward
        
        # Computing the TD error
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            _, next_value = model(next_state_tensor)
        
        td_target = reward + gamma * next_value * (1 - int(done))
        td_error = td_target - value
        
        # Losses
        actor_loss = -torch.log(policy[0, action]) * td_error.detach()
        critic_loss = td_error.pow(2)
        
        loss = actor_loss + 0.5 * critic_loss
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        state = next_state
    
    if episode % 100 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}")
```

---

## 10. Entropy regularization in A2C

To keep the policy from becoming too deterministic, an **entropy bonus** is added:

$$
L_{\text{total}} = L_{\text{actor}} + c_1 \cdot L_{\text{critic}} - c_2 \cdot H(\pi)
$$

where the policy's entropy is:

$$
H(\pi_\theta(\cdot|s)) = - \sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)
$$

**Hyperparameters:**
* $c_1 = 0.5$ (the critic loss weight)
* $c_2 = 0.01$ (the entropy weight)

```python
# Adding entropy to the loss
entropy = -(policy * torch.log(policy + 1e-8)).sum(dim=-1)
loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
```

---

## 11. A3C: the asynchronous version of A2C

**A3C (Asynchronous Advantage Actor-Critic)** extends A2C with parallel training.

### Key differences

| Property | A2C | A3C |
|----------------|-----|-----|
| Parallelism | A single process | Many parallel processes (workers) |
| Updates | Synchronous | Asynchronous |
| Experience buffer | Not needed | Not needed |
| Speed | Moderate | High (thanks to parallelism) |

### How A3C works

```
Global Network (θ_global, φ_global)
         │
    ┌────┼────┬────┬────┐
    │    │    │    │    │
Worker 1 Worker 2 ... Worker N
    │    │    │    │    │
   Env1  Env2  ...  EnvN
    │    │    │    │    │
    └────┴────┴────┴────┘
         Asynchronous gradients → Global Network
```

Each worker:
1. Copies the parameters from the global network.
2. Collects a trajectory in its own environment.
3. Computes gradients.
4. Updates the global network.

---

## 12. Comparing methods: DQN vs REINFORCE vs A2C

| Property | DQN | REINFORCE | A2C |
|----------------|-----|-----------|-----|
| Method type | Value-based | Policy-based | Actor-Critic |
| Action space | Discrete | Any | Any |
| Variance | Low | High | Moderate |
| Sample efficiency | High (off-policy) | Low (on-policy) | Moderate (on-policy) |
| Stability | Unstable (overestimation) | Slow convergence | More stable |
| Requires a replay buffer | Yes | No | No |
| Convergence | To the optimal Q-function | To a local policy optimum | To a local policy optimum |

---

## 13. When to use A2C

### Well suited for:

* **Continuous action spaces** (robot control, autopilots).
* **Long-episode tasks** (no need to wait for the episode to end before updating).
* **Stochastic environments** (where a deterministic policy is suboptimal).
* **Tasks requiring exploration** (via entropy regularization).

### Not well suited for:

* Tasks with very high reward variance (a replay-buffer-based DQN is a better fit).
* Tasks requiring maximal sample efficiency (off-policy methods are better).

---

## 14. Practical tips for training A2C

### Hyperparameters

| Parameter | Typical value | Purpose |
|----------|-------------------|------------|
| Learning rate (Actor) | $3 \times 10^{-4}$ | The policy's update speed |
| Learning rate (Critic) | $1 \times 10^{-3}$ | The value function's update speed |
| Discount factor $\gamma$ | $0.99$ | Discounting future rewards |
| GAE $\lambda$ | $0.95$ | The bias-variance trade-off |
| Entropy coefficient $c_2$ | $0.01$ | The strength of exploration |
| Value loss coefficient $c_1$ | $0.5$ | The critic loss's weight in the total loss |

### Normalization

1. **State normalization (running statistics):**
   ```python
   # Initialize the running statistics
   running_mean = np.zeros(state_dim)
   running_std = np.ones(state_dim)
   alpha = 0.01  # the statistics' update rate
   
   # Updating and applying
   running_mean = alpha * state + (1 - alpha) * running_mean
   running_std = alpha * np.abs(state - running_mean) + (1 - alpha) * running_std
   state = (state - running_mean) / (running_std + 1e-8)
   
   # Alternative: use VecNormalize from stable-baselines3
   from stable_baselines3.common.vec_env import VecNormalize
   ```

2. **Advantage normalization:**
   ```python
   advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
   ```

3. **Gradient clipping:**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
   ```

### Debugging

* **Monitor the TD error** — it should decrease over time.
* **Entropy** — it shouldn't drop too quickly (the policy becoming deterministic).
* **The actor/critic loss ratio** — should stay balanced.
* **Average reward** — should trend upward.

---

## 15. Extensions and improvements to A2C

### Proximal Policy Optimization (PPO)

Limits how much the policy can change in a single update via clipping:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\big(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t\big) \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio.

**Advantage:** more stable training, better sample efficiency.

### Soft Actor-Critic (SAC)

Maximizes not only reward but also entropy (maximum entropy RL):

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_t r(s_t, a_t) + \alpha H(\pi(\cdot|s_t)) \right]
$$

**Advantage:** better exploration, works with continuous actions.

---

## 16. Key formulas (cheat sheet)

**TD error (advantage):**

$$
\boxed{ \delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) }
$$

**Actor (policy) update:**

$$
\boxed{ \theta \leftarrow \theta + \alpha_\theta \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \delta_t }
$$

**Critic (value function) update:**

$$
\boxed{ \phi \leftarrow \phi - \alpha_\phi \cdot \nabla_\phi \big(V_\phi(s_t) - y_t\big)^2 }
$$

where $y_t = r_{t+1} + \gamma V_\phi(s_{t+1})$ is the TD target.

**GAE (Generalized Advantage Estimation):**

$$
\boxed{ A_t^{\text{GAE}(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l} }
$$

**The total loss with entropy:**

$$
\boxed{ L_{\text{total}} = L_{\text{actor}} + c_1 \cdot L_{\text{critic}} - c_2 \cdot H(\pi) }
$$

---

## 17. Visualizing A2C training

```
Episode reward vs Training steps
    │
400 │                              ╱───
    │                          ╱───
300 │                      ╱───
    │                  ╱───
200 │              ╱───
    │          ╱───
100 │      ╱───
    │  ╱───
  0 │───────────────────────────────────
    0    5k   10k   15k   20k   25k   30k
                Training steps
```

**Typical behavior:**
* **0-5k steps:** Random behavior, low rewards.
* **5k-15k steps:** Rapid improvement, the policy learns the main patterns.
* **15k-30k steps:** Stabilization, fine-tuning the policy.

---

## 18. Comparison with other algorithms

```
                Sample Efficiency
                        │
    DQN ────────────────┼─────────→ (High)
                        │
    A2C ────────┼───────┤
                        │
    REINFORCE ──┼───────┤
                        │
                        ↓
                  Stability
                        │
    PPO ────────────────┼─────────→ (High)
                        │
    A2C ────────┼───────┤
                        │
    DQN ────────┼───────┤
                        │
    REINFORCE ──┼───────┤
```

---

## Summary

| Concept | Description |
|---------|----------|
| **Actor-Critic** | A combination of policy-based and value-based methods |
| **Actor** | The policy $\pi_\theta(a\|s)$, which chooses actions |
| **Critic** | The value function $V_\phi(s)$, which evaluates actions |
| **TD error** | $\delta_t = r + \gamma V(s') - V(s)$ |
| **Advantage** | $A_t = Q(s,a) - V(s) \approx \delta_t$ |
| **A2C** | Advantage Actor-Critic with GAE and entropy |
| **A3C** | The asynchronous version with parallel workers |
| **Applications** | Continuous actions, online learning, stochastic environments |

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Mnih et al., *Asynchronous Methods for Deep Reinforcement Learning* (2016)
* Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015)
* Schulman et al., *Proximal Policy Optimization Algorithms* (2017)
* Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning* (2018)
* Hugging Face Deep RL Course, Unit 5
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
