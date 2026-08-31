# Theoretical Note #11

## Topic: Policy Gradient Methods and REINFORCE

> **Related to:** [note_04_policy_vs_value_methods.md](note_04_policy_vs_value_methods.md) — Policy-Based methods · [note_09_q_learning.md](note_09_q_learning.md) — Q-Learning · [note_10_deep_q_network.md](note_10_deep_q_network.md) — Deep Q-Network

---

## 1. From value-based to policy-based: why do we need Policy Gradients?

So far we've studied **value-based methods** (Q-Learning, DQN), which estimate the value of actions and derive the policy indirectly:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

But this approach has limitations:

* **Discrete actions** — in continuous spaces (robot control, autopilot), searching for the $\arg\max$ becomes computationally infeasible.
* **Determinism** — a greedy policy always picks the same action, which can be suboptimal in stochastic environments.
* **Instability** — small changes in Q-values can drastically change the policy.

**The solution:** learn the policy $\pi_\theta(a|s)$ directly, where $\theta$ are the parameters (a neural network's weights).

---

## 2. Intuition: what is a Policy Gradient?

**Policy Gradient** methods are a family of methods that train a policy by **directly maximizing expected reward** via gradient ascent.

### The idea

Instead of asking "How good is action $a$ in state $s$?" (Q-Learning),
we ask: "How should the policy's parameters $\theta$ change to get more reward?"

---

## 3. Formalizing the objective: J(θ)

Define the **objective function** as the expected total reward when following policy $\pi_\theta$:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [G_0] = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_{t+1} \right]
$$

where:

* $\tau = (s_0, a_0, r_1, s_1, a_1, \dots)$ — a trajectory,
* $G_0$ — the return from the start of the episode.

**The training goal:**

$$
\theta^* = \arg\max_\theta J(\theta)
$$

---

## 4. The gradient of the objective: the Policy Gradient Theorem

To maximize $J(\theta)$, we need its gradient $\nabla_\theta J(\theta)$.

### The Policy Gradient Theorem (PG Theorem)

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]
$$

**Intuitive interpretation:**

> "Increase the probability of actions that led to high reward, and decrease the probability of actions that led to low reward."

---

## 5. Breaking down the gradient's components

| Component | What it means | Intuition |
|-----------|--------------|----------|
| $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ | The gradient of the log-probability of the action | The direction to change the parameters to increase $P(a_t)$ |
| $G_t$ | The total reward from time $t$ onward | The action's "weight" (how rewarding it was) |
| $\mathbb{E}_{\tau \sim \pi_\theta}$ | Averaging over trajectories | We must gather experience by interacting with the environment |

---

## 6. The REINFORCE algorithm (Monte Carlo Policy Gradient)

**REINFORCE** is the simplest Policy Gradient algorithm, using Monte Carlo to estimate the return.

### Pseudocode

1. Initialize the policy's parameters $\theta$ randomly.
2. For every episode $k = 1, 2, \dots$:
   1. Generate a trajectory $\tau = (s_0, a_0, r_1, s_1, a_1, \dots, s_T)$, following $\pi_\theta$.
   2. For every time step $t = 0, 1, \dots, T-1$:
      * Compute the return:
        $$
        G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_{k+1}
        $$
      * Update the parameters:
        $$
        \theta \leftarrow \theta + \alpha \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t
        $$

---

## 7. An intuitive example: CartPole

Consider the `CartPole-v1` environment, where the agent balances a pole:

* **State $s$:** cart position, pole angle, velocities.
* **Actions:** push left (0) or right (1).
* **Reward:** +1 for every step the pole hasn't fallen.

### How REINFORCE works

1. The agent plays through an episode (say, 50 steps before falling).
2. The return $G_0 = 50$ (the pole balanced for 50 steps).
3. For every action in the episode:
   * If the action was "right" and $G_t$ is high → increase $P(\text{right}|s)$.
   * If the action was "left" and $G_t$ is low → decrease $P(\text{left}|s)$.

After many episodes, the policy starts to favor actions that lead to longer balancing.

---

## 8. The high-variance problem

**REINFORCE's main challenge:** high gradient variance.

The return $G_t$ can vary widely between episodes even under the same policy:

* Episode 1: $G_0 = 200$ (lucky)
* Episode 2: $G_0 = 10$ (unlucky)

This leads to **unstable training** — gradients "jump around," and convergence is slow.

---

## 9. The solution: a Baseline

To reduce variance, we subtract a **baseline** $b(s_t)$ from the return — an average value that doesn't depend on the action:

$$
\nabla_\theta J(\theta) \approx \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \big(G_t - b(s_t)\big) \right]
$$

### Popular baseline choices

| Baseline type | Formula | Intuition |
|--------------|---------|----------|
| **A constant** | $b = \bar{G}$ (the average over episodes) | The simplest option |
| **State-value function** | $b(s_t) = V_\pi(s_t)$ | An estimate of the expected return from $s_t$ |
| **Running average** | $b = \alpha \cdot b + (1-\alpha) \cdot G_t$ | A moving average |

**Important:** subtracting a baseline **does not introduce bias** into the gradient, but it does reduce variance.

---

## 10. The Advantage Function

When $V(s_t)$ is used as the baseline, the difference $G_t - V(s_t)$ is called the **advantage function**:

$$
A_t = G_t - V(s_t)
$$

Intuition:

> "How much **better than average** was action $a_t$ for state $s_t$?"

* $A_t > 0$ → the action was better than expected → increase its probability.
* $A_t < 0$ → the action was worse than expected → decrease its probability.

---

## 11. REINFORCE with a baseline (using a V-function)

### The algorithm

1. Initialize:
   * The policy $\pi_\theta$ (a neural network).
   * The value function $V_\phi$ (a separate network, or a shared encoder).

2. For every episode:
   1. Collect a trajectory $\tau$, following $\pi_\theta$.
   2. For every $t$:
      * Compute $G_t$ (the return).
      * Compute the advantage:
        $$
        A_t = G_t - V_\phi(s_t)
        $$
      * Update the policy:
        $$
        \theta \leftarrow \theta + \alpha_\theta \cdot \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t
        $$
      * Update the V-function (minimizing MSE):
        $$
        \phi \leftarrow \phi - \alpha_\phi \cdot \nabla_\phi \big(V_\phi(s_t) - G_t\big)^2
        $$

---

## 12. Neural network architecture for the policy

### Discrete actions (e.g. CartPole)

The network's output is a vector of logits (or probabilities) for each action:

$$
\pi_\theta(a|s) = \text{Softmax}\big(\text{NN}_\theta(s)\big)
$$

Example:

```python
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)
```

---

### Continuous actions (e.g. LunarLander continuous)

The network's output is the parameters of a distribution (e.g. mean $\mu$ and variance $\sigma$ for a normal distribution):

$$
a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))
$$

```python
class ContinuousPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
```

---

## 13. Comparing REINFORCE and Q-Learning

| Property | Q-Learning (DQN) | REINFORCE |
|----------------|------------------|-----------|
| Method type | Value-based | Policy-based |
| What's learned | The Q-function $Q(s,a)$ | The policy $\pi_\theta(a\|s)$ |
| Action space | Discrete | Discrete and continuous |
| Needs a replay buffer | Yes | No (on-policy) |
| Variance | Low | High (needs a baseline) |
| Convergence | Faster, but unstable | Slower, but more stable |
| Stochastic policies | No | Yes |

---

## 14. A hands-on example: training on CartPole

```python
import torch
import torch.optim as optim
import gymnasium as gym

# Setup
env = gym.make('CartPole-v1')
policy = PolicyNetwork(state_dim=4, action_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=0.01)
gamma = 0.99

for episode in range(1000):
    states, actions, rewards = [], [], []
    state, _ = env.reset()
    done = False

    # Collect a trajectory
    while not done:
        # Add the batch dimension: [state_dim] -> [1, state_dim]
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = policy(state_tensor).squeeze(0)  # [1, action_dim] -> [action_dim]

        # Use torch.distributions for numerical stability
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample().item()

        next_state, reward, terminated, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        done = terminated or truncated

    # Compute the returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize

    # Update the policy
    policy_loss = []
    for state, action, G_t in zip(states, actions, returns):
        # Handle dimensions correctly
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = policy(state_tensor).squeeze(0)

        # Use Categorical.log_prob for numerical stability
        dist = torch.distributions.Categorical(probs=probs)
        log_prob = dist.log_prob(torch.tensor(action))

        policy_loss.append(-log_prob * G_t)

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
```

---

## 15. Entropy regularization

To keep the policy from becoming **too deterministic** (which hurts exploration), an **entropy bonus** is added to the objective:

$$
J(\theta) = \mathbb{E}_\tau \left[ \sum_t \log \pi_\theta(a_t|s_t) \cdot G_t + \beta \cdot H(\pi_\theta(\cdot|s_t)) \right]
$$

where the policy's entropy is:

$$
H(\pi) = - \sum_a \pi(a|s) \log \pi(a|s)
$$

**Intuition:** high entropy = a more uniform probability distribution = more exploration.

---

## 16. Key formulas (cheat sheet)

**Policy Gradient Theorem:**

$$
\boxed{ \nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right] }
$$

**REINFORCE with a baseline:**

$$
\boxed{ \nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \big(G_t - b(s_t)\big) }
$$

**Advantage function:**

$$
\boxed{ A_t = G_t - V(s_t) }
$$

**Parameter update (gradient ascent):**

$$
\boxed{ \theta \leftarrow \theta + \alpha \cdot \nabla_\theta J(\theta) }
$$

---

## 17. Advantages and disadvantages of Policy Gradients

### Advantages

* **Continuous actions** — works naturally with continuous spaces.
* **Stochastic policies** — can learn probabilistic strategies.
* **Stability** — the policy changes smoothly (no sharp jumps like in value-based methods).
* **Convergence guarantee** — under the right conditions, converges to a local optimum.

### Disadvantages

* **High variance** — needs many trajectories for a reliable gradient estimate.
* **Sample inefficiency** — an on-policy method, needs fresh data after every update.
* **Slow convergence** — compared to DQN, can need more training steps.
* **Hyperparameter sensitivity** — $\alpha$, $\gamma$, and the network architecture are all critical.

---

## Summary

| Concept | Formula/description |
|---------|------------------|
| **Objective function** | $J(\theta) = \mathbb{E}_\tau[G_0]$ |
| **Policy Gradient** | $\nabla_\theta J(\theta) = \mathbb{E}[\nabla \log \pi \cdot G_t]$ |
| **REINFORCE** | Monte Carlo PG using the full return |
| **Baseline** | $G_t - b(s_t)$, to reduce variance |
| **Advantage** | $A_t = G_t - V(s_t)$ |
| **Applications** | Discrete and continuous actions |

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015)
* Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (1992)
* Hugging Face Deep RL Course, Unit 4
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
