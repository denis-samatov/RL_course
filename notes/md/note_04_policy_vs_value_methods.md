# Theoretical Note #4

## Topic: The Two Main Approaches to Solving Reinforcement Learning Problems

> **Related to:** [note_01_introduction_to_deep_rl.md](note_01_introduction_to_deep_rl.md) · [note_02_rl_framework_and_mdp.md](note_02_rl_framework_and_mdp.md) · [note_03_exploration_vs_exploitation.md](note_03_exploration_vs_exploitation.md)

---

## 1. What are we looking for?

We already know the agent's goal is to find the **optimal policy $\pi^*$**,
which maximizes expected reward:

$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi}[G_t]
$$

where

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

---

## 2. The policy $\pi$: the agent's brain

A **policy** is a function that determines the agent's behavior:

$$
\pi(a|s) = P(A_t = a \mid S_t = s)
$$

In other words, the policy says: *"In state $s$, choose action $a$ with this probability."*

The learning goal is to find a policy $\pi^*$ that maximizes total reward.

---

## 3. Two approaches to finding the optimal policy

### I. Policy-Based methods

### II. Value-Based methods

---

## 4. Policy-Based Methods

### The idea

We **learn the policy function directly**, without estimating state values.
It can be:

* **deterministic** — one action is chosen for each state;
* **stochastic** — a probability distribution over actions is defined for each state.

$$
\pi_\theta(a|s)
$$

where $\theta$ are the parameters (e.g. a neural network's weights).

![A Policy-Based policy example](images/Screenshot%202025-10-24%20at%2010.09.25.png)

---

### Example: a stochastic policy

The agent chooses an action randomly, but **according to probabilities** predicted by the model:

$$
\pi(a|s) = [0.1, 0.7, 0.2]
$$

i.e. 70% — go right, 10% — go left, 20% — jump.

---

### Advantages of Policy-Based methods

* Work well in **continuous action spaces** (e.g. robot control).
* Stable — don't suffer from the sharp value jumps Q-functions can have.
* Naturally represent stochastic strategies, which matters for **exploration**.

---

### Limitations

* Need more data (high variance).
* Gradients on the policy's parameters can be **unstable** → requires careful normalization (e.g. REINFORCE, Actor-Critic, PPO).

---

### Example algorithms

| Algorithm | Key idea |
|----------|---------------|
| **REINFORCE** | The pure policy gradient $\nabla_\theta J(\theta)$ |
| **Actor-Critic** | Combines Policy-Based and Value-Based |
| **PPO (Proximal Policy Optimization)** | A stable policy update constrained by KL divergence |

---

## 5. Value-Based Methods

### The core idea

Instead of learning the policy directly, we **estimate the "value" of states and actions**.
That is, the function says: *"how good is it to be in this state, or to take this action."*

![A Value-Based approach example](images/Screenshot%202025-10-24%20at%2010.09.43.png)

---

### Two value functions

1. **Value function (V-function):**
   $$
   V_\pi(s) = \mathbb{E}_{\pi}[G_t | S_t = s]
   $$
   — the expected reward starting from state $s$.

2. **Action-Value function (Q-function):**
   $$
   Q_\pi(s,a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]
   $$
   — the expected reward from taking action $a$ in state $s$,
   and then following policy $\pi$.

An example algorithm built on the Q-function is Q-Learning (off-policy TD control). For details and the update formula, see [note_09_q_learning.md](note_09_q_learning.md).

---

### The optimal policy via the Q-function

$$
\pi^*(s) = \arg\max_a Q^*(s,a)
$$

That is, the agent chooses the action with the **highest value estimate**.

---

### Advantages of Value-Based methods

* Simplicity — no need to explicitly define a stochastic policy.
* Effective for discrete-action tasks (Atari, FrozenLake, CartPole).
* Often converge faster.

---

### Drawbacks

* Work poorly in **continuous action spaces** — the maximum is hard to search for.
* Sometimes suffer from **training instability** (the Q-function can be over-estimated).
* Don't naturally model probabilistic behavior (purely deterministic policies).
* **A scaling problem:** tabular methods (classical Q-Learning) need to store a value for every pair $(s, a)$ — infeasible for large state spaces.

> **The solution to the scaling problem:** Deep Q-Learning (DQN) uses a neural network to approximate the Q-function. More in [note_05_deep_rl_approximators.md](note_05_deep_rl_approximators.md).

---

### Main algorithms

| Algorithm | Key idea |
|----------|---------------|
| **Q-Learning** | Updates $Q(s,a)$ via the Bellman equation |
| **SARSA** | An "on-policy" version of Q-Learning |
| **DQN (Deep Q-Network)** | Approximates $Q(s,a)$ with a neural network |

---

## 6. Comparing the approaches

| Property | Value-Based | Policy-Based |
|----------------|-------------|--------------|
| Main goal | Learn state/action values | Learn the policy directly |
| Action type | Discrete | Continuous / stochastic |
| Convergence | Fast, but unstable | Slower, but more stable |
| Examples | Q-Learning, DQN | REINFORCE, PPO |
| Applications | Games, discrete environments | Robotics, control, complex policies |

![Comparing Policy-Based and Value-Based approaches](images/Screenshot%202025-10-24%20at%2010.12.22.png)

---

## 7. A hybrid approach: Actor-Critic

Modern methods combine both ideas:

* **Actor** — the Policy-Based part (chooses the action);
* **Critic** — the Value-Based part (evaluates how good that action is).

**In short:**

> The Actor updates the strategy,
> the Critic steers the learning direction using its value estimate.

$$
\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q_\pi(s_t, a_t)
$$

---

## Summary

| Term | Definition |
|--------|-------------|
| **Policy** | The function that defines the agent's behavior |
| **Value function** | An estimate of the expected reward from a state |
| **Q-function** | An estimate of the expected reward from a (state, action) pair |
| **Value-Based** | Learn values, pick the action that maximizes Q |
| **Policy-Based** | Learn the strategy directly |
| **Actor-Critic** | Combine both approaches |

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
* RL Theory Book (Forts & Mills, 2022)
