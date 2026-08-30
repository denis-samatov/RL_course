# Theoretical Note #2

## Topic: The Reinforcement Learning Framework

> **Related to:** [note_01_introduction_to_deep_rl.md](note_01_introduction_to_deep_rl.md) — Introduction to Deep Reinforcement Learning

---

## 1. The general idea of the agent-environment loop

Reinforcement learning (RL) is an **iterative interaction process** between two components:

* **Agent** — makes decisions, chooses actions.
* **Environment** — reacts to those actions, changing its state and returning a reward.

---

### The Agent-Environment Loop

At every time step $t$:

1. The agent observes the environment's current **state**
   $$
   s_t \in \mathcal{S}
   $$
   where $\mathcal{S}$ is the set of all possible states.

2. Based on its policy, the agent chooses an **action**
   $$
   a_t \in \mathcal{A}(s_t)
   $$
   where $\mathcal{A}(s_t)$ is the set of actions available in state $s_t$.

3. The environment executes this action and returns:

   * A **new state** $s_{t+1}$
   * A **reward**
     $$
     r_{t+1} = r(s_t, a_t)
     $$

4. The agent updates its knowledge/policy and moves to the next step.

---

The interaction process can thus be written as a **sequence:**
$$
(s_0, a_0, r_1, s_1, a_1, r_2, \dots)
$$

---

## 2. Formalizing it as an MDP

To describe RL agents mathematically, we use the **Markov Decision Process (MDP)** model:

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma)
$$

where:

| Symbol | Component | Description |
|:------:|-----------|----------|
| $\mathcal{S}$ | State space | The set of environment states |
| $\mathcal{A}$ | Action space | The set of agent actions |
| $P(s' \mid s,a)$ | Transition probability | The probability of transitioning from $s$ to $s'$ under action $a$ |
| $r(s,a)$ | Reward function | The average reward for taking $a$ in state $s$ |
| $\gamma \in [0,1)$ | Discount factor | The discount rate for future rewards |

---

## 3. The agent's goal

The agent seeks to **maximize the expected return** (the expected cumulative reward):

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

Intuition:

* $r_{t+k+1}$ — the reward $k$ steps in the future;
* $\gamma^k$ — shrinks that reward's contribution as time passes;
* the closer a reward is, the more it "weighs" in the decision.

---

## 4. The Reward Hypothesis

> **Reinforcement learning's central principle:**
>
> Any goal can be represented as **maximizing expected cumulative reward**.

### Formalization

If an agent interacts with the environment over time, at time $t$ it receives a sequence of rewards:

$$
r_{t+1}, r_{t+2}, r_{t+3}, \dots
$$

Its **cumulative reward** (return, $G_t$) is then defined as:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

where $\gamma \in [0, 1)$ is the discount factor, which reduces the importance of future rewards.

### The key observation

This hypothesis claims that **every form of goal-directed behavior** — whether it's

* learning to walk, play, or trade,
* controlling a rocket or a car,
* optimizing an investment strategy,

— can be reduced to **maximizing expected cumulative reward**.

> Even when an agent's goal looks complex on the surface (e.g. "land on the Moon"), in RL terms it's formulated as "get the highest total reward from the actions that lead to a successful landing."

### Why this matters

1. **A single mathematical foundation.** Any task can be expressed in terms of rewards and their expectation, which makes RL a universal theory of learning through experience.

2. **Label-free learning.** Unlike supervised learning, the agent doesn't know the "right answer" — it **forms its own strategy** through trial, error, and rewards.

3. **A flexible objective.** Changing how the reward is defined can completely change the agent's behavior:
   * a reward for speed → fast play;
   * a reward for accuracy → cautious behavior.

4. **Emergent behavior.** Complex skills often arise *automatically*, as a side effect of maximizing the overall reward (e.g. agents that learn to jump to avoid enemies, even though that was never explicitly specified).

### The mathematical formulation of the goal

The agent seeks to find the optimal policy $\pi^*$ that maximizes the expected sum of discounted rewards:

$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi}[G_t]
$$

where $\pi(a|s)$ is the policy (the probability of choosing action $a$ in state $s$).

The agent learns a sequence of actions that **maximizes the expected value of $G_t$**.

---

## 5. The Markov Property

**Definition:**
A process satisfies the **Markov property** if:
$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0, a_0) = P(s_{t+1} \mid s_t, a_t)
$$

In other words:

> The future depends **only on the current state and action**, not on the entire history.

This property is what makes an MDP (Markov Decision Process) **tractable** — we can reason about an agent's learning based only on the current observation.

---

## 6. A simple illustration (a game example)

Suppose an agent is learning to play a simple platformer:

1. **State $s_t$** — the current frame of the level.
2. **Action $a_t$** — move right, jump, etc.
3. **Reward $r_t$** — +1 for collecting a coin, -10 for dying.
4. **New state $s_{t+1}$** — the next frame.

The agent goes through this loop thousands of times, until it develops a strategy that **maximizes the total reward**.

---

## 7. The agent's policy

A policy is how the agent chooses actions:

* **Stochastic policy:**
  $$
  \pi(a|s) = P(A_t = a \mid S_t = s)
  $$
* **Deterministic policy:**
  $$
  a_t = \pi(s_t)
  $$

The learning goal is to find the **optimal policy** $\pi^*$, such that:
$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\pi}[G_t]
$$

---

## 8. The difference between state and observation

Sometimes the agent doesn't see the whole state of the world (a partially observable environment):

| Environment type | What the agent sees | Example |
|-----------|-----------------|--------|
| **Fully observable** | The whole state $s_t$ | Chess (the whole board is visible) |
| **Partially observable** | Only an observation $o_t$ | Super Mario (only a fragment of the level is visible) |

---

## 9. Action spaces

* **Discrete space:** a finite number of actions (left, right, jump).
* **Continuous space:** an action can take any value (e.g. the steering angle in an autopilot).

---

## 10. Reward discounting

To avoid over-valuing distant rewards, we use **discounting**:

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots
$$

If $\gamma = 0.9$, then after 3 steps a reward's weight is $0.9^3 = 0.729$.

→ In this way, the agent **balances** short-term and long-term gains.

---

## Summary

| Component | Symbol | Description |
|-----------|-------------|----------|
| State | $s_t$ | a description of the environment |
| Action | $a_t$ | the agent's choice |
| Reward | $r_t$ | the feedback signal |
| Transition | $P(s' \mid s,a)$ | the environment's dynamics |
| Policy | $\pi(a \mid s)$ | the agent's strategy |
| Goal | $\max_\pi \mathbb{E}[G_t]$ | maximizing expected reward |

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
* RL Theory Book (Forts & Mills, 2022)
