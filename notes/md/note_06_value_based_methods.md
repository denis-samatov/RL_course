# Theoretical Note #6

## Topic: The Two Kinds of Value-Based Methods

> **Related to:** [note_04_policy_vs_value_methods.md](note_04_policy_vs_value_methods.md) — Policy-Based vs Value-Based methods · [note_05_deep_rl_approximators.md](note_05_deep_rl_approximators.md) — Deep RL

---

## 1. A reminder: why value-based methods exist

In *value-based* approaches, the agent **doesn't learn the policy directly**, as in policy-based methods. Instead, it learns to **estimate the "value" of states or actions**, then **picks the best one** based on those estimates.

> We don't tell the agent "do this" — we teach it to understand "how good is it to be in this state, and what's worth doing next."

---

## 2. Defining the value function

A **value function** maps a state (or a state+action) to the expected **discounted return**:

$$
V_\pi(s) = \mathbb{E}_{\pi} \Big[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots \ \Big| \ S_t = s \Big].
$$

Where:

* $V_\pi(s)$ — the value of state $s$ under policy $\pi$;
* $\gamma \in [0,1)$ — the discount factor;
* $R_{t+1}, R_{t+2}, \dots$ — future rewards.

---

## 3. The relationship between a policy and a value function

In value-based methods, the policy **isn't learned directly**, but it **still exists**. It's usually a **greedy policy**:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

That is, the agent **acts based on the value $Q$**, choosing the action with the highest value. So: **learning the value function → gets us the optimal policy**.

---

## 4. The two kinds of value functions

### 1) The state-value function (V-function)

$$
V_\pi(s) = \mathbb{E}_{\pi} [G_t \mid S_t = s]
$$

Interpretation: "The expected total reward if I start in state $s$ and follow policy $\pi$."

**Example (FrozenLake):** if the agent is on cell $s_5$, then $V(s_5) = 0.72$ means it expects a reward of $0.72$ (accounting for probabilities and discounting).

### 2) The action-value function (Q-function)

$$
Q_\pi(s,a) = \mathbb{E}_{\pi} [G_t \mid S_t = s, A_t = a]
$$

Interpretation: "The expected total reward if, in state $s$, I take action $a$, and then continue following policy $\pi$."

**Example:** if the agent is standing on cell $s_5$ and can go:

* right: $Q(s_5, \text{right}) = 0.75$,
* up: $Q(s_5, \text{up}) = 0.32$,
* down: $Q(s_5, \text{down}) = -0.1$.

Then the optimal action is:

$$
a^* = \arg\max_a Q(s_5,a) = \text{right}.
$$

---

## 5. Why two kinds?

| What | What it evaluates | Used when |
| ---------- | -------------------------------- | ----------------------------- |
| $$V(s)$$ | the value of a state | the policy is fixed |
| $$Q(s,a)$$ | the value of a state-action pair | the agent needs to choose an action |

The $Q$-function is more flexible, because it directly answers the question "what should I do now?" That's why **Q-Learning is built on $Q(s,a)$**.

---

## 6. Getting a policy out of a value function

Even though the policy isn't learned explicitly, it's **still needed** for the agent to act.

### Greedy Policy

$$
\pi(s) = \arg\max_a Q(s,a)
$$

### $\varepsilon$-Greedy Policy

$$
\pi_\varepsilon(a\mid s) =
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|}, & \text{if } a = \arg\max\limits_a Q(s,a) \\
\dfrac{\varepsilon}{|\mathcal{A}|}, & \text{otherwise}
\end{cases}
$$

Here $\varepsilon$ adds exploration, so the agent doesn't get stuck at local maxima.

---

## 7. The relationship between $V$ and $Q$ through the policy

$$
V_\pi(s) = \sum_a \pi(a\mid s) \, Q_\pi(s,a)
$$

That is, the value of a state is the **weighted average value of every action** available in it (weighted by the policy's probabilities).

---

## 8. An intuitive example

Imagine an environment (say, a platformer). Suppose the estimates at some states are:

| State | Available actions | Q-values | Choice |
| ----------- | ------------------ | -------------- | ---------------------------- |
| Start | Jump, Run | $Q=0.5,\ 0.6$ | $\Rightarrow\ \text{Run}$ |
| Middle | Jump, Run | $Q=0.9,\ 0.7$ | $\Rightarrow\ \text{Jump}$ |
| Near Finish | Jump, Run | $Q=0.1,\ 1.0$ | $\Rightarrow\ \text{Run}$ |

The resulting policy:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## 9. Why computing the value function directly is hard

To compute $V(s)$ or $Q(s,a)$ directly, we'd need to sum over all possible future rewards across every trajectory — this is **exponentially hard**.

**Example:** with 10 steps and 4 actions at each: $4^{10} = 1{,}048{,}576$ possible paths.

There's no way to compute this analytically for real problems — a recursive approximation is needed.

---

## 10. Setting up the Bellman equations

To avoid summing rewards directly, we use **recursive definitions**:

$$
V_\pi(s) = \mathbb{E}_{\pi} \big[ R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s \big]
$$

and similarly for the $Q$-function:

$$
Q_\pi(s,a) = \mathbb{E}_{\pi} \big[ R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a \big]
$$

The idea: the value of the current state equals the current reward plus the discounted value of the next state.

This is **the foundation for Q-Learning, SARSA, TD(0)**, and even **DQN**.

---

## Summary table

| Concept | Formula | Intuition |
| --------------------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| **V-function** | $V_\pi(s) = \mathbb{E}_{\pi}[G_t \mid S_t = s]$ | "how good is it to be in this state" |
| **Q-function** | $Q_\pi(s,a) = \mathbb{E}_{\pi}[G_t \mid S_t = s, A_t = a]$ | "how good is it to take this action" |
| **Greedy Policy** | $\pi(s)=\arg\max_a Q(s,a)$ | picks the action with the highest $Q$ |
| **$\varepsilon$-Greedy Policy** | $\pi_\varepsilon(a\mid s)$ as above | balances exploration/exploitation |
| **The Bellman idea** | $V = R + \gamma V'$ | a recursive approximation of value |

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Hugging Face Deep RL Course, Unit 2
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
