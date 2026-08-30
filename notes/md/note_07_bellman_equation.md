# Theoretical Note #7

## Topic: The Bellman Equation — the Foundation of Reinforcement Learning

> **Related to:** [note_06_value_based_methods.md](note_06_value_based_methods.md) — The V and Q functions · [note_04_policy_vs_value_methods.md](note_04_policy_vs_value_methods.md) — Value-Based methods

---

## 1. Intuition: why do we need the Bellman equation?

Earlier we introduced the **value function**:

$$
V_\pi(s) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+1+k} \mid S_t = s \right]
$$

This function tells us:

> *What reward the agent can expect if it starts in state $s$ and follows policy $\pi$.*

---

### The problem

To compute $V_\pi(s)$, we'd need to sum **every future reward**. That's expensive: the agent would have to "live out" every possible trajectory to the end.

---

## 2. Bellman's idea: a recursive definition of value

Richard Bellman (1957) proposed a brilliant simplification:

> "The value of a state equals *the immediate reward, plus the value of the next state*."

### Formally

$$
V_\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s \right]
$$

This is the **Bellman equation**: the infinite sum is replaced by a **recursive computation**, where each value builds on the next.

### Intuition using the "mouse and cheese" example

* The agent (mouse) in state $s_t$ gets a reward $R_{t+1}$ for that step.
* The next state $s_{t+1}$ has value $V(s_{t+1})$.
* So the value of the starting state is:

$$
V(s_t) = R_{t+1} + \gamma V(s_{t+1})
$$

If we know the values of neighboring states, we can iteratively update each new one.

---

## 3. Key parameters of the equation

| Parameter | What it means | Example |
| -------------- | ---------------------------------------------------- | ----------------------------------------------------------- |
| $R_{t+1}$ | the immediate reward | $+1$ for a step forward |
| $\gamma$ | the discount factor $(0 \le \gamma \le 1)$ | $0.99$ — "think about the future," $0.1$ — "live in the moment" |
| $V(S_{t+1})$ | the value of the next state | the expected reward "over there" |

**Edge cases:**

* If $\gamma = 1$ → we weight **all future rewards** equally (a long-term planner).
* If $\gamma = 0$ → we only consider **the current reward** (a greedy, "short-sighted" agent).

---

## 4. The analogy with dynamic programming (DP)

Bellman formulated the idea of **breaking a problem into subproblems**: if we know what the current step "costs" and what it "costs" to be further along, we can build the entire value function step by step.

---

## 5. Extending this to the Q-function

Sometimes we care about "the value of an action in a state":

$$
Q_\pi(s, a) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma \, V_\pi(S_{t+1}) \mid S_t=s,\ A_t=a \right].
$$

The Bellman expectation equation for $Q_\pi$, accounting for the environment's dynamics:

$$
Q_\pi(s,a) = r(s,a) + \gamma \sum_{s'} P(s'\mid s,a) \sum_{a'} \pi(a'\mid s') \, Q_\pi(s',a').
$$

The **optimality** equation for the optimal function $Q^*$:

$$
Q^*(s,a) = r(s,a) + \gamma \sum_{s'} P(s'\mid s,a) \, \max_{a'} Q^*(s',a').
$$

This is the foundation of the **Q-Learning** algorithm.

---

## 6. A numerical example

A simple chain of states:

| State | Reward | Next state | $V(S_{t+1})$ | $\gamma$ | $V(S_t)$ |
| --------- | ------- | ------------------- | -------------- | ---------- | -------------------------- |
| $S_1$ | $+1$ | $S_2$ | $4$ | $0.9$ | $1 + 0.9 \times 4 = 4.6$ |
| $S_2$ | $+2$ | $S_3$ | $5$ | $0.9$ | $2 + 0.9 \times 5 = 6.5$ |
| $S_3$ | $+5$ | (goal) | $0$ | $0.9$ | $5 + 0.9 \times 0 = 5$ |

---

## 7. The key takeaways

1. **Bellman is a principle.** Every RL model rests on the idea of "current reward + the value of the future."
2. **Q-Learning is a special case of Bellman.** We don't know the real value function, but we **approximate** it by interacting with the environment.
3. **There's no RL without Bellman.** It's what connects the theory (expectations, probabilities) to the practice (iterative updates in code).

---

## 8. Key formulas

### For the $V$-function (the expectation equation)

$$
\boxed{ V_\pi(s) = \sum_a \pi(a\mid s) \Big[ r(s,a) + \gamma \sum_{s'} P(s'\mid s,a) \, V_\pi(s') \Big] }
$$

### For the $Q$-function (optimality)

$$
\boxed{ Q^*(s, a) = r(s,a) + \gamma \sum_{s'} P(s'\mid s,a) \, \max_{a'} Q^*(s', a') }
$$

---

## 9. What's next

DP built on the Bellman equations leads to **Policy Iteration** and **Value Iteration** (evaluating a policy and improving it, or iterating directly on optimality). See [note_13_dynamic_programming.md](note_13_dynamic_programming.md) for the full treatment, with a hands-on implementation in [`code/13_dynamic_programming/`](../../code/13_dynamic_programming/).

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Bellman, R., *Dynamic Programming* (1957)
* Hugging Face Deep RL Course, Unit 2
