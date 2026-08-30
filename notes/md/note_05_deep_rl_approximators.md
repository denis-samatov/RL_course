# Theoretical Note #5

## Topic: What "Deep" Means in Deep Reinforcement Learning

> **Related to:** [note_01_introduction_to_deep_rl.md](note_01_introduction_to_deep_rl.md) · [note_02_rl_framework_and_mdp.md](note_02_rl_framework_and_mdp.md) · [note_03_exploration_vs_exploitation.md](note_03_exploration_vs_exploitation.md) · [note_04_policy_vs_value_methods.md](note_04_policy_vs_value_methods.md)

---

## 1. A reminder: classical Reinforcement Learning

Before 2013-2014, most RL algorithms used **tabular methods**.

**Example — Q-Learning.**
It stores a table $Q(s, a)$, where each cell holds the "value" of taking action $a$ in state $s$.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \Big[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\Big]
$$

This table is updated as the agent gains experience.

![A Q-table example](images/Screenshot%202025-10-24%20at%2010.06.31.png)

---

### The problem with the tabular approach

It **only works when the number of states is small**.

For example:

* In FrozenLake (4x4) — only 16 states → fine.
* In Atari Breakout (a 210x160x3 screen) →
  $210 \times 160 \times 3 = 100{,}800$ pixels!
  You can't build a table over millions of states.

---

## 2. Enter "Deep" — a neural-network approximator

This is where the **Deep RL revolution** begins.

Instead of a table $Q(s,a)$,
we use a **deep neural network** that **approximates** the function $Q(s,a)$:

$$
Q(s,a; \theta) \approx Q^*(s,a)
$$

where:

* $\theta$ — the neural network's parameters (weights);
* the input — state $s$ (e.g. a frame from the game);
* the output — a vector of $Q(s,a)$ estimates for every action.

---

### Example

Instead of a table:

| State | Action | Q |
|-----------|----------|---|
| s₁ | a₁ | 0.6 |
| s₁ | a₂ | 0.2 |

Now the neural network takes a **frame image** as input and **outputs Q values for every action**:

$$
\text{output} = [Q(s, a_1), Q(s, a_2), Q(s, a_3), \dots]
$$

---

## 3. What we gain

| Benefit | Explanation |
|--------------|------------|
| **Scalability** | The network generalizes — no need to store Q for every state. |
| **Working with images and sensor data** | The network extracts features automatically. |
| **An "internal representation" effect** | The model learns the structure of the environment (e.g. where the player, the goal, and enemies are). |
| **Integration with DL infrastructure** | We can use PyTorch, TensorFlow, GPUs — hundreds of times faster. |

---

## 4. Example: classical Q-Learning vs. Deep Q-Learning

| Property | Q-Learning | Deep Q-Learning (DQN) |
|----------------|------------|----------------------|
| How Q is represented | A table | A neural network |
| Environment type | Small, discrete | Large, visual |
| Memory | Enormous | Approximated |
| Generalization | No | Yes |
| Example | FrozenLake | Atari, CarRacing |

---

## 5. How Deep Q-Learning works (the intuition)

We replace the Q-table with a neural network and train it via **backpropagation**.

### The loss function

$$
L(\theta) = \mathbb{E}\left[\Big(y - Q(s,a;\theta)\Big)^2\right]
$$

where

$$
y = r + \gamma \max_{a'} Q(s', a'; \theta^-)
$$

is the target value, computed using a **frozen network** $\theta^-$.

---

### Key improvements from DeepMind (2015)

1. **Experience Replay** — a memory buffer where the agent stores experience $(s, a, r, s')$ and trains on random mini-batches.
   → reduces data correlation.

2. **Target Network** — a copy of the main network, updated less often.
   → stabilizes training.

**Result:**
DeepMind trained an agent to **play Atari 2600 games** without knowing the rules — from pixels alone!
This became **a landmark moment in RL history (Nature, 2015)**.

---

## 6. When "Deep" helps, and when it doesn't

| Situation | Approach |
|----------|--------|
| A simple discrete environment (FrozenLake, GridWorld) | Classical RL |
| Complex visual data (Atari, Doom, robot sensors) | Deep RL |
| Limited data or compute | Classical RL |
| Large state spaces and continuous actions | Deep RL (PPO, SAC, TD3, etc.) |

---

## 7. Summary

> Deep Reinforcement Learning is the combination of **reinforcement learning** and **deep neural networks**,
> where the network serves as an **approximator** for the value function, the policy, or both at once.

---

## Key takeaways

| Component | Classical RL | Deep RL |
|-----------|----------------|---------|
| State representation | A vector/index | An image, signal, or embedding |
| The function $Q(s,a)$ | A table | A neural network |
| Approximation | No | Yes |
| Training | Tabular updates | Gradient descent |
| Applications | Simple environments | Complex visual and continuous tasks |

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Mnih et al., *Human-level control through deep reinforcement learning* (Nature, 2015)
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
