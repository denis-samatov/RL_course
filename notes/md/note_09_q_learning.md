# Theoretical Note #9

## Topic: **Q-Learning — the Foundational Action-Learning Algorithm**

---

## 1. What is Q-Learning?

**Q-Learning** is an *off-policy, value-based* reinforcement learning algorithm
that learns the **action-value function** (Q-function) via **TD updates** (Temporal Difference).

---

### Terminology:

* **Value-based** — the agent doesn't learn the policy directly; it *estimates* the value of actions.
* **TD Learning** — updates happen after every step, not after the whole episode.
* **Off-policy** — the policy being learned and the policy used to choose actions differ (explore vs. exploit).

---

### Formal definition:

The **optimal Q-function** describes the value of taking action $a$ in state $s$, from the perspective of the optimal strategy:

$$
Q^*(s, a) = \mathbb{E} \big[ R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') \mid S_t = s, A_t = a \big].
$$

Intuition:

> "If I take action $a$ in state $s$ right now,
> how good is that for me in the long run?"

---

## 2. The difference between Reward and Value

| Concept | What it means |
| ------------------- | ------------------------------------------------------------------------ |
| **Reward** | The immediate reward for the current action, $R_{t+1}$ |
| **Value / Q-value** | The expected *cumulative* reward over the entire future, under the optimal policy |

That is:

> Reward is an instant "feedback signal,"
> Value is "how worthwhile it is to be in this spot and take this action."

---

## 3. The Q-table: the agent's memory

The Q-function is represented as a **Q-table**, where:

* rows — the environment's states $S$
* columns — the available actions $A$
* cell $Q(s, a)$ — a value showing the "quality" of action $a$ in state $s$.

Example:

```
States →       Left    Right    Up      Down
----------------------------------------------
S0             0.1      0.0      0.5     0.0
S1             0.2      0.8      0.0     0.1
S2             0.0      0.3      0.0     0.9
```

> In this example, the agent believes that in state S1, the best action is "right."

---

## 4. How Q-Learning works — the general algorithm

### Pseudocode:

1. Initialize the Q-table to zeros.
2. Repeat (for every episode):

   * Initialize the starting state $s$
   * While the episode hasn't ended:

     1. Choose action $a$ via an ε-greedy policy
     2. Take action $a$, get reward $r$ and the new state $s'$
     3. Update Q via the Bellman formula:
        $$
        Q(s,a) \leftarrow Q(s,a) + \alpha \big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \big]
        $$
     4. Move to the new state $s'$
3. Decay ε over time (less randomness → more "exploitation").

---

## 5. Walking through the algorithm's steps

### Step 1: Initialization

$$
Q(s,a) = 0 \ \text{for all } s, a
$$

The agent starts with no knowledge — an "empty table."

---

### Step 2: Choosing an action — the ε-greedy strategy

$$
a_t =
\begin{cases}
\text{a random action}, & \text{with probability } \varepsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1 - \varepsilon
\end{cases}
$$

Example:

* Early in training, ε = 1.0 → almost always random actions (exploring the environment).
* Over time, ε → 0.1 or 0.01 → the agent uses what it has learned (exploitation).

---

### Step 3: Gaining experience

The agent takes the action, and receives:

* a new state $s'$
* a reward $r$

It remembers the experience tuple:

$$
(s_t, a_t, r_{t+1}, s_{t+1})
$$

---

### Step 4: Updating the Q-table

Now we apply the **Q-Learning update** (an approximation of the optimal Bellman equation):

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \big]
$$

Where:

* $\alpha$ — the learning rate
* $\gamma$ — the discount factor
* $\max_{a'} Q(s',a')$ — "the best possible future" from the next state

---

### An update example:

Suppose:

$$
Q(s,a) = 0, \quad r = 1, \quad \gamma = 1, \quad \max_{a'} Q(s',a') = 5, \quad \alpha = 0.1
$$

Then:

$$
Q(s,a) = 0 + 0.1 \times [1 + 1 \times 5 - 0] = 0.6
$$

The action's value (quality) in this state has now increased — the agent has "learned" that this action is good.

---

## 6. Off-policy: what does that mean?

**Off-policy** means:

> The agent *learns* using a different policy than the one it *acts* with.

* **Acting:** chosen via **ε-greedy** (a mixed policy — some random actions).
* **Updating:** uses the **greedy** (maximizing) value $\max_{a'} Q(s', a')$.

That is:

* we *explore* the world with an ε-greedy policy,
* but we *learn* as if we always chose the best action.

---

## 7. The difference from On-policy methods (e.g. SARSA)

| Algorithm | Update | Learning policy |
| --------------------------- | -------------------------------------------------- | ------------------------- |
| **Q-Learning (off-policy)** | $r + \gamma \max_{a'} Q(s',a')$ | greedy |
| **SARSA (on-policy)** | $r + \gamma Q(s',a')$ where $a'$ is chosen ε-greedily | the same one used to act |

An intuitive example:

* **Q-Learning** learns "optimistically," as if it always picks the best action.
* **SARSA** learns "realistically," based on how the agent actually acts (accounting for ε-greediness).

---

## 8. Overall visual schematic

```
     ┌──────────────────────────────┐
     │   Environment                │
     │  ┌────────────────────────┐  │
     │  │  State s, reward r      │  │
     │  └────────────────────────┘  │
     └──────────────┬───────────────┘
                    │
              action a_t
                    │
             ┌─────────────┐
             │  Agent      │
             │  (Q-table)  │
             └─────────────┘
                 ↑      ↓
             update   choose action
```

---

## 9. TL;DR — the Q-Learning formulas

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]
$$

Once converged:

$$
\pi^*(s) = \arg\max_a Q^*(s,a)
$$

That is:

* The agent trains its Q-table as it interacts with the environment.
* Once trained, the table itself "contains" the optimal strategy.

---

## 10. A hands-on example: a mini maze

### Goal:

Understand how the agent **updates the Q-table step by step**,
and how the optimal policy $\pi^*(s)$ **emerges through experience**.

---

### The setup

A simple grid: a mouse must reach a big piece of cheese while avoiding poison.

**Environment parameters:**

* Size: 3x2 cells
* The agent always starts from the same position
* An episode ends if:
  * the mouse eats the poison (reward = -10)
  * the mouse reaches the big cheese (reward = +10)
  * more than 5 steps have passed (reward = 0)

**The reward function:**

| Transition | Reward |
| ----------------------------------- | ------- |
| Moving to an empty cell | +0 |
| Moving to a cell with a small piece of cheese | +1 |
| Moving to the cell with the big cheese | +10 |
| Moving to the poison cell | -10 |

---

### Initial setup

$$
\alpha = 0.1, \quad \gamma = 0.99, \quad \varepsilon = 1.0
$$

The starting Q-table (all values zero):

| State | Up | Down | Left | Right |
| ----- | -- | ---- | ---- | ----- |
| S0 | 0 | 0 | 0 | 0 |
| S1 | 0 | 0 | 0 | 0 |
| S2 | 0 | 0 | 0 | 0 |

The agent knows nothing — it acts randomly.

---

### Timestep 1

**Step 1. Choosing an action (ε-greedy strategy)**

$$
\varepsilon = 1.0
$$

→ 100% random action. The mouse chooses **right**.

**Step 2. Taking the action**

* New state $S_1$
* Reward: $R_{t+1} = +1$ (found a small piece of cheese)

**Step 3. Updating the Q-table**

The Q-Learning formula (the Bellman equation for Q):

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[ R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t) \big]
$$

Since every value is still zero:

$$
Q(S_0, Right) = 0 + 0.1 \big[ 1 + 0.99 \cdot 0 - 0 \big] = 0.1
$$

**Updated table:**

| State | Up | Down | Left | Right |
| ----- | -- | ---- | ---- | ------- |
| S0 | 0 | 0 | 0 | **0.1** |
| S1 | 0 | 0 | 0 | 0 |
| S2 | 0 | 0 | 0 | 0 |

✅ The agent has "learned": going right at the start is slightly useful.

---

### Timestep 2

**Step 1. Choosing an action**

$$
\varepsilon = 0.99
$$

→ nearly random. The mouse chooses **down** (unluckily!).

**Step 2. The result**

* New state: $S_2$ — the poison cell
* Reward: $R_{t+1} = -10$
* The episode ends (the mouse died)

**Step 3. Updating the Q-table**

$$
Q(S_1, Down) = 0 + 0.1 [ -10 + 0.99 \times 0 - 0 ] = -1
$$

**The table after the second step:**

| State | Up | Down | Left | Right |
| ----- | -- | -------- | ---- | ------- |
| S0 | 0 | 0 | 0 | **0.1** |
| S1 | 0 | **-1.0** | 0 | 0 |
| S2 | 0 | 0 | 0 | 0 |

The agent now knows:

* going **down** from state S1 is bad,
* going **right** from S0 is good.

---

### What just happened, conceptually

1. **The experience tuple**:
   $$
   (S_t, A_t, R_{t+1}, S_{t+1})
   $$
   captures one fragment of the agent's "life."

2. **Q-Learning** *updates its knowledge* at every step (a TD update).
   This lets the agent learn **while playing**, with no need to wait for the episode to end.

3. **The ε-greedy strategy** provides the balance:
   * Early on → almost all exploration (ε ≈ 1)
   * Over time → almost all exploitation (ε → 0)

---

### Observation

After just two steps, the agent already:

* knows that getting poisoned is bad,
* knows that the "right" action from the starting position gives a reward.

With further training, the agent will:

* increasingly choose "right" → "down" → "right" (the optimal route),
* gradually converge to the **optimal Q-function** $Q^*(s,a)$.

---

### Repeating episodes

After tens of iterations, the Q-table starts to stabilize:

$$
Q^*(s,a) \approx \text{the expected total reward under the optimal policy}
$$

Once the values stop changing much, **Q-Learning has converged**.

---

### How the agent becomes "smart"

| Stage | Behavior | Q-table |
| ------------------- | ---------------------- | ----------------------- |
| Start | random steps | all zeros |
| After 10 episodes | knows where the poison is | negative Q values |
| After 100 episodes | knows where the cheese is | positive Q values |
| After 1000 episodes | takes the optimal path | Q-table ≈ optimal |

---

### The final formulas

**TD Target:**

$$
r + \gamma \max_{a'} Q(s',a')
$$

**TD Error:**

$$
\delta = \text{Target} - Q(s,a)
$$

**An update = "correcting the prediction error."**

---

### Conclusion

After many episodes, the agent:

* learns an approximation of the optimal function $Q^*(s,a)$,
* obtains the optimal policy:
  $$
  \pi^*(s) = \arg\max_a Q^*(s,a)
  $$
* and acts **efficiently**, almost like a human "remembering" where it's worthwhile to go.

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Hugging Face Deep RL Course, Unit 2
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
