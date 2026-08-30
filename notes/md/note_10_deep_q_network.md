# Theoretical Note #10

## Deep Q-Network (DQN): session notes

## Session goals

* Understand why tabular Q-Learning doesn't scale, and what DQN replaces it with.
* Work through the math of **TD-learning** and DQN's objective function.
* Master the pipeline: **frame preprocessing → CNN → replay buffer → target network → ε-greedy**.
* Learn the stability improvements: **Double DQN, Dueling, PER, N-step**.
* Discuss debugging and quality metrics, and walk through the training pseudocode.

---

## Session plan

1. Intro and problem statement (10 min)
2. From Q-Learning to Deep Q-Learning (intuition → math) (20 min)
3. DQN architecture and data (preprocessing, frame stacking, CNN) (15 min)
4. Training: the objective, loss, replay buffer, target network (20 min)
5. The ε-greedy policy and the ε schedule (5 min)
6. Stability improvements: Double/Dueling/PER/N-step (20 min)
7. Hands-on/discussion: hyperparameters, debugging, metrics (15 min)
8. Q&A and homework (5 min)

---

## 1) The dimensionality problem: why a table doesn't work

**State as an image** (example: Atari).

$$
\text{Frame: } 210 \times 160 \times 3 \Rightarrow 100{,}800 \text{ pixels}
$$

Each pixel takes 256 values. The number of possible states:

$$
256^{210\times160\times3} = 256^{100800}
$$

**Conclusion:** storing action values in a table becomes infeasible.

---

## 2) The approximation idea: from a table to a network

Replace the table with a parametric function (a neural network) that predicts Q-values.

$$
Q_\theta(s,a) \approx Q^*(s,a)
$$

The classical Q-Learning update:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \big[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\big]
$$

There's no table in DQN, so instead we train the network's parameters to minimize the TD error.

---

## 3) DQN's formalism: the objective and the loss

**The TD target** for a transition $(s,a,r,s',\text{done})$:

$$
y = \begin{cases}
r, & \text{if } \text{done} = 1\\
r + \gamma \max_{a'} Q_{\theta^-}(s', a'), & \text{otherwise}
\end{cases}
$$

**Loss (MSE over a mini-batch):**

$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s',\text{done})\sim \mathcal{D}}\big[\big(y - Q_\theta(s,a)\big)^2\big]
$$

A gradient step on the parameters $\theta$.

---

## 4) Architecture and data

### Preprocessing and frame stacking

* Convert frames to grayscale and resize to 84x84.
* Stack 4 consecutive frames, so the network can "see" motion.

### An example CNN (the classic Nature DQN)

$$
\text{Conv1: } 32,\text{ filters } 8\times 8,\ \text{stride }4\quad\to\quad
\text{Conv2: } 64,\text{ filters } 4\times 4,\ \text{stride }2\quad\to\quad
\text{Conv3: } 64,\text{ filters } 3\times 3,\ \text{stride }1
$$

Followed by a fully connected layer of 512 neurons, and an output of $N$ values (one per action):

$$
\big[Q(s,a_1),\ Q(s,a_2),\ \dots,\ Q(s,a_N)\big]
$$

---

## 5) The Replay Buffer and Target Network

The **replay buffer** stores transitions and lets us train on shuffled mini-batches:

$$
\mathcal{D} = \{(s_t,a_t,r_t,s_{t+1},\text{done}_t)\}_{t=1}^T
$$

The **target network** copies the online network's parameters periodically:

$$
\theta^- \leftarrow \theta\quad\text{every } C \text{ steps}
$$

This stabilizes the target values $y$.

---

## 6) The action-selection policy (ε-greedy)

With probability $\varepsilon$ — a random action; otherwise — the action with the highest predicted value.

A linear ε schedule over steps:

$$
\varepsilon_t = \max\big(\varepsilon_{\min},\ \varepsilon_{\max} - k \cdot t\big)
$$

An exponential alternative:

$$
\varepsilon_t = \varepsilon_{\min} + (\varepsilon_{\max}-\varepsilon_{\min}) \cdot e^{-t/\tau}
$$

---

## 7) Stability improvements

### Double DQN

Removes overestimation by decoupling the $\arg\max$ from the evaluation:

$$
y = r + \gamma \cdot Q_{\theta^-}\Big(s',\ \arg\max_{a'} Q_\theta(s',a')\Big)
$$

### Dueling Network

Decomposes into the state value and the action advantage:

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')
$$

### Prioritized Experience Replay (PER)

Sampling probability by priority $p_i$:

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}},\qquad
w_i = \left(\frac{1}{N \cdot P(i)}\right)^{\beta}
$$

where the weights $w_i$ are normalized against the batch max: $w_i \leftarrow w_i / \max_j w_j$, to avoid destabilizing the updates.

### N-step Returns

A multi-step target strengthens the signal from delayed rewards:

$$
y^{(n)} = r_0 + \gamma r_1 + \dots + \gamma^{n-1} r_{n-1}

+\ \gamma^n \max_{a'} Q_{\theta^-}(s_{t+n}, a')
$$

---

## 8) Training-loop pseudocode

1. Initialize $\theta$, copy to $\theta^-$. An empty buffer $\mathcal{D}$.
2. For every step:

   1. With probability $\varepsilon$, choose a random action; otherwise $\arg\max_a Q_\theta(s,a)$.
   2. Get $(r, s', \text{done})$, put the transition into $\mathcal{D}$.
   3. Sample a mini-batch from $\mathcal{D}$, compute $y$, minimize $\mathcal{L}(\theta)$.
   4. Every $C$ steps: $\theta^- \leftarrow \theta$.
   5. Update $\varepsilon$ per its schedule.

---

## 9) Hyperparameters (typical for Atari)

$$
\gamma = 0.99,\quad \text{batch} = 32,\quad |\mathcal{D}| = 10^6,\quad
\alpha_{\text{Adam}} \approx 2.5\times10^{-4},\quad C\in[10^3,10^4]
$$

A linear ε schedule:

$$
\varepsilon: 1.0 \to 0.1 \text{ over } 10^6 \text{ steps}
$$

---

## 10) Metrics and validation

* Average episode reward (a rolling window):

$$
\overline{R}_t = \frac{1}{K}\sum_{i=t-K+1}^{t} R_i
$$

* The fraction of actions taken with maximal Q (exploit vs. explore).
* How often the target network updates, and the loss's stability.

---

## 11) Debugging and common pitfalls

* Mixed-up axes during preprocessing → an incorrect CNN input.
* A target network that doesn't update, or updates too rarely → oscillation.
* A buffer that's too small or too correlated → overfitting.
* No "warm-up" fill of the buffer before training → a poor start.
* Inconsistent $\gamma$, reward scales, and frame normalization.

---

## 12) Hands-on exercises (for discussion/homework)

1. Implement a basic DQN with replay and a target network; compare linear vs. exponential $\varepsilon$ schedules.
2. Add Double DQN and show the reduction in overestimation (compare the average TD target).
3. Add a Dueling head and evaluate the convergence speed via $\overline{R}_t$.
4. Implement PER and measure its effect on selecting hard states (the distribution of $P(i)$).
5. Run an ablation study: vary $C$, buffer size, and batch size.

---

## 13) Brief summary

* DQN scales Q-learning to visual states by approximating $Q_\theta$.
* Key components: CNN feature extraction, a replay buffer, a target network, an ε-greedy policy.
* The improvements (Double/Dueling/PER/N-step) increase stability and efficiency.

---

## 14) Review questions for the session

1. Why doesn't the tabular approach work on images? Estimate the dimensionality formally.
2. Write out DQN's target equation, and explain the role of $\theta^-$.
3. What's the core idea of Double DQN, and how is its target formed?
4. Why the dueling decomposition, and how does it resolve the ambiguity in $A$?
5. How does the $\varepsilon$ schedule affect the early vs. late stages of training?
6. What symptoms indicate correlated samples in the buffer, and how do you fix it?

---

## 15) A mini formula cheat sheet

**Bellman optimality:**

$$
Q^*(s,a) = \mathbb{E}\big[r + \gamma\max_{a'} Q^*(s',a') \big| s,a\big]
$$

**DQN's TD target:**

$$
y = r + \gamma\max_{a'} Q_{\theta^-}(s', a')
$$

**Loss:**

$$
\mathcal{L}(\theta) = \mathbb{E}\big[(y - Q_\theta(s,a))^2\big]
$$

**Double DQN's target:**

$$
y = r + \gamma \cdot Q_{\theta^-}\big(s',\ \arg\max_{a'} Q_\theta(s',a')\big)
$$

**Dueling composition:**

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')
$$

**PER sampling and weights:**

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}},\qquad
w_i = \left(\frac{1}{N \cdot P(i)}\right)^{\beta},\quad
w_i \leftarrow \frac{w_i}{\max_j w_j}
$$
