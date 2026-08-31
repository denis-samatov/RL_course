# Theoretical Note #13

## Topic: Dynamic Programming in Reinforcement Learning

> **Related to:** [note_02_rl_framework_and_mdp.md](note_02_rl_framework_and_mdp.md) — MDP formalization · [note_07_bellman_equation.md](note_07_bellman_equation.md) — The Bellman equation

---

## 1. What is Dynamic Programming (DP) in RL?

**Dynamic Programming (DP)** is a class of methods for finding an optimal policy when the environment's model (the MDP) is **fully known**:

- The transition probabilities $P(s'|s,a)$ are known
- The rewards $r(s,a)$ or $r(s,a,s')$ are known

> "DP is computing the optimal policy through iterative application of the Bellman equations."

---

### Model-based vs Model-free

| Approach | Requires a model? | Examples |
|--------|----------------|---------|
| **Model-based** (DP) | ✅ Yes | Policy Iteration, Value Iteration |
| **Model-free** | ❌ No | Q-Learning, SARSA, MC, TD |

**Important:** In real-world tasks the environment's model is often unknown, so DP is rarely applied directly. Still, studying DP is critically important because it:

1. Forms the theoretical foundation for all RL algorithms
2. Introduces the concept of **Generalized Policy Iteration (GPI)**
3. Shows how to iteratively improve a policy

---

## 2. The Bellman equation for the optimal policy

Recall the Bellman equations (details in [note_07_bellman_equation.md](note_07_bellman_equation.md)):

**For the state-value function:**

$$
V_\pi(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s \right]
$$

$$
V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_\pi(s') \right]
$$

**For the optimal value function:**

$$
V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V^*(s') \right]
$$

**For the Q-function:**

$$
Q^*(s,a) = \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]
$$

---

## 3. Policy Evaluation

**Task:** Given a fixed policy $\pi$. Compute $V_\pi(s)$ for all states.

### The iterative algorithm

Start with arbitrary values $V_0(s)$ and iteratively apply the Bellman equation:

$$
V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_k(s') \right]
$$

**Algorithm (pseudocode):**

```python
# Initialization
V = {s: 0 for s in states}
theta = 1e-6  # convergence threshold

while True:
    delta = 0
    for s in states:
        v = V[s]
        # Bellman update
        V[s] = sum(
            pi[a|s] * sum(
                P[s'|s,a] * (r[s,a,s'] + gamma * V[s'])
                for s' in next_states(s, a)
            )
            for a in actions(s)
        )
        delta = max(delta, abs(v - V[s]))
    
    if delta < theta:
        break
```

**Convergence:** Guaranteed when $\gamma < 1$, or when every state is reachable and finite.

---

## 4. Policy Improvement

**Task:** Given $V_\pi(s)$. Construct a **better policy** $\pi'$.

### Greedy improvement

Idea: choose the action that maximizes the expected value:

$$
\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_\pi(s') \right]
$$

Equivalently, via the Q-function:

$$
\pi'(s) = \arg\max_a Q_\pi(s,a)
$$

where

$$
Q_\pi(s,a) = \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_\pi(s') \right]
$$

**The Policy Improvement Theorem:**

If $\pi'$ is obtained from $\pi$ via greedy improvement, then:

$$
V_{\pi'}(s) \geq V_\pi(s) \quad \text{for all } s
$$

Equality holds only if $\pi$ is already optimal: $\pi = \pi^*$.

---

## 5. Policy Iteration

**Idea:** Alternate between **evaluating** and **improving** the policy until convergence.

### The Policy Iteration algorithm

1. **Initialization:**
   - An arbitrary policy $\pi_0$ (e.g., uniform)
   - Arbitrary values $V_0(s) = 0$

2. **Policy Evaluation:**
   - Compute $V_{\pi_k}(s)$ for all $s$ (iteratively, until it converges)

3. **Policy Improvement:**
   - For every state:
     $$
     \pi_{k+1}(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_{\pi_k}(s') \right]
     $$

4. **Convergence check:**
   - If $\pi_{k+1} = \pi_k$, stop (we've found $\pi^*$)
   - Otherwise return to step 2

**Pseudocode (runnable Python):**

```python
import numpy as np

# 1. Initialization
n_states = env.observation_space.n
n_actions = env.action_space.n

# Uniform policy: pi[s, a] = probability of action a in state s
pi = np.ones((n_states, n_actions)) / n_actions
V = np.zeros(n_states)

theta = 1e-6  # Convergence threshold
gamma = 0.9

while True:
    # 2. Policy Evaluation
    while True:
        delta = 0
        V_new = np.zeros(n_states)
        
        for s in range(n_states):
            v = 0.0
            # Sum over every action
            for a in range(n_actions):
                # Get the environment's dynamics: [(prob, next_s, reward, done)]
                transitions = env.get_transition_prob(s, a)
                
                for prob_transition, s_prime, reward, done in transitions:
                    # Bellman update
                    v += pi[s, a] * prob_transition * (
                        reward + gamma * V[s_prime] * (1 - int(done))
                    )
            
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        if delta < theta:
            break
    
    # 3. Policy Improvement
    policy_stable = True
    
    for s in range(n_states):
        # Save the previous action
        old_action = np.argmax(pi[s])
        
        # Compute Q(s, a) for every action
        q_values = np.zeros(n_actions)
        
        for a in range(n_actions):
            transitions = env.get_transition_prob(s, a)
            q_sa = 0.0
            
            for prob, s_prime, reward, done in transitions:
                q_sa += prob * (reward + gamma * V[s_prime] * (1 - int(done)))
            
            q_values[a] = q_sa
        
        # Greedy improvement: pick the best action
        best_action = np.argmax(q_values)
        
        # Update the policy (deterministic)
        pi[s] = np.zeros(n_actions)
        pi[s, best_action] = 1.0
        
        # Check stability
        if best_action != old_action:
            policy_stable = False
    
    # 4. Convergence check
    if policy_stable:
        print("Policy Iteration converged!")
        break
```

**Complexity per iteration:** $O(|\mathcal{S}|^2 |\mathcal{A}|)$

**Convergence:** Guaranteed within a **polynomial number** of iterations.

---

## 6. Value Iteration

**Idea:** Merge evaluation and improvement into a **single step**.

Instead of fully evaluating $V_\pi$, we perform a **single** Bellman update with maximization:

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_k(s') \right]
$$

This is the **Bellman optimality equation** written as an update rule.

### The Value Iteration algorithm

1. **Initialization:** $V_0(s) = 0$ for all $s$

2. **Iterative update:**
   - For every state $s$:
     $$
     V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V_k(s') \right]
     $$
   - Continue until $\max_s |V_{k+1}(s) - V_k(s)| < \theta$

3. **Policy extraction:**
   - Once converged, $V^* \approx V_k$:
     $$
     \pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ r(s,a,s') + \gamma V^*(s') \right]
     $$

**Pseudocode:**

```python
# Initialization
V = {s: 0 for s in states}
theta = 1e-6

while True:
    delta = 0
    for s in states:
        v = V[s]
        # The optimal Bellman update
        V[s] = max(
            sum(P[s'|s,a] * (r[s,a,s'] + gamma * V[s'])
                for s' in next_states(s,a))
            for a in actions(s)
        )
        delta = max(delta, abs(v - V[s]))
    
    if delta < theta:
        break

# Extracting the optimal policy
pi = {}
for s in states:
    pi[s] = argmax_a(
        sum(P[s'|s,a] * (r + gamma * V[s'])
            for s' in next_states(s,a))
    )
```

**Complexity per iteration:** $O(|\mathcal{S}|^2 |\mathcal{A}|)$

**Convergence:** **Exponential rate** — usually faster than Policy Iteration.

---

## 7. Generalized Policy Iteration (GPI)

**GPI** is the general idea underlying **every RL algorithm**:

> "Let the policy-evaluation process and the policy-improvement process run in parallel, interacting with each other."

### The GPI schematic

```
       ┌─────────────┐
       │  Policy π   │
       └──────┬──────┘
              │
         (improvement)
              │
              ▼
       ┌─────────────┐
       │ V-function V │◄──────┐
       └──────┬──────┘       │
              │              │
          (evaluation)  (improvement)
              │              │
              ▼              │
       ┌─────────────┐       │
       │  Policy π'  │───────┘
       └─────────────┘
```

**GPI's key ideas:**

1. **Evaluation and improvement compete:**
   - Evaluation makes $V$ consistent with the current $\pi$
   - Improvement makes $\pi$ greedy with respect to the current $V$

2. **You don't have to wait for full convergence:**
   - Policy Iteration: full evaluation before improvement
   - Value Iteration: a single evaluation update before improvement
   - You can do **asynchronous updates** (see below)

3. **Guaranteed convergence to $\pi^*$ and $V^*$:**
   - Both processes stabilize only at the optimum

---

## 8. Comparing Policy Iteration and Value Iteration

| Aspect | Policy Iteration | Value Iteration |
|--------|------------------|-----------------|
| **Update** | Full evaluation of $V_\pi$, then greedy improvement | A single optimal Bellman update |
| **Iterations to convergence** | Few (typically 3-10) | More (depends on $\gamma$ and $\theta$) |
| **Cost per iteration** | High (many evaluation sub-iterations) | Low (a single pass over states) |
| **Total time** | Depends on the task | Usually faster in practice |
| **Policy** | Always defined explicitly | Extracted at the end |
| **Applicability** | Small MDPs (hundreds of states) | Medium MDPs (thousands of states) |

**Practical tip:**

- Value Iteration is generally preferable for environments with discrete states.
- Policy Iteration is better when policy evaluation converges quickly.

---

## 9. Asynchronous DP methods

**The problem with synchronous methods:** They require a full pass over **every state** at each iteration → expensive for large MDPs.

**The solution:** Update states **asynchronously**, in an arbitrary order.

### Types of asynchronous methods:

1. **In-place updates:**
   - Update $V(s)$ immediately, using already-updated neighbor values
   - Converges faster (uses the freshest information)

2. **Prioritized Sweeping:**
   - Maintain a priority queue of states
   - Priority = the magnitude of the expected change $|V_{\text{new}}(s) - V_{\text{old}}(s)|$
   - Update the states with the largest change first

3. **Real-time DP:**
   - Update only the states the agent actually visits
   - Useful when most states are unreachable

**Example: In-place Value Iteration:**

```python
# Synchronous (classic)
V_new = {}
for s in states:
    V_new[s] = max_a bellman_update(s, a, V_old)
V_old = V_new  # copy the whole thing

# Asynchronous (in-place)
for s in states:
    V[s] = max_a bellman_update(s, a, V)  # use V directly
```

**Advantages:**
- Less memory (no need for a copy of $V$)
- Converges faster (uses updated values)

---

## 10. Limitations of Dynamic Programming

| Limitation | Description | How to overcome it |
|-------------|----------|----------------|
| **Requires a model** | Needs $P(s'\|s,a)$ and $r(s,a)$ | Model-free methods (Q-Learning, SARSA) |
| **The curse of dimensionality** | $O(\|\mathcal{S}\|^2 \|\mathcal{A}\|)$ is infeasible for large MDPs | Function approximation (Deep RL) |
| **Discrete states** | Hard for continuous spaces | Discretization or function approximation |
| **A full pass over states** | Updates even unreachable states | Asynchronous methods, Real-time DP |

**Conclusion:**

> DP is rarely applied directly in modern RL, but its principles (GPI, iterative improvement) underlie every algorithm.

---

## 11. From DP to Model-Free RL

**The relationship between methods:**

| DP method | Model-free analogue | Key difference |
|----------|-------------------|------------------|
| Policy Evaluation | Monte Carlo Prediction | Requires no model, uses sample returns |
| Policy Iteration | SARSA (on-policy TD) | Updates Q(s,a) from sample transitions |
| Value Iteration | Q-Learning (off-policy TD) | Updates Q(s,a) from sample transitions with a max |

**The general idea:**

$$
\text{DP: } V(s) \leftarrow \mathbb{E}[\cdots] \quad \Rightarrow \quad \text{Model-free: } V(s) \leftarrow \text{sample}
$$

Instead of a **full mathematical expectation** (requiring a model), we use **sampled trajectories** (requiring no model).

---

## 12. A hands-on example: GridWorld

Consider a 4×4 grid with one obstacle and a goal state:

```
┌───┬───┬───┬───┐
│ S │   │   │ G │  S = Start, G = Goal
├───┼───┼───┼───┤
│   │ X │   │   │  X = Obstacle
├───┼───┼───┼───┤
│   │   │   │   │
├───┼───┼───┼───┤
│   │   │   │   │
└───┬───┬───┴───┘
```

**The MDP:**
- States: 16 cells (15 regular + 1 terminal Goal)
- Actions: {↑, ↓, ←, →}
- Transitions: deterministic (unless blocked by a wall)
- Rewards: -1 per step, +10 for reaching the Goal

**Applying Value Iteration:**

```python
# Pseudocode for GridWorld
V = np.zeros((4, 4))
gamma = 0.9
theta = 1e-4

while True:
    delta = 0
    for i in range(4):
        for j in range(4):
            if (i,j) == (0,3):  # Goal
                continue
            if (i,j) == (1,1):  # Obstacle
                continue
            
            v = V[i,j]
            # Max over the 4 directions
            values = []
            for action in ['up', 'down', 'left', 'right']:
                ni, nj = next_pos(i, j, action)
                reward = 10 if (ni,nj)==(0,3) else -1
                values.append(reward + gamma * V[ni, nj])
            
            V[i,j] = max(values)
            delta = max(delta, abs(v - V[i,j]))
    
    if delta < theta:
        break
```

**Result:**

After convergence, $V(s)$ reflects "distance to the goal" (weighted by $\gamma$).

The optimal policy: arrows pointing toward the Goal.

---

## 13. Summary

| Concept | Description |
|-----------|----------|
| **DP** | Iterative application of the Bellman equations given a known MDP model |
| **Policy Evaluation** | Computing $V_\pi(s)$ for a fixed policy |
| **Policy Improvement** | Greedily improving the policy based on $V$ |
| **Policy Iteration** | Alternating evaluation and improvement until convergence |
| **Value Iteration** | Combining evaluation and improvement into a single optimal update |
| **GPI** | The general scheme of evaluation/improvement interplay (the basis of all RL) |
| **Asynchronous methods** | Updating a subset of states to speed things up |

**Key takeaways:**

1. DP requires full knowledge of the MDP → applicable only in simulations
2. DP's principles (GPI) underlie every RL algorithm
3. Model-free methods replace the mathematical expectation with sampling
4. Value Iteration is usually more efficient than Policy Iteration in practice

---

## 14. Connections to earlier sessions

- **[note_02_rl_framework_and_mdp.md](note_02_rl_framework_and_mdp.md):** MDP formalization (states, actions, transitions, rewards)
- **[note_07_bellman_equation.md](note_07_bellman_equation.md):** The Bellman equation — the foundation of DP
- **[note_08_monte_carlo_vs_td.md](note_08_monte_carlo_vs_td.md):** Monte Carlo — a model-free alternative to Policy Evaluation
- **[note_09_q_learning.md](note_09_q_learning.md):** Q-Learning — a model-free alternative to Value Iteration

---

## 15. Further reading

Recommended sources:

- **Sutton & Barto, Chapter 4:** Dynamic Programming
- **Maxim Lapan, Chapter 5:** Tabular Learning and the Bellman Equation
- **David Silver's RL Course, Lecture 3:** Planning by Dynamic Programming

---

## 16. Hands-on assignment

Implemented in `code/13_dynamic_programming/`:

1. **GridWorld environment** — a custom environment with obstacles
2. **Policy Evaluation** — iterative evaluation of an arbitrary policy
3. **Policy Iteration** — the full algorithm alternating evaluation/improvement
4. **Value Iteration** — the optimal Bellman update
5. **Visualization** — heatmaps of $V(s)$, policy arrows, a convergence animation

**Experiments:**
- Compare the convergence speed of Policy Iteration vs Value Iteration
- Study the effect of $\gamma$ on the optimal policy
- Implement Prioritized Sweeping and compare it against the synchronous method

---

**Next:** [note_14_ppo_trpo.md](note_14_ppo_trpo.md) — PPO and TRPO
