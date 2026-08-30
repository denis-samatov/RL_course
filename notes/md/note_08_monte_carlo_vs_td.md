# Theoretical Note #8

## Topic: Monte Carlo vs Temporal Difference Learning (MC vs TD)

> **Related to:** [note_07_bellman_equation.md](note_07_bellman_equation.md) — The Bellman equation · [note_06_value_based_methods.md](note_06_value_based_methods.md) — The V and Q functions

---

## 1. Why this section matters

Now that we've covered the Bellman equation, a practical question arises:

> "How can an agent *approximately compute* this value, given it doesn't have access to the whole environment or all future rewards?"

Two fundamental approaches:

* **Monte Carlo (MC)** — learning from *completed episodes*;
* **Temporal Difference (TD)** — learning *during* the interaction, step by step.

---

## 2. Monte Carlo: learning from an entire episode

**Intuition:** the agent **plays out the whole episode first**, then aggregates the rewards and updates its value estimates.

### Mathematically

$$
V(S_t) \leftarrow V(S_t) + \alpha \big[G_t - V(S_t)\big]
$$
where the return
$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots
$$

### What happens in practice

1. The agent starts an episode and acts under its current policy (e.g. $\varepsilon$-greedy).
2. Once the episode ends, it computes $G_t$ for every time step.
3. It updates every visited state $S_t$ using its corresponding $G_t$.

### Example

Say the episode's rewards are: $R = [1, 0, 0, 0, 1, 1]$.
Then $G_0 = 1 + 0 + 0 + 0 + 1 + 1 = 3$. With learning rate $\alpha = 0.1$:
$$
V(S_0) \leftarrow 0 + 0.1\,(3 - 0) = 0.3
$$
After several episodes, the estimates $V(S_t)$ begin to reflect the expected quality of the state.

### Advantages of MC

* Simple to understand and implement.
* Doesn't require knowledge of the environment's dynamics $P(s'\mid s,a)$.
* Gives **unbiased** estimates of $V_\pi(s)$ (given enough samples).

### Drawbacks of MC

* Requires waiting until the episode ends.
* Inefficient/unusable for infinite-horizon tasks (continuous tasks with no natural terminal state).
* Updates are infrequent → high variance in the estimates.

---

## 3. Temporal Difference (TD): learning during the interaction

**Intuition:** the agent **updates its estimates right after every step**, without waiting for the episode to end, using its current estimate of future value $V(S_{t+1})$.

### The TD(0) formula

$$
V(S_t) \leftarrow V(S_t) + \alpha \big[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \big]
$$
Here
$$
R_{t+1} + \gamma V(S_{t+1})
$$
is the **TD target**, and the expression in brackets is the **TD error** ($\delta_t$):
$$
\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t).
$$

### Intuition (an example)

If $R_{t+1}=1$ and $V(S_{t+1})=0$, then with $\alpha=0.1,\ \gamma=1$:
$$
V(S_0) \leftarrow 0 + 0.1\,[1 + 1\cdot 0 - 0] = 0.1.
$$
We updated the value **right away**, without waiting for the episode to finish.

### TD's key feature: bootstrapping

TD *partially* relies on its own estimate of the future, $V(S_{t+1})$ (bootstrapping), rather than the full return $G_t$ as in MC.

---

## 4. Comparing Monte Carlo and Temporal Difference

| Criterion | Monte Carlo | Temporal Difference |
| ------------------------------ | ------------------------------------ | -------------------------------------------- |
| Based on | The full episode | A single step |
| What's used | The actual $G_t$ | The approximate $R_{t+1} + \gamma V(S_{t+1})$ |
| Updates | After the episode | After every step |
| Requires the episode to end | Yes | No |
| Type of estimate | Unbiased, but high variance | Biased, but lower variance |
| Suited to infinite-horizon tasks | No | Yes |
| Bootstrapping | No | Yes |
| Examples | Blackjack, short episodes | CartPole, FrozenLake |

---

## 5. A unifying idea: TD($\lambda$)

In practice, both approaches are often combined: **MC** gives accuracy, **TD** gives speed. The intermediate form is **TD($\lambda$)**, where $\lambda \in [0,1]$:

* $\lambda = 0$ → TD(0)
* $\lambda = 1$ → Monte Carlo
* Intermediate values of $\lambda$ trade off bias against variance.

(Implemented via **eligibility traces**.)

---

## 6. A visual analogy

**Monte Carlo:** "I'll wait until everything is over, then tally up the results."

**TD:** "I've already spotted the trend, and I adjust at every step."

---

## 7. Code updates (intuitively)

**Note:** The code uses the Gymnasium API (version >= 0.26.0). If you're using an older Gym (<0.26), replace:
- `state, info = env.reset()` → `state = env.reset()`
- `next_state, reward, terminated, truncated, info = env.step(action)` → `next_state, reward, done, _ = env.step(action)`

```python
# Monte Carlo (at the end of an episode)
# First, collect the trajectory
trajectory = []  # List[(state, reward)]
state, info = env.reset()
done = False

while not done:
    action = choose_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    trajectory.append((state, reward))
    state = next_state
    done = terminated or truncated

# Then update V for every visited state
G = 0
visited_states = set()
for state, reward in reversed(trajectory):
    G = reward + gamma * G
    # First-visit MC: only update on the first visit
    state_key = tuple(state) if isinstance(state, np.ndarray) else state
    if state_key not in visited_states:
        V[state_key] = V.get(state_key, 0.0) + alpha * (G - V.get(state_key, 0.0))
        visited_states.add(state_key)

# Temporal Difference (during the episode)
state, info = env.reset()
for t in range(max_steps):
    action = choose_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Convert states to hashable keys
    state_key = tuple(state) if isinstance(state, np.ndarray) else state
    next_state_key = tuple(next_state) if isinstance(next_state, np.ndarray) else next_state

    # TD(0) update
    V[state_key] = V.get(state_key, 0.0) + alpha * (
        reward + gamma * V.get(next_state_key, 0.0) - V.get(state_key, 0.0)
    )

    state = next_state
    if done:
        break
```

---

## 8. Conclusion

Both methods learn the **value of states, $V(s)$**, but they use experience differently. TD became the foundation of many algorithms:

* **SARSA** (on-policy TD)
* **Q-Learning** (off-policy TD)
* **Expected SARSA**
* **TD($\lambda$)**
* **DQN** (Deep Q-Network)
* **A3C/A2C**

---

## 9. Formula comparison (cheat sheet)

**Monte Carlo:**
$$
\boxed{ V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)] }
$$

**Temporal Difference:**
$$
\boxed{ V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] }
$$

Where $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the TD error.

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Hugging Face Deep RL Course, Unit 2
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
