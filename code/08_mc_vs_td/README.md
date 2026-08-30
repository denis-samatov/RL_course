### FrozenLake-v1: MC prediction vs TD(0) under a fixed policy

- **Environment**: `FrozenLake-v1` — a discrete MDP on a 4x4 ice grid. The goal is to reach the target while avoiding holes.
- **States**: a finite set $\mathcal{S}$; **actions**: a finite set $\mathcal{A}$.
- **Dynamics**: stochastic when `is_slippery=True`. Reward $r \in \{0,1\}$, given only on reaching the goal.

This experiment does not solve the control problem (it does not search for an optimal policy); instead it compares two approaches to estimating $V_\pi(s)$ for a fixed policy $\pi$:
- First-visit Monte Carlo (MC) prediction
- Temporal Difference TD(0) prediction

The code uses a uniform policy $\pi$ (all actions equally likely):
$$
\pi(a\mid s) = \tfrac{1}{|\mathcal{A}|} \quad \text{for all } s, a.
$$

### Problem statement (estimating $V^\pi$)

Estimate the state-value function under a given policy $\pi$:
$$
V_\pi(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \mid s_0 = s \right], \quad \gamma \in [0,1).
$$

---

## Monte Carlo prediction (first-visit)

MC uses complete returns without bootstrapping.
- Return from step $t$:
$$
G_t = \sum_{k=t}^{T-1} \gamma^{\,k-t}\, r_{k+1}.
$$
- The first-visit estimate updates $V(s)$ only on the first occurrence of state $s$ in an episode, via an incremental average:
$$
V(s) \leftarrow V(s) + \frac{1}{N(s)}\bigl(G_t - V(s)\bigr),
$$
where $N(s)$ is the number of first visits to $s$ (a per-episode counter).

In `mc_td_algorithm.py`: `monte_carlo_prediction(...)` generates episodes under $\pi$ and tracks the history of the starting state's estimate $V(\text{start})$.

---

## TD(0) prediction

TD(0) uses one-step bootstrapping, updating the estimate based on the current estimate of the next state:
$$
V(s) \leftarrow V(s) + \alpha\,\bigl(r + \gamma V(s') - V(s)\bigr),
$$
where $\alpha$ is the learning rate.

In `mc_td_algorithm.py`: `td0_prediction(...)` performs online updates under the same policy $\pi$, and likewise accumulates a history of $V(\text{start})$.

---

## What's compared and what's plotted

- **Convergence curves**: a plot of the moving average of the $V(\text{start})$ estimates for MC (first-visit) and TD(0), letting you compare convergence speed and stability.
- **Value maps**: heatmaps of the $V(s)$ estimates for every state in the 4x4 grid, for MC and TD(0) separately.

---

## Experiment parameters

The `run_comparison(...)` function takes:
- `episodes` — number of training episodes (default 5000)
- `gamma` — discount factor
- `alpha` — learning rate for TD(0)
- `max_steps` — maximum steps per episode
- `slippery` — whether to use stochastic environment transitions
- `seed` — base seed

The policy is uniform (see `uniform_policy(...)`). Episode generation and the environment step use the Gym v0.26+ API.

---

## How to run

- Running the file directly will show the plots:
```bash
python code/08_mc_vs_td/mc_td_algorithm.py
```
- If needed, change the parameters in `run_comparison(...)` (at the bottom of the file), or call the function from your own code.

---

## Key differences: MC vs TD(0)

- **Source of information**: MC uses only complete returns; TD bootstraps via current estimates.
- **Bias/variance**: MC is unbiased but high-variance; TD is biased but lower-variance.
- **Online learning**: TD can learn incrementally at every step; MC requires completed episodes.

---

## Mapping to the code

- `monte_carlo_prediction` — first-visit MC estimation of $V^\pi$.
- `td0_prediction` — TD(0) estimation of $V_\pi$.
- `uniform_policy` — the fixed uniform policy $\pi$.
- `run_comparison` — runs both methods, plotting $V(\text{start})$ curves and $V(s)$ heatmaps.
