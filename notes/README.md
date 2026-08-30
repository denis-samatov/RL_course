# Theoretical notes on Reinforcement Learning

A series of structured notes for studying reinforcement learning.

## Notes vs. coding sessions

There are 17 notes but only 10 coding sessions (`code/08_mc_vs_td` through `code/17_final_project`). Notes 01-07 are **theory-only** — they build the conceptual foundation (RL framework, MDPs, exploration/exploitation, policy vs. value methods, Bellman equations) that sessions 08 onward put into code:

| Note | Topic | Coding session |
|---|---|---|
| note_01 | Introduction to Deep RL | *(theory only)* |
| note_02 | RL framework and MDP | *(theory only)* |
| note_03 | Exploration vs exploitation | *(theory only)* |
| note_04 | Policy-based vs value-based methods | *(theory only)* |
| note_05 | Deep RL function approximators | *(theory only)* |
| note_06 | Value-based methods | *(theory only)* |
| note_07 | Bellman equation | *(theory only)* |
| note_08 | Monte Carlo vs TD | [`code/08_mc_vs_td/`](../../code/08_mc_vs_td/) |
| note_09 | Q-Learning | [`code/09_q_learning_bellman/`](../../code/09_q_learning_bellman/) |
| note_10 | Deep Q-Network | [`code/10_deep_q_network/`](../../code/10_deep_q_network/) |
| note_11 | Policy Gradients / REINFORCE | [`code/11_policy_gradient/`](../../code/11_policy_gradient/) |
| note_12 | Actor-Critic / A2C | [`code/12_actor_critic/`](../../code/12_actor_critic/) |
| note_13 | Dynamic Programming | [`code/13_dynamic_programming/`](../../code/13_dynamic_programming/) |
| note_14 | PPO / TRPO | [`code/14_ppo_trpo/`](../../code/14_ppo_trpo/) |
| note_15 | RLHF pipeline | [`code/15_rlhf_basics/`](../../code/15_rlhf_basics/) |
| note_16 | DPO and variants | [`code/16_dpo/`](../../code/16_dpo/) |
| note_17 | Final project | [`code/17_final_project/`](../../code/17_final_project/) |

## Course structure

### 📘 [note_01_introduction_to_deep_rl.md](note_01_introduction_to_deep_rl.md) — Introduction to Deep Reinforcement Learning

**Main topics:**

* What Reinforcement Learning is
* The Agent-Environment loop
* The Markov Decision Process (MDP)
* Example: LunarLander
* Stable-Baselines3: a PPO implementation
* The reward hypothesis (introduction)

**Level:** Introductory, hands-on

---

### 📗 [note_02_rl_framework_and_mdp.md](note_02_rl_framework_and_mdp.md) — The Reinforcement Learning Framework

**Main topics:**

* The agent-environment interaction loop, in detail
* Formalizing it as an MDP
* The reward hypothesis (in depth)
* The Markov property
* An agent's policy (stochastic and deterministic)
* State vs. observation
* Action spaces (discrete and continuous)
* Reward discounting

**Level:** Intermediate, theoretical

---

### 📙 [note_03_exploration_vs_exploitation.md](note_03_exploration_vs_exploitation.md) — Exploration vs Exploitation

**Main topics:**

* The exploration/exploitation trade-off
* A formal representation
* Strategies for managing the trade-off:
  * ε-greedy
  * Softmax (Boltzmann Exploration)
  * Upper Confidence Bound (UCB)
* Practical applications in RL algorithms

**Level:** Intermediate, conceptual

---

### 📕 [note_04_policy_vs_value_methods.md](note_04_policy_vs_value_methods.md) — Policy-Based vs Value-Based Methods

**Main topics:**

* Two approaches to solving RL problems
* Policy-Based methods (learn the policy directly)
  * REINFORCE, PPO, Actor-Critic
  * Stochastic and deterministic policies
* Value-Based methods (learn value functions)
  * The V-function and Q-function
  * Q-Learning, SARSA, DQN
* Comparing the approaches and their applications
* Actor-Critic as a hybrid method

**Level:** Intermediate-advanced, methodological

---

### 📕 [note_05_deep_rl_approximators.md](note_05_deep_rl_approximators.md) — What "Deep" Means in Deep RL

**Main topics:**

* The problem with tabular methods and their limitations
* Neural networks as function approximators
* Classical Q-Learning vs. Deep Q-Learning
* DeepMind's 2015 breakthrough — DQN for Atari
* Experience Replay and the Target Network
* When to use Deep RL
* Comparing classical and Deep RL

**Level:** Intermediate-advanced, technical

---

### 📙 [note_06_value_based_methods.md](note_06_value_based_methods.md) — Two Kinds of Value-Based Methods

**Main topics:**

* A detailed explanation of value functions
* The state-value function (V-function)
* The action-value function (Q-function)
* The relationship between a policy and its value function
* Greedy and ε-greedy strategies
* The relationship between V and Q
* The computation problem, and an introduction to the Bellman equation

**Level:** Intermediate, mathematical

---

### 📘 [note_07_bellman_equation.md](note_07_bellman_equation.md) — The Bellman Equation

**Main topics:**

* The problem of computing value functions directly
* Bellman's idea: a recursive definition
* The Bellman equation for the V-function
* The Bellman equation for the Q-function
* Key parameters (reward, discounting)
* The analogy to dynamic programming
* Worked examples

**Level:** Intermediate-advanced, mathematical

---

### 📗 [note_08_monte_carlo_vs_td.md](note_08_monte_carlo_vs_td.md) — Monte Carlo vs Temporal Difference

**Main topics:**

* Two practical approaches to computing value
* Monte Carlo: learning from completed episodes
* Temporal Difference: learning step by step
* Comparison: accuracy vs. speed
* The concept of bootstrapping
* TD(λ) as a compromise between MC and TD
* Code examples and visualization

**Level:** Intermediate-advanced, hands-on

---

### 📘 [note_09_q_learning.md](note_09_q_learning.md) — Q-Learning: the Foundational Action-Learning Algorithm

**Main topics:**

* What Q-Learning is (off-policy, value-based TD)
* The difference between Reward and Value
* The Q-table: the agent's memory
* The Q-Learning algorithm, step by step
* The Q-function update formula
* Off-policy vs. On-policy (Q-Learning vs. SARSA)
* ε-greedy exploration
* Worked examples of Q-table updates

**Level:** Intermediate-advanced, algorithmic

---

### 📘 [note_10_deep_q_network.md](note_10_deep_q_network.md) — Deep Q-Network (DQN)

**Main topics:**

* Scaling tabular Q-Learning to large state spaces
* The DQN architecture (MLP Q-function approximator)
* Experience Replay
* The Target Network
* The TD-target and loss
* Companion coding session: [`code/10_deep_q_network/`](../../code/10_deep_q_network/)

**Level:** Advanced, hands-on

---

### 📗 [note_11_policy_gradients_reinforce.md](note_11_policy_gradients_reinforce.md) — Policy Gradients and REINFORCE

**Main topics:**

* The Policy Gradient Theorem
* The REINFORCE algorithm (Monte Carlo Policy Gradient)
* Using a baseline to reduce variance
* Entropy regularization
* Companion coding session: [`code/11_policy_gradient/`](../../code/11_policy_gradient/)

**Level:** Advanced, hands-on

---

### 📙 [note_12_actor_critic_a2c.md](note_12_actor_critic_a2c.md) — Actor-Critic Methods and A2C

**Main topics:**

* The Actor-Critic architecture
* Advantage Actor-Critic (A2C)
* The TD-error as an advantage estimate
* Continuous action spaces (Gaussian policies)
* Companion coding session: [`code/12_actor_critic/`](../../code/12_actor_critic/)

**Level:** Advanced, hands-on

---

### 📕 [note_13_dynamic_programming.md](note_13_dynamic_programming.md) — Dynamic Programming in Reinforcement Learning

**Main topics:**

* Policy Evaluation, Policy Iteration, Value Iteration
* Generalized Policy Iteration (GPI)
* Requires a known environment model
* Companion coding session: [`code/13_dynamic_programming/`](../../code/13_dynamic_programming/)

**Level:** Advanced, mathematical

---

### 📘 [note_14_ppo_trpo.md](note_14_ppo_trpo.md) — Proximal Policy Optimization (PPO) and Trust Region Policy Optimization (TRPO)

**Main topics:**

* The PPO-Clip objective
* Generalized Advantage Estimation (GAE)
* TRPO's trust-region approach
* Companion coding session: [`code/14_ppo_trpo/`](../../code/14_ppo_trpo/)

**Level:** Advanced, hands-on

---

### 📗 [note_15_rlhf_pipeline.md](note_15_rlhf_pipeline.md) — RLHF: Reinforcement Learning from Human Feedback

**Main topics:**

* The 3-stage RLHF pipeline: SFT → Reward Model → PPO
* The Bradley-Terry preference model
* The KL penalty against the reference policy
* Reward hacking
* Companion coding session: [`code/15_rlhf_basics/`](../../code/15_rlhf_basics/)

**Level:** Advanced, applied

---

### 📙 [note_16_dpo_and_variants.md](note_16_dpo_and_variants.md) — DPO: Direct Preference Optimization

**Main topics:**

* Optimizing directly on preferences, without a separate Reward Model
* The DPO loss
* Comparison with PPO-RLHF
* Companion coding session: [`code/16_dpo/`](../../code/16_dpo/)

**Level:** Advanced, applied

---

### 📕 [note_17_final_project.md](note_17_final_project.md) — Final Project: Course Wrap-Up and Directions Forward

**Main topics:**

* Integrating everything covered in the course
* The mini-RLHF pipeline project
* Where to go next in RL

**Level:** Capstone

---

## Recommended study order

```text
note_1 → note_2 → note_3 → note_4 → note_5 → note_6 → note_7 → note_8 → note_9 → note_10 → note_11 → note_12 → note_13 → note_14 → note_15 → note_16 → note_17
```

1. Start with **note_1** for a general overview and a hands-on example
2. Move to **note_2** for a deep understanding of the formal framework
3. Study **note_3** to master the core exploration/exploitation concept
4. Work through **note_4** to understand the approaches to solving RL problems
5. Go through **note_5** — why and how neural networks are used in RL
6. Dig into **note_6** — a detailed understanding of the V and Q functions
7. Study **note_7** — the Bellman equation as RL's mathematical foundation
8. Work through **note_8** — practical estimation methods (MC vs TD)
9. Implement **note_9** — your first practical algorithm, Q-Learning
10. Continue through **note_10-17** as the coding sessions get more advanced, from Deep Q-Networks through PPO, RLHF, DPO, and the final project

---

## Navigation

All notes are cross-linked. Each note ends with a "What's next" section with recommendations for continuing.

### How the material is spread across the notes

**Formulas and concepts are introduced gradually:**

* **note_1:** Basic concepts (MDP, $G_t$, $\pi$) — a first introduction
* **note_2:** Formalizing the MDP, a detailed explanation of $G_t$, the Markov property
* **note_3:** Exploration strategies (ε-greedy, Softmax, UCB)
* **note_4:** Introducing the Q-function and the basic Q-Learning formula
* **note_5:** A detailed explanation of Q-Learning and the transition to DQN (Experience Replay, Target Network)
* **note_6:** A deep understanding of the V and Q functions, their relationship, and a first mention of Bellman
* **note_7:** The Bellman equation — the recursive definition of value, the mathematical foundation
* **note_8:** Monte Carlo vs Temporal Difference — practical estimation methods
* **note_9:** The Q-Learning algorithm — the TD update to the Q-table, off-policy learning
* **note_10-17:** Each builds on the previous ones, moving from Deep Q-Networks through policy gradients, Actor-Critic, dynamic programming, PPO/TRPO, and RLHF/DPO to the final project

Each note builds on the previous ones without duplicating their content.

---

## Sources

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
* Hadelin de Ponteves, *AI Crash Course* (2019)
* RL Theory Book (Forts & Mills, 2022)
