# Theoretical Note #1

## Topic: Introduction to Deep Reinforcement Learning

---

## 1. What is Reinforcement Learning (RL)

**Reinforcement Learning** is a branch of machine learning in which an **agent** interacts with an **environment** by taking **actions**, in order to **maximize the reward** it receives over time.
The core idea:

> "Learn from your own experience through trial and error."

---

### The Agent-Environment loop

1. The agent is in the environment's state $s_t$
2. It chooses an action $a_t$
3. The environment returns:

   * a new state $s_{t+1}$
   * and a reward $r_{t+1}$
4. The agent updates its policy $\pi(a|s)$ to **maximize the total reward**.

$$
s_t \xrightarrow{a_t} (r_{t+1}, s_{t+1})
$$

![The Agent-Environment loop](images/Screenshot%202025-10-24%20at%2009.59.06.png)

---

### The Markov Decision Process (MDP)

Intuitively: the environment is described by states, actions, transition probabilities, rewards, and discounting. The full formal definition and a breakdown of the components (including the relationship $r(s,a)=\mathbb{E}[R_{t+1}\mid s,a]$) are in [note_02_rl_framework_and_mdp.md](note_02_rl_framework_and_mdp.md).

---

### The agent's goal

Maximize the **expected sum of discounted rewards**:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

Find the **optimal policy** $\pi^*(a|s)$ such that:

$$
\pi^* = \arg\max_\pi \mathbb{E}_\pi [G_t]
$$

---

## 2. Why *Deep* Reinforcement Learning?

Classical RL used tables (e.g. Q-tables), but this is infeasible for large state spaces.
The solution is **deep neural networks**, which approximate value functions or policies.

| Classical RL | Deep RL |
|-----------------|---------|
| Tabular methods | Neural networks |
| Simple environments | Complex tasks (video, robotics) |
| Limited scalability | High generalization capacity |

Examples of the transformation:

* Q-Learning → **Deep Q-Learning (DQN)**
* Policy Gradient → **Deep Policy Gradient**
* Actor-Critic → **Deep Actor-Critic**

> **For more on how and why neural networks work in RL:** see [note_05_deep_rl_approximators.md](note_05_deep_rl_approximators.md)

---

## 3. Example: LunarLander

The `LunarLander-v2` environment from **Gymnasium**.

**Task:** land the lunar module between the flags, using the minimum amount of fuel.

**State $s_t$:**

* coordinates (x, y)
* velocities (vx, vy)
* angle and angular velocity
* leg contact flags (0/1)

**Actions $a_t$:**

1. do nothing
2. fire the left engine
3. fire the main engine
4. fire the right engine

**Rewards:**

* +100...+140 — a successful landing
* -100 — a crash
* -0.3 — for using fuel
* +10 — for touching down without crashing

---

## 4. Stable-Baselines3: a PPO implementation

```python
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("lunar_lander_agent")
```

---

## 5. What you'll learn

* Understand the fundamental principles of RL
* Run your first environment (`LunarLander-v2`)
* Train an agent using PPO or DQN
* Evaluate its behavior
* Upload the model to the Hugging Face Hub

---

## 6. The Reward Hypothesis

> **The Reward Hypothesis — RL's central idea:**
>
> Any goal can be described as the problem of **maximizing expected cumulative reward**.

**Why does this matter?**

Reinforcement learning rests on a fundamental idea: any goal task — whether it's learning to walk, playing games, controlling a robot, or trading stocks — can be represented as maximizing expected cumulative reward.

**Key consequences:**

* **A single mathematical foundation** for every RL task
* **No direct labels** — the agent forms its own strategy through experience
* **Flexibility** — changing the rewards can completely change the agent's behavior

> Even when an agent's goal looks complex on the surface (e.g. "land the lunar module"), in RL terms it's formulated as "maximize the cumulative reward from its actions."

**Detailed explanation:** for the mathematical formalization, examples, and a detailed analysis, see section 4 of [note_02_rl_framework_and_mdp.md](note_02_rl_framework_and_mdp.md)

---

**Based on:**

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2020)
* Andrea Lonza, *Reinforcement Learning Algorithms with Python* (2020)
* Hadelin de Ponteves, *AI Crash Course* (2019)
* RL Theory Book (Forts & Mills, 2022)
