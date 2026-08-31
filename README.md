# 🎓 Reinforcement Learning — Practical Course (2025)

[![CI](https://github.com/denis-samatov/reinforcement_learning_course/actions/workflows/ci.yml/badge.svg)](https://github.com/denis-samatov/reinforcement_learning_course/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A practical, code-first reinforcement learning course for anyone who wants to build and
run RL algorithms themselves rather than only read about them — from tabular Q-learning
through policy gradients to RLHF (PPO, DPO), with runnable sessions in Python, Gymnasium,
and PyTorch. Licensed under [MIT](LICENSE).

> **Goal:** Understand and implement the core reinforcement learning (RL) algorithms —
> from MDPs and the Bellman equations through RLHF (PPO, DPO).
> **Format:** 10 one-hour sessions, with code in Python, Gymnasium, and PyTorch.

---

## 📘 Overview

| Parameter | Description |
|-----------|-----------|
| **Period** | October 20 → December 28, 2025 |
| **Cadence** | 1 session per week (10 sessions × 60 minutes) |
| **Session format** | 10 min theory → 40 min code/experiment → 10 min discussion |
| **Tools** | `python 3.10`, `gymnasium`, `torch`, `numpy`, `matplotlib`, `stable-baselines3`, `trl`, `wandb` |
| **Assessment** | mini-homework after each session + a final mini-project |
| **Resources** | CPU / GPU (recommended), `venv` environment |

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/denis-samatov/reinforcement_learning_course.git
cd reinforcement_learning_course
```

### 2. Create and activate a virtual environment

Using a virtual environment to isolate dependencies is recommended.

```bash
# Create the environment
python3 -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install dependencies

All required libraries are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 🗂️ Project structure

- **`code/`**: hands-on code for each session, grouped by topic. Each subfolder
  contains Python scripts (`.py`) and/or interactive notebooks (`.ipynb`).
- **`notes/`**: theory notes and slides for each session.
- **`.gitignore`**: tells Git which files/folders to ignore.
- **`README.md`**: this file — the main course overview.
- **`requirements.txt`**: Python dependencies for the project.

## ⚡ Running the examples

Each session in `code/` has its own set of scripts and notebooks.

**Running a Python script:**

1. Go to the relevant session's folder.
2. Run the main script.

```bash
# Example for session 8
cd code/08_mc_vs_td/
python mc_td_algorithm.py
```

**Running a Jupyter Notebook:**

1. Make sure Jupyter is installed (`pip install jupyter`).
2. Launch Jupyter Lab or Notebook from the project root.
3. Navigate to the relevant folder in the Jupyter interface and open the `.ipynb` file.

```bash
jupyter lab
```

> **Important:** Make sure the Jupyter kernel uses the virtual environment you created
> (`venv`), so all dependencies are available.

---

## 🗓️ Session plan (October → December 2025)

> **Note on numbering:** the `#` column below uses this table's original planning order (1-10). The actual `code/` directories are numbered 08-17 and don't line up 1:1 with these rows — some environments also changed during implementation (e.g. the DQN session below lists CartPole-v1, but `code/10_deep_q_network/` implements it on LunarLander-v2). See [`notes/README.md`](notes/README.md) for the accurate note-to-code-directory mapping.

| # | Topic | Key concepts | Practice |
|---|------|------------------|-----------|
| **1** | Intro to RL and MDPs | agent–environment–reward, `V`/`Q` functions, Bellman equations | `CartPole-v1` (random policy, MC estimation) |
| **2** | Dynamic Programming | Policy / Value Iteration, GPI | `GridWorld` — visualizing `V(s)` |
| **3** | Monte Carlo and TD | Sarsa, Q-Learning, ε-greedy | `FrozenLake-v1` (on/off-policy) |
| **4** | DQN and extensions | Target network, Replay, Double/Dueling DQN | `CartPole-v1` — PyTorch implementation |
| **5** | Policy Gradients, REINFORCE | PG theorem, baseline, entropy | `LunarLander-v2` — PG with baseline |
| **6** | Actor–Critic, A2C, and SAC (concept) | GAE, A2C, max-entropy RL | `CartPole`, `Pendulum-v1` |
| **7** | PPO and TRPO | surrogate objective, clipping, KL control | `Pendulum-v1` — PPO (SB3) |
| **8** | RLHF: SFT → RM → PPO | KL penalty, stability and drift | `trl` — PPO fine-tune on toy data |
| **9** | Reward Modeling and DPO | binary RM, DPO update | `trl.DPOTrainer` — PPO vs DPO comparison |
| **10** | Final project and safety | reward/KL/safety metrics, report | Mini RLHF pipeline (SFT + RM + PPO / DPO) |

---

## 🗂️ Session structure

Every coding session under `code/` follows the same template:

- `code/NN_topic/`
  - `README.md` — goals, how to run the algorithm/notebook, links to the matching theory note.
  - `*_algorithm.py` (or similarly named) — the core algorithm implementation.
  - `*_demo.ipynb` — a runnable demo notebook.
  - `homework.ipynb` — exercises for the session (with `TODO` placeholders).
  - `homework_solution.ipynb` — worked solutions.

Coding sessions are numbered 08-17 (see [`notes/README.md`](notes/README.md) for how each note maps to its coding session — notes 01-07 are theory-only, with no matching code directory).

---

## 🎯 Learning outcomes

| Session | Outcome |
|----------|------------|
| 1 | Formalizes MDPs, explains `V`, `Q`, `π` |
| 2 | Implements Value / Policy Iteration and analyzes convergence |
| 3 | Implements Sarsa and Q-Learning, tunes ε-greedy |
| 4 | Builds a DQN (PyTorch), adds Double / Dueling |
| 5 | Derives the PG formulas, reduces variance (baseline/entropy) |
| 6 | Runs A2C, understands GAE and the max-entropy idea |
| 7 | Tunes PPO (clip / KL / entropy), compares against A2C |
| 8 | Understands the RLHF pipeline and the role of the KL penalty |
| 9 | Trains a Reward Model and performs a DPO update |
| 10 | Assembles a mini-RLHF pipeline and evaluates its metrics |

---

## 💡 Final projects and research directions

1. **Mini-RLHF** — fine-tuning an LLM on short instructions: SFT → RM (200–500 pairs)
   → PPO / DPO.
2. **PPO vs DPO** — analyzing stability and resource cost across different `β`, `LR`,
   `batch_size`.
3. **RL + Human Scoring** — manual scoring of trajectories and measuring policy drift.

---

## 📚 Recommended reading

| Source | Content |
|-----------|-------------|
| **Sutton & Barto** — *Reinforcement Learning: An Introduction* | the classic reference (MDPs, DP, TD, PG) |
| **Maxim Lapan** — *Deep RL Hands-On* | practical PyTorch implementations |
| **Andrea Lonza** — *RL Algorithms with Python* | SARSA, Q-Learning, PPO, TRPO, DDPG / TD3 |
| **RL Theory Book** — Forts & Mills | rigorous derivations of PG, PPO, GAE |
| **Hugging Face TRL Docs** | RLHF, PPO, DPO, ORPO — modern pipelines |

---

> 💬 Built on ideas from **Sutton & Barto**, **Lapan**, **Lonza**, **RL Theory Book**,
> and **Hugging Face TRL**.
> Author: *Denis Samatov, TPU / 2025*
>
> 📂 This repository includes:
> — study notes (formulas, visualizations)
> — hands-on notebooks with code
> — demos and video recordings, mini-MDP examples
> — shared modules (algorithms, policies, utilities)
> — RL and RLHF mini-projects
>
> 🧭 Goal: bridge **academic RL** and **applied Deep RL / RLHF**.
