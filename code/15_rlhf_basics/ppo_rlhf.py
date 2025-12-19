"""RLHF-style fine-tuning with a tiny PPO/REINFORCE hybrid.

We intentionally keep the update rule simple so it can be derived live:
total_reward = reward_model - beta * KL_to_SFT
loss ≈ -E[advantage * log π(a|s)]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from simple_text_env import CANDIDATES, SimpleTextEnv
from sft_model import TabularPolicy
from reward_model import RewardModel


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(p || q) for two categorical distributions."""
    eps = 1e-9
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def ppo_train(
    actor: TabularPolicy,
    ref_policy: TabularPolicy,
    rm: RewardModel,
    env: SimpleTextEnv,
    beta: float = 0.01,
    epochs: int = 120,
    batch_size: int = 16,
    lr: float = 0.1,
) -> TabularPolicy:
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        prompts = env.sample_prompts(batch_size)
        actions = []
        rewards = []
        kls = []
        logps = []
        for pid, prompt in zip(prompts.prompt_ids, prompts.prompts):
            probs = actor.predict_probs(pid)
            act = int(rng.choice(len(CANDIDATES), p=probs))
            ref_probs = ref_policy.predict_probs(pid)
            r = rm.score(prompt, CANDIDATES[act])
            kl = kl_divergence(probs, ref_probs)
            actions.append(act)
            rewards.append(r - beta * kl)
            kls.append(kl)
            logps.append(np.log(probs[act] + 1e-9))

        rewards_arr = np.array(rewards)
        baseline = rewards_arr.mean()
        advantages = rewards_arr - baseline

        # Simple policy-gradient style update.
        for pid, act, adv in zip(prompts.prompt_ids, actions, advantages):
            probs = actor.predict_probs(pid)
            # Gradient of log-softmax for categorical variables.
            grad = -probs
            grad[act] += 1.0
            actor.logits[pid] += lr * adv * grad
    return actor


def run(beta: float, output_dir: Path) -> None:
    env = SimpleTextEnv(seed=0)
    base = Path(__file__).parent
    ckpt_dir = output_dir
    actor_path = ckpt_dir / "sft_policy.npy"
    rm_path = ckpt_dir / "reward_model.npy"

    ref_policy = TabularPolicy.load(actor_path)
    actor = TabularPolicy.load(actor_path)
    rm = RewardModel.load(rm_path)

    actor = ppo_train(actor, ref_policy, rm, env, beta=beta)
    (ckpt_dir / "ppo_actor.npy").parent.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / f"ppo_actor_beta_{beta}.npy", actor.logits)

    # Quick sanity print
    for i, prompt in enumerate(env.prompts):
        probs = actor.predict_probs(i)
        print(f"Prompt: {prompt}")
        for resp, p in zip(CANDIDATES, probs):
            print(f"  π(a)={p:0.3f} -> {resp}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty weight")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data"),
        help="Directory with SFT/RM checkpoints",
    )
    args = parser.parse_args()
    run(beta=args.beta, output_dir=args.output_dir)

