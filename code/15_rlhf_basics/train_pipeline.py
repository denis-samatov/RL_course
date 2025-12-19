"""Run full toy RLHF pipeline: data gen -> SFT -> RM -> PPO.

The code is intentionally tiny to make it easy to live-code edits during
the seminar. All checkpoints land in `data/`.
"""

from __future__ import annotations

from pathlib import Path

from generate_data import main as generate_data
from reward_model import train_reward_model
from sft_model import train_sft
from ppo_rlhf import run as run_ppo


def main() -> None:
    base = Path(__file__).parent
    data_dir = base / "data"

    # 1) Synthetic data
    generate_data(output_dir=data_dir)

    # 2) SFT
    sft_ckpt = data_dir / "sft_policy.npy"
    train_sft(sft_path=data_dir / "sft_data.json", ckpt_path=sft_ckpt)

    # 3) Reward model
    rm_ckpt = data_dir / "reward_model.npy"
    train_reward_model(pref_path=data_dir / "preferences.json", ckpt_path=rm_ckpt)

    # 4) PPO with default beta
    run_ppo(beta=0.01, output_dir=data_dir)


if __name__ == "__main__":
    main()

