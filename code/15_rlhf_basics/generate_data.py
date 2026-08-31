"""Generate tiny SFT pairs and preference data for the RLHF demo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from simple_text_env import CANDIDATES, PROMPTS, SimpleTextEnv


def build_sft_data() -> List[Dict]:
    # Map each prompt to the most aligned candidate (index 1).
    best_resp = CANDIDATES[1]
    data = [{"prompt": prompt, "response": best_resp} for prompt in PROMPTS]
    return data


def build_preferences(env: SimpleTextEnv, num_samples: int = 80) -> List[Dict]:
    prefs: List[Dict] = []
    for _ in range(num_samples):
        batch = env.sample_prompts(batch_size=1)
        prompt = batch.prompts[0]
        # Sample two distinct candidates.
        a, b = env.rng.sample(CANDIDATES, 2)
        ra, rb = env.hidden_reward(prompt, a), env.hidden_reward(prompt, b)
        if ra == rb:
            continue
        chosen, rejected = (a, b) if ra > rb else (b, a)
        prefs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return prefs


def main(output_dir: Path = Path("data"), num_samples: int = 80) -> None:
    env = SimpleTextEnv(seed=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_data = build_sft_data()
    prefs = build_preferences(env, num_samples=num_samples)
    (output_dir / "sft_data.json").write_text(json.dumps(sft_data, indent=2))
    (output_dir / "preferences.json").write_text(json.dumps(prefs, indent=2))
    print(f"Wrote {len(sft_data)} SFT pairs and {len(prefs)} preferences to {output_dir}")


if __name__ == "__main__":
    main()


