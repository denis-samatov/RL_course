"""Tiny SFT model: tabular softmax over fixed candidate responses.

We keep the model intentionally simple so that every math step can be
derived on a whiteboard during the seminar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from simple_text_env import CANDIDATES, PROMPTS


class TabularPolicy:
    """Softmax logits per prompt over the shared candidate set."""

    def __init__(self) -> None:
        self.logits = np.zeros((len(PROMPTS), len(CANDIDATES)), dtype=np.float64)

    def predict_probs(self, prompt_id: int) -> np.ndarray:
        logits = self.logits[prompt_id]
        z = logits - logits.max()
        exp = np.exp(z)
        return exp / exp.sum()

    def fit(self, data: List[Dict], lr: float = 0.2, epochs: int = 200) -> None:
        """Cross-entropy training on SFT pairs."""
        for _ in range(epochs):
            for sample in data:
                pid = PROMPTS.index(sample["prompt"])
                target = CANDIDATES.index(sample["response"])
                probs = self.predict_probs(pid)
                grad = -probs
                grad[target] += 1.0
                self.logits[pid] += lr * grad

    def sample(self, prompt_id: int, rng: np.random.Generator) -> int:
        probs = self.predict_probs(prompt_id)
        return int(rng.choice(len(CANDIDATES), p=probs))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.logits)

    @classmethod
    def load(cls, path: str | Path) -> "TabularPolicy":
        obj = cls()
        obj.logits = np.load(path)
        return obj


def train_sft(sft_path: str | Path, ckpt_path: str | Path) -> TabularPolicy:
    data = json.loads(Path(sft_path).read_text())
    model = TabularPolicy()
    model.fit(data)
    model.save(ckpt_path)
    return model


__all__ = ["TabularPolicy", "train_sft"]


