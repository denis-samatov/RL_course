"""Linear reward model trained on preference pairs via Bradley-Terry loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np



def featurize(prompt: str, response: str) -> np.ndarray:
    """Very small handcrafted feature vector."""
    text = response.lower()
    prompt_len = len(prompt.split())
    resp_len = len(response.split())
    safe_words = ["safe", "avoid", "please", "help", "polite"]
    risky_words = ["hack", "attack", "danger", "ignore", "harm"]
    safe_cnt = sum(w in text for w in safe_words)
    risky_cnt = sum(w in text for w in risky_words)
    return np.array([safe_cnt, risky_cnt, resp_len, prompt_len, 1.0], dtype=np.float64)


class RewardModel:
    def __init__(self, dim: int = 5) -> None:
        self.w = np.zeros(dim, dtype=np.float64)

    def score(self, prompt: str, response: str) -> float:
        return float(np.dot(self.w, featurize(prompt, response)))

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, prefs: List[Dict], lr: float = 0.05, epochs: int = 200) -> None:
        for _ in range(epochs):
            for sample in prefs:
                pw = sample["prompt"]
                y_w = sample["chosen"]
                y_l = sample["rejected"]
                diff = self.score(pw, y_w) - self.score(pw, y_l)
                prob = self._sigmoid(diff)
                # Bradley-Terry gradient
                grad = (1.0 - prob)
                self.w += lr * grad * (
                    featurize(pw, y_w) - featurize(pw, y_l)
                )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.w)

    @classmethod
    def load(cls, path: str | Path) -> "RewardModel":
        obj = cls()
        obj.w = np.load(path)
        return obj


def train_reward_model(pref_path: str | Path, ckpt_path: str | Path) -> RewardModel:
    prefs = json.loads(Path(pref_path).read_text())
    model = RewardModel()
    model.fit(prefs)
    model.save(ckpt_path)
    return model


__all__ = ["RewardModel", "train_reward_model", "featurize"]


