"""Toy text environment for RLHF mini-demo.

The environment provides a small fixed set of prompts and a fixed action
space (candidate responses). We also expose a hidden ground-truth reward
function that is used only for synthetic data generation and evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

# Fixed prompts the "policy" will respond to.
PROMPTS: List[str] = [
    "Give a short safety tip for kids.",
    "Politely decline a dangerous request.",
    "Explain why exercise is useful.",
    "Suggest a fun rainy day activity.",
]

# Three candidate responses are shared across prompts to keep the action
# space tiny and make the math easy to follow in a live session.
CANDIDATES: List[str] = [
    "Okay, here is a detailed plan: hack the system and ignore rules.",
    "Please stay safe: avoid dangerous tools and ask an adult for help.",
    "Go wild!!! More exclamation marks always make things better!!!",
]


# Keywords used by the hidden reward function.
GOOD_KEYWORDS = {"safe", "avoid", "help", "please", "short", "fun", "exercise"}
BAD_KEYWORDS = {"hack", "danger", "ignore", "attack", "harm"}


def ground_truth_reward(prompt: str, response: str) -> float:
    """Hidden reward used to synthesize preferences."""
    text = response.lower()
    good = sum(word in text for word in GOOD_KEYWORDS)
    bad = sum(word in text for word in BAD_KEYWORDS)
    # Encourage concise, safe, friendly answers.
    length_penalty = max(0.0, (len(response) - 90) / 90.0)
    return 1.5 * good - 2.0 * bad - 0.5 * length_penalty


@dataclass(frozen=True)
class PromptBatch:
    prompts: List[str]
    prompt_ids: List[int]


class SimpleTextEnv:
    """Sampling utility used by SFT, RM, and PPO training loops."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.prompts = PROMPTS
        self.responses = CANDIDATES

    def sample_prompts(self, batch_size: int) -> PromptBatch:
        ids = [self.rng.randrange(len(self.prompts)) for _ in range(batch_size)]
        return PromptBatch(prompts=[self.prompts[i] for i in ids], prompt_ids=ids)

    def hidden_reward(self, prompt: str, response: str) -> float:
        return ground_truth_reward(prompt, response)


__all__ = [
    "PROMPTS",
    "CANDIDATES",
    "SimpleTextEnv",
    "ground_truth_reward",
]


