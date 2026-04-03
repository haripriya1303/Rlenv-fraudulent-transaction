"""
Fraud Detection OpenEnv – Task Definitions
Each task configures its own transaction stream, difficulty, and step budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..transaction_generator import Transaction, TransactionGenerator


@dataclass
class TaskConfig:
    """Immutable task configuration."""
    name: str
    difficulty: str       # easy | medium | hard
    description: str
    max_steps: int
    fraud_rate: float     # Target fraud prevalence in the episode
    seed: int = 42


# ---------------------------------------------------------------------------
# Canonical Task Configurations
# ---------------------------------------------------------------------------

EASY_TASK = TaskConfig(
    name="easy_fraud_detection",
    difficulty="easy",
    description=(
        "Detect obvious fraudulent transactions where single high-confidence "
        "signals are present (e.g. crypto at 3am from a high-risk country, "
        "or a 15x amount spike with a new device). Each transaction carries "
        "at least one very strong fraud signal or is clearly legitimate."
    ),
    max_steps=20,
    fraud_rate=0.40,
    seed=1001,
)

MEDIUM_TASK = TaskConfig(
    name="medium_fraud_detection",
    difficulty="medium",
    description=(
        "Classify transactions that require reasoning across multiple features "
        "simultaneously. No single signal is conclusive — the agent must weigh "
        "geo risk, merchant type, velocity, device consistency, and amount "
        "anomaly together. ~40% of transactions are fraudulent."
    ),
    max_steps=30,
    fraud_rate=0.40,
    seed=2002,
)

HARD_TASK = TaskConfig(
    name="hard_fraud_detection",
    difficulty="hard",
    description=(
        "Optimize long-horizon cumulative reward over a 50-step episode. "
        "Fraud patterns are subtle and adversarial. The agent must balance "
        "fraud detection against false positives: systematic over-blocking "
        "incurs a cascade penalty. Maximizing cumulative reward requires "
        "calibrated, nuanced decision-making at scale."
    ),
    max_steps=50,
    fraud_rate=0.35,
    seed=3003,
)

TASK_REGISTRY = {
    "easy":   EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard":   HARD_TASK,
}


class TaskLoader:
    """Generates the transaction sequence for a given task."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self._generator = TransactionGenerator(seed=config.seed)

    def load_transactions(self) -> List[Transaction]:
        """Return deterministic transaction list for this task."""
        return self._generator.generate_batch(
            n=self.config.max_steps,
            fraud_rate=self.config.fraud_rate,
            difficulty=self.config.difficulty,
        )
