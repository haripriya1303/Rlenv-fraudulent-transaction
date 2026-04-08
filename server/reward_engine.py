"""
Fraud Detection OpenEnv – Reward Engine
Dense, step-wise reward shaping that models real operational trade-offs:
  - Fraud prevention (catching bad transactions)
  - Customer experience (not blocking legit transactions)
  - Efficiency (not over-blocking)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


import math

# ---------------------------------------------------------------------------
# Base reward table
# ---------------------------------------------------------------------------

BASE_REWARDS: Dict[Tuple[str, bool], float] = {
    # (decision, is_fraud) -> base_reward
    ("BLOCK",   True):  +2.00,   # Correctly stopped fraud
    ("FLAG",    True):  +1.00,   # Good flag – partial catch
    ("APPROVE", True):  -3.00,   # Catastrophic: Fraud APPROVED
    ("BLOCK",   False): -2.50,   # Severe: Inconvenienced good customer (Churn)
    ("FLAG",    False): -0.60,   # Small annoyance for legit user
    ("APPROVE", False): +0.80,   # Smooth legitimate experience
}


@dataclass
class RewardBreakdown:
    """Structured reward output for explainability and logging."""
    base_reward: float
    over_block_penalty: float
    confidence_bonus: float
    total_reward: float
    correctness: str   # "correct" | "partial" | "incorrect"
    label: str         # human-readable explanation


class RewardEngine:
    """
    Winning Reward Engine:
    1. Strong learning signals ([-3.0, +2.0])
    2. Exploit prevention for 'Always Block' or 'Always Flag'
    3. Balanced trade-offs between UX and Fraud Loss
    """

    def __init__(
        self,
        block_threshold: float = 0.25,      # Normal bank blocks < 25% of flagged tx
        penalty_scale: float = -4.0,       # Very aggressive penalty if lazy
    ):
        self.block_threshold = block_threshold
        self.penalty_scale = penalty_scale

    def compute(
        self,
        decision: str,
        is_fraud: bool,
        block_rate_so_far: float,
        confidence: float = 1.0,
    ) -> RewardBreakdown:
        base = BASE_REWARDS.get((decision, is_fraud), 0.0)

        # Anti-Exploit: If the agent blocks too much, they are likely just guessing
        # to avoid the massive -3.0 penalty of missing fraud.
        over_block_penalty = 0.0
        if block_rate_so_far > self.block_threshold:
            excess = block_rate_so_far - self.block_threshold
            over_block_penalty = self.penalty_scale * (excess ** 2) # Exponential penalty

        # Confidence modulation: Higher confidence boosts correct rewards and increases penalties for mistakes
        confidence_bonus = base * (confidence * 0.2)
        raw_total = base + over_block_penalty + confidence_bonus
        
        # 🛡️ Sigmoid Normalization [0, 1] for Hackathon Compliance
        # Ensures rewards are strictly in (0, 1) to avoid validator boundary errors
        total = 1.0 / (1.0 + math.exp(-raw_total))
        total = round(max(0.001, min(0.999, total)), 4)

        correctness = self._classify_correctness(decision, is_fraud)
        label = self._human_label(decision, is_fraud, over_block_penalty < -0.1)

        return RewardBreakdown(
            base_reward=round(base, 4),
            over_block_penalty=round(over_block_penalty, 4),
            confidence_bonus=round(confidence_bonus, 4),
            total_reward=total,
            correctness=correctness,
            label=label,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_correctness(decision: str, is_fraud: bool) -> str:
        if decision == "BLOCK" and is_fraud:
            return "correct"
        if decision == "FLAG" and is_fraud:
            return "partial"
        if decision == "APPROVE" and not is_fraud:
            return "correct"
        if decision == "FLAG" and not is_fraud:
            return "partial"   # partial credit: safer than blocking legit
        return "incorrect"

    @staticmethod
    def _human_label(decision: str, is_fraud: bool, over_blocked: bool) -> str:
        labels = {
            ("BLOCK",  True,  False): "✅ Fraud correctly blocked",
            ("BLOCK",  True,  True):  "✅ Fraud blocked (but over-blocking penalty applied)",
            ("FLAG",   True,  False): "⚠️  Fraud flagged for review",
            ("FLAG",   True,  True):  "⚠️  Fraud flagged (over-blocking penalty)",
            ("APPROVE",True,  False): "❌ FRAUD MISSED – approved",
            ("APPROVE",True,  True):  "❌ FRAUD MISSED – approved",
            ("BLOCK",  False, False): "❌ Legitimate transaction wrongly blocked",
            ("BLOCK",  False, True):  "❌ Legitimate transaction blocked (severe penalty)",
            ("FLAG",   False, False): "⚠️  Legitimate transaction flagged",
            ("FLAG",   False, True):  "⚠️  Legitimate transaction flagged (over-blocking)",
            ("APPROVE",False, False): "✅ Legitimate transaction correctly approved",
            ("APPROVE",False, True):  "✅ Correct approval (over-blocking still high)",
        }
        return labels.get((decision, is_fraud, over_blocked), "Unknown")
