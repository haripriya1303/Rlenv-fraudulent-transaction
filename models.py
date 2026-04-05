"""
Fraud Detection OpenEnv – Typed Models
All data structures shared between client and server.
"""
from __future__ import annotations

from pydantic import Field
from typing import Dict, List, Literal, Optional
import uuid

from openenv.core.env_server import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class FraudAction(Action):
    """
    Agent action: classify a transaction as APPROVE, FLAG, or BLOCK.

    Attributes:
        decision:   The primary decision for this transaction.
        confidence: Agent's self-reported confidence [0.0, 1.0].
        reasoning:  Optional free-text explanation (ignored by grader).
    """
    decision: Literal["APPROVE", "FLAG", "BLOCK"] = "APPROVE"
    confidence: float = 1.0
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.decision not in ("APPROVE", "FLAG", "BLOCK"):
            raise ValueError(f"Invalid decision: {self.decision}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class FraudObservation(Observation):
    """
    Full observable state presented to the agent per transaction step.

    Transaction Features
    --------------------
    transaction_id:         Unique ID for tracking
    amount:                 Transaction amount in USD
    country:                ISO-2 country code of transaction origin
    merchant_type:          Category of merchant (retail, crypto, gambling, …)
    device_type:            Device used (mobile, desktop, tablet, unknown)
    user_age:               Customer age in years
    transaction_velocity:   Number of transactions in the last 24 hours
    is_night:               True if transaction happened between 23:00–05:00
    account_age_days:       Age of account in days
    amount_zscore:          Standardized amount vs. user historical average
    geo_risk_score:         Country-level risk score [0.0–1.0]
    merchant_risk_score:    Merchant-category risk score [0.0–1.0]
    device_consistency:     1.0 if device matches prior pattern, 0.0 if new

    Episode Context
    ---------------
    step:                   Current step within episode
    episode_id:             UUID for episode tracking
    fraud_rate_so_far:      Fraction of transactions flagged as fraud so far
    block_rate_so_far:      Fraction of transactions blocked so far
    cumulative_reward:      Total reward accumulated so far this episode
    """

    # --- Transaction features ---
    transaction_id: str = ""
    amount: float = 0.0
    country: str = "US"
    merchant_type: str = "retail"
    device_type: str = "mobile"
    user_age: int = 30
    transaction_velocity: int = 1
    is_night: bool = False
    account_age_days: int = 365

    # --- Derived risk signals ---
    amount_zscore: float = 0.0
    geo_risk_score: float = 0.0
    merchant_risk_score: float = 0.0
    device_consistency: float = 1.0

    # --- Episode context ---
    step: int = 0
    episode_id: str = ""
    fraud_rate_so_far: float = 0.0
    block_rate_so_far: float = 0.0
    cumulative_reward: float = 0.0

    # --- Step results ---
    reward: float = 0.0
    done: bool = False

    # --- Task metadata (for agent orientation) ---
    task_name: str = ""
    max_steps: int = 0
    description: str = ""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class FraudState(State):
    """
    Full internal episode state – includes ground truth labels for logging
    and grader use.
    """
    # Inherited: episode_id, step_count

    # Episode log: list of dicts, one per step
    episode_log: List[Dict] = Field(default_factory=list)

    # Running counters
    total_reward: float = 0.0
    correct_decisions: int = 0
    false_positives: int = 0       # legitimate tx blocked/flagged
    false_negatives: int = 0       # fraud tx approved
    blocks_issued: int = 0
    flags_issued: int = 0
    approvals_issued: int = 0

    # Task configuration
    task_name: str = ""
    task_difficulty: Literal["easy", "medium", "hard"] = "easy"
    target_fraud_rate: float = 0.30

    # Seed for reproducibility
    seed: int = 42
