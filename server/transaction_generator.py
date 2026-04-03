"""
Fraud Detection OpenEnv – Transaction Generator
Hybrid probabilistic + rule-based generation of realistic bank transactions.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Risk lookup tables
# ---------------------------------------------------------------------------

# Country risk tiers: 0.0 = safe, 1.0 = very high risk
GEO_RISK: Dict[str, float] = {
    "US": 0.10, "CA": 0.10, "GB": 0.12, "DE": 0.12, "FR": 0.13,
    "AU": 0.11, "JP": 0.09, "SG": 0.14, "NL": 0.13, "CH": 0.11,
    "IN": 0.22, "BR": 0.35, "MX": 0.38, "ZA": 0.42, "NG": 0.72,
    "RU": 0.75, "UA": 0.55, "RO": 0.48, "CN": 0.30, "PK": 0.60,
    "VN": 0.40, "PH": 0.38, "ID": 0.35, "TR": 0.35, "AR": 0.45,
    "KP": 0.95, "IR": 0.88, "SY": 0.85, "VE": 0.70, "BY": 0.65,
}

HIGH_RISK_COUNTRIES = [c for c, r in GEO_RISK.items() if r >= 0.45]
MED_RISK_COUNTRIES  = [c for c, r in GEO_RISK.items() if 0.20 <= r < 0.45]
LOW_RISK_COUNTRIES  = [c for c, r in GEO_RISK.items() if r < 0.20]

# Merchant type risk scores
MERCHANT_RISK: Dict[str, float] = {
    "retail":          0.05,
    "grocery":         0.03,
    "restaurant":      0.04,
    "travel":          0.18,
    "electronics":     0.22,
    "pharmacy":        0.08,
    "utilities":       0.05,
    "subscription":    0.12,
    "wire_transfer":   0.55,
    "crypto_exchange": 0.80,
    "gambling":        0.75,
    "gift_card":       0.65,
    "atm_withdrawal":  0.35,
    "money_order":     0.60,
    "pawn_shop":       0.50,
}

MERCHANT_TYPES = list(MERCHANT_RISK.keys())
HIGH_RISK_MERCHANTS = [m for m, r in MERCHANT_RISK.items() if r >= 0.50]
MED_RISK_MERCHANTS  = [m for m, r in MERCHANT_RISK.items() if 0.15 <= r < 0.50]
LOW_RISK_MERCHANTS  = [m for m, r in MERCHANT_RISK.items() if r < 0.15]

DEVICE_TYPES = ["mobile", "desktop", "tablet", "unknown"]


# ---------------------------------------------------------------------------
# Synthetic Transaction
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """A fully synthesized transaction with ground-truth fraud label."""

    transaction_id: str
    amount: float
    country: str
    merchant_type: str
    device_type: str
    user_age: int
    transaction_velocity: int
    is_night: bool
    account_age_days: int

    # Derived risk signals
    amount_zscore: float
    geo_risk_score: float
    merchant_risk_score: float
    device_consistency: float

    # Ground truth (never exposed to agent in observation directly)
    is_fraud: bool
    fraud_score: float          # Raw composite score [0, 1]
    fraud_signals: List[str]    # List of triggered signals for debugging


# ---------------------------------------------------------------------------
# User Profile (tracks behavioral norms for velocity / device consistency)
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    """Simulates a customer's historical transaction profile."""
    user_id: str
    avg_amount: float
    std_amount: float
    home_country: str
    usual_device: str
    usual_merchants: List[str]
    account_age_days: int
    user_age: int


# ---------------------------------------------------------------------------
# Transaction Generator
# ---------------------------------------------------------------------------

class TransactionGenerator:
    """
    Generates realistic synthetic transactions using a hybrid approach:
      1. Build user profiles with behavioral norms
      2. Select fraud / legit based on task difficulty + target rate
      3. Apply probabilistic noise + rule-based overrides
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Pool of synthetic users
        self._users: List[UserProfile] = self._init_users(n=200)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        n: int,
        fraud_rate: float = 0.30,
        difficulty: str = "medium",
    ) -> List[Transaction]:
        """Generate `n` transactions with target fraud prevalence."""
        transactions = []
        for _ in range(n):
            is_fraud = self.rng.random() < fraud_rate
            tx = self._generate_one(is_fraud=is_fraud, difficulty=difficulty)
            transactions.append(tx)
        return transactions

    def generate_one(
        self,
        is_fraud: Optional[bool] = None,
        difficulty: str = "medium",
    ) -> Transaction:
        if is_fraud is None:
            is_fraud = self.rng.random() < 0.30
        return self._generate_one(is_fraud=is_fraud, difficulty=difficulty)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _generate_one(self, is_fraud: bool, difficulty: str) -> Transaction:
        user = self.rng.choice(self._users)
        signals: List[str] = []

        if is_fraud:
            return self._build_fraud_tx(user, difficulty, signals)
        else:
            return self._build_legit_tx(user, difficulty, signals)

    def _build_legit_tx(
        self,
        user: UserProfile,
        difficulty: str,
        signals: List[str],
    ) -> Transaction:
        """Legitimate transaction – follows user behavioral norms."""

        # Amount: normally distributed around user average
        amount = max(1.0, self.np_rng.normal(user.avg_amount, user.std_amount))
        amount = round(float(amount), 2)
        amount_zscore = (amount - user.avg_amount) / max(user.std_amount, 1.0)

        country = user.home_country
        merchant_type = self.rng.choice(user.usual_merchants)
        device_type = user.usual_device
        device_consistency = 1.0
        is_night = self.rng.random() < 0.12
        velocity = max(1, int(self.np_rng.normal(3, 2)))

        # For medium/hard tasks, occasionally add slight anomalies on legit
        if difficulty in ("medium", "hard") and self.rng.random() < 0.15:
            # Slightly high amount but still legit
            amount = round(amount * self.rng.uniform(1.5, 2.5), 2)
            amount_zscore = (amount - user.avg_amount) / max(user.std_amount, 1.0)
            signals.append("slightly_elevated_amount")

        fraud_score = self._compute_fraud_score(
            amount, amount_zscore, country, merchant_type,
            device_consistency, is_night, velocity, user
        )

        # Ensure legit score stays low
        fraud_score = min(fraud_score, 0.35)

        return Transaction(
            transaction_id=str(uuid.uuid4())[:8],
            amount=amount,
            country=country,
            merchant_type=merchant_type,
            device_type=device_type,
            user_age=user.user_age,
            transaction_velocity=velocity,
            is_night=is_night,
            account_age_days=user.account_age_days,
            amount_zscore=round(amount_zscore, 3),
            geo_risk_score=GEO_RISK.get(country, 0.3),
            merchant_risk_score=MERCHANT_RISK.get(merchant_type, 0.2),
            device_consistency=device_consistency,
            is_fraud=False,
            fraud_score=round(fraud_score, 3),
            fraud_signals=signals,
        )

    def _build_fraud_tx(
        self,
        user: UserProfile,
        difficulty: str,
        signals: List[str],
    ) -> Transaction:
        """
        Fraudulent transaction – selected pattern based on difficulty:
          easy:   High-confidence single signal (obvious fraud)
          medium: 2-3 moderate signals requiring reasoning
          hard:   Subtle, multi-feature adversarial patterns
        """

        if difficulty == "easy":
            return self._fraud_easy(user, signals)
        elif difficulty == "medium":
            return self._fraud_medium(user, signals)
        else:
            return self._fraud_hard(user, signals)

    # ------------------------------------------------------------------
    # Fraud patterns by difficulty
    # ------------------------------------------------------------------

    def _fraud_easy(self, user: UserProfile, signals: List[str]) -> Transaction:
        """
        Obvious fraud: one or two very strong signals.
        Examples: crypto at 3am from Nigeria, 10x normal amount + new device
        """
        pattern = self.rng.choice(["high_value_foreign", "crypto_night", "velocity_spike"])

        if pattern == "high_value_foreign":
            amount = round(user.avg_amount * self.rng.uniform(8, 20), 2)
            country = self.rng.choice(HIGH_RISK_COUNTRIES)
            merchant_type = self.rng.choice(LOW_RISK_MERCHANTS + MED_RISK_MERCHANTS)
            device_type = "unknown"
            device_consistency = 0.0
            is_night = self.rng.random() < 0.60
            velocity = self.rng.randint(1, 4)
            signals += ["high_value_anomaly", "foreign_high_risk_country", "new_device"]

        elif pattern == "crypto_night":
            amount = round(self.rng.uniform(500, 5000), 2)
            country = self.rng.choice(HIGH_RISK_COUNTRIES + MED_RISK_COUNTRIES)
            merchant_type = "crypto_exchange"
            device_type = self.rng.choice(["mobile", "unknown"])
            device_consistency = 0.0
            is_night = True
            velocity = self.rng.randint(1, 5)
            signals += ["crypto_merchant", "night_transaction", "geo_risk_elevated"]

        else:  # velocity_spike
            amount = round(user.avg_amount * self.rng.uniform(1.2, 3.0), 2)
            country = user.home_country
            merchant_type = "atm_withdrawal"
            device_type = user.usual_device
            device_consistency = 1.0
            is_night = self.rng.random() < 0.30
            velocity = self.rng.randint(18, 35)
            signals += ["velocity_spike", "atm_withdrawal_pattern"]

        amount_zscore = (amount - user.avg_amount) / max(user.std_amount, 1.0)
        fraud_score = self._compute_fraud_score(
            amount, amount_zscore, country, merchant_type,
            device_consistency, is_night, velocity, user
        )
        fraud_score = max(fraud_score, 0.75)  # ensure obviously high

        return Transaction(
            transaction_id=str(uuid.uuid4())[:8],
            amount=amount,
            country=country,
            merchant_type=merchant_type,
            device_type=device_type,
            user_age=user.user_age,
            transaction_velocity=velocity,
            is_night=is_night,
            account_age_days=user.account_age_days,
            amount_zscore=round(amount_zscore, 3),
            geo_risk_score=GEO_RISK.get(country, 0.5),
            merchant_risk_score=MERCHANT_RISK.get(merchant_type, 0.5),
            device_consistency=device_consistency,
            is_fraud=True,
            fraud_score=round(min(fraud_score, 0.99), 3),
            fraud_signals=signals,
        )

    def _fraud_medium(self, user: UserProfile, signals: List[str]) -> Transaction:
        """
        Medium fraud: 2–3 moderate signals – requires multi-feature reasoning.
        """
        pattern = self.rng.choice(["gift_card_chain", "account_takeover", "foreign_gambling"])

        if pattern == "gift_card_chain":
            amount = round(user.avg_amount * self.rng.uniform(2, 5), 2)
            country = self.rng.choice(MED_RISK_COUNTRIES + LOW_RISK_COUNTRIES[:3])
            merchant_type = "gift_card"
            device_type = self.rng.choice(["mobile", "desktop"])
            device_consistency = self.rng.choice([0.0, 1.0])
            is_night = self.rng.random() < 0.40
            velocity = self.rng.randint(6, 14)
            signals += ["gift_card_merchant", "elevated_velocity", "elevated_amount"]

        elif pattern == "account_takeover":
            amount = round(user.avg_amount * self.rng.uniform(1.0, 3.0), 2)
            country = self.rng.choice(MED_RISK_COUNTRIES)
            merchant_type = self.rng.choice(["electronics", "wire_transfer"])
            device_type = "unknown"
            device_consistency = 0.0
            is_night = self.rng.random() < 0.45
            velocity = self.rng.randint(4, 10)
            signals += ["new_device", "geo_deviation", "suspicious_merchant"]

        else:  # foreign_gambling
            amount = round(self.rng.uniform(200, 2000), 2)
            country = self.rng.choice(MED_RISK_COUNTRIES)
            merchant_type = "gambling"
            device_type = self.rng.choice(DEVICE_TYPES)
            device_consistency = self.rng.uniform(0.0, 1.0)
            is_night = self.rng.random() < 0.50
            velocity = self.rng.randint(3, 10)
            signals += ["gambling_merchant", "foreign_country", "night_possibility"]

        amount_zscore = (amount - user.avg_amount) / max(user.std_amount, 1.0)
        fraud_score = self._compute_fraud_score(
            amount, amount_zscore, country, merchant_type,
            device_consistency, is_night, velocity, user
        )
        fraud_score = max(fraud_score, 0.50)
        fraud_score = min(fraud_score, 0.85)

        return Transaction(
            transaction_id=str(uuid.uuid4())[:8],
            amount=amount,
            country=country,
            merchant_type=merchant_type,
            device_type=device_type,
            user_age=user.user_age,
            transaction_velocity=velocity,
            is_night=is_night,
            account_age_days=user.account_age_days,
            amount_zscore=round(amount_zscore, 3),
            geo_risk_score=GEO_RISK.get(country, 0.3),
            merchant_risk_score=MERCHANT_RISK.get(merchant_type, 0.4),
            device_consistency=round(device_consistency, 2),
            is_fraud=True,
            fraud_score=round(fraud_score, 3),
            fraud_signals=signals,
        )

    def _fraud_hard(self, user: UserProfile, signals: List[str]) -> Transaction:
        """
        Hard fraud: subtle, adversarial patterns designed to evade simple rules.
        Low-amount crypto, domestic geo with device anomaly, slightly high velocity.
        """
        pattern = self.rng.choice(["low_slow_fraud", "domestic_anomaly", "behavioral_drift"])

        if pattern == "low_slow_fraud":
            # Small amounts that bypass high-threshold rules
            amount = round(self.rng.uniform(50, 300), 2)
            country = self.rng.choice(LOW_RISK_COUNTRIES)
            merchant_type = self.rng.choice(["gift_card", "money_order", "pawn_shop"])
            device_type = user.usual_device
            device_consistency = 1.0
            is_night = self.rng.random() < 0.20
            velocity = self.rng.randint(4, 9)
            signals += ["low_amount_probe", "high_risk_merchant_subtle"]

        elif pattern == "domestic_anomaly":
            amount = round(user.avg_amount * self.rng.uniform(1.5, 4.0), 2)
            country = user.home_country
            merchant_type = self.rng.choice(["electronics", "wire_transfer"])
            device_type = self.rng.choice(["tablet", "unknown"])
            device_consistency = 0.0
            is_night = self.rng.random() < 0.35
            velocity = self.rng.randint(5, 12)
            signals += ["domestic_device_anomaly", "elevated_amount", "unusual_merchant"]

        else:  # behavioral_drift
            amount = round(user.avg_amount * self.rng.uniform(2.0, 6.0), 2)
            country = self.rng.choice(MED_RISK_COUNTRIES[:5])
            merchant_type = self.rng.choice(MED_RISK_MERCHANTS)
            device_type = user.usual_device
            device_consistency = self.rng.uniform(0.3, 0.8)
            is_night = self.rng.random() < 0.25
            velocity = self.rng.randint(7, 16)
            signals += ["behavioral_drift", "geo_shift", "velocity_creep"]

        amount_zscore = (amount - user.avg_amount) / max(user.std_amount, 1.0)
        fraud_score = self._compute_fraud_score(
            amount, amount_zscore, country, merchant_type,
            device_consistency, is_night, velocity, user
        )
        fraud_score = max(fraud_score, 0.45)
        fraud_score = min(fraud_score, 0.78)  # intentionally ambiguous

        return Transaction(
            transaction_id=str(uuid.uuid4())[:8],
            amount=amount,
            country=country,
            merchant_type=merchant_type,
            device_type=device_type,
            user_age=user.user_age,
            transaction_velocity=velocity,
            is_night=is_night,
            account_age_days=user.account_age_days,
            amount_zscore=round(amount_zscore, 3),
            geo_risk_score=GEO_RISK.get(country, 0.3),
            merchant_risk_score=MERCHANT_RISK.get(merchant_type, 0.3),
            device_consistency=round(device_consistency, 2),
            is_fraud=True,
            fraud_score=round(fraud_score, 3),
            fraud_signals=signals,
        )

    # ------------------------------------------------------------------
    # Fraud scoring engine (used internally for consistency)
    # ------------------------------------------------------------------

    def _compute_fraud_score(
        self,
        amount: float,
        amount_zscore: float,
        country: str,
        merchant_type: str,
        device_consistency: float,
        is_night: bool,
        velocity: int,
        user: UserProfile,
    ) -> float:
        score = 0.0

        # Amount anomaly
        if amount_zscore > 3.0:
            score += 0.30
        elif amount_zscore > 2.0:
            score += 0.18
        elif amount_zscore > 1.0:
            score += 0.08

        # Geo risk
        score += GEO_RISK.get(country, 0.3) * 0.35

        # Merchant risk
        score += MERCHANT_RISK.get(merchant_type, 0.2) * 0.30

        # Device consistency
        score += (1.0 - device_consistency) * 0.20

        # Night flag
        if is_night:
            score += 0.07

        # Velocity anomaly
        if velocity > 20:
            score += 0.20
        elif velocity > 10:
            score += 0.10
        elif velocity > 6:
            score += 0.04

        # New account vulnerability
        if user.account_age_days < 30:
            score += 0.12

        return min(score, 0.99)

    # ------------------------------------------------------------------
    # User profile builder
    # ------------------------------------------------------------------

    def _init_users(self, n: int) -> List[UserProfile]:
        users = []
        for i in range(n):
            # Bimodal: most users are low/mid income, some are high-value
            if self.rng.random() < 0.20:
                avg_amount = self.rng.uniform(1000, 8000)
                std_amount = avg_amount * 0.4
            else:
                avg_amount = self.rng.uniform(30, 600)
                std_amount = avg_amount * 0.5

            home_country = self.rng.choice(
                LOW_RISK_COUNTRIES * 5 + MED_RISK_COUNTRIES * 2 + HIGH_RISK_COUNTRIES
            )
            usual_merchants = self.rng.sample(LOW_RISK_MERCHANTS + MED_RISK_MERCHANTS, k=3)
            usual_device = self.rng.choice(["mobile", "mobile", "desktop", "tablet"])
            account_age_days = int(self.np_rng.exponential(730))  # most accounts older
            account_age_days = max(7, min(account_age_days, 3650))
            user_age = self.rng.randint(18, 75)

            users.append(UserProfile(
                user_id=f"user_{i:04d}",
                avg_amount=round(avg_amount, 2),
                std_amount=round(std_amount, 2),
                home_country=home_country,
                usual_device=usual_device,
                usual_merchants=usual_merchants,
                account_age_days=account_age_days,
                user_age=user_age,
            ))
        return users
