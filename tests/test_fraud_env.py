"""
Fraud Detection OpenEnv – Test Suite
Fully deterministic tests — no external API calls required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import FraudAction, FraudObservation, FraudState
from server.episode_logger import EpisodeLogger
from server.fraud_environment import FraudEnvironment
from server.graders import EasyGrader, HardGrader, MediumGrader, grade_episode
from server.reward_engine import RewardEngine
from server.tasks import TASK_REGISTRY, TaskLoader
from server.transaction_generator import TransactionGenerator


# ===========================================================================
# Transaction Generator Tests
# ===========================================================================

class TestTransactionGenerator:
    def test_deterministic_generation(self):
        """Same seed must produce the same transactions."""
        gen1 = TransactionGenerator(seed=42)
        gen2 = TransactionGenerator(seed=42)
        t1 = gen1.generate_batch(n=10, fraud_rate=0.5, difficulty="easy")
        t2 = gen2.generate_batch(n=10, fraud_rate=0.5, difficulty="easy")
        for a, b in zip(t1, t2):
            assert a.transaction_id == b.transaction_id
            assert a.is_fraud == b.is_fraud
            assert a.amount == b.amount

    def test_fraud_rate_approximate(self):
        """Fraud rate in a large batch should approximate the target."""
        gen = TransactionGenerator(seed=99)
        txs = gen.generate_batch(n=200, fraud_rate=0.40, difficulty="medium")
        fraud_count = sum(1 for t in txs if t.is_fraud)
        actual_rate = fraud_count / 200
        assert 0.25 <= actual_rate <= 0.55, f"Fraud rate {actual_rate} out of expected range"

    def test_fraud_score_range(self):
        """Fraud score must always be in [0, 1]."""
        gen = TransactionGenerator(seed=7)
        txs = gen.generate_batch(n=50, fraud_rate=0.5, difficulty="hard")
        for t in txs:
            assert 0.0 <= t.fraud_score <= 1.0, f"Score {t.fraud_score} out of range"

    def test_easy_fraud_has_strong_signals(self):
        """Easy fraud transactions should have high fraud scores."""
        gen = TransactionGenerator(seed=1001)
        frauds = [
            gen.generate_one(is_fraud=True, difficulty="easy")
            for _ in range(20)
        ]
        avg_score = sum(t.fraud_score for t in frauds) / len(frauds)
        assert avg_score >= 0.65, f"Easy fraud avg score too low: {avg_score}"

    def test_legit_tx_has_low_score(self):
        """Legitimate transactions must have bounded fraud scores."""
        gen = TransactionGenerator(seed=42)
        legits = [
            gen.generate_one(is_fraud=False, difficulty="easy")
            for _ in range(30)
        ]
        for t in legits:
            assert t.fraud_score <= 0.36, f"Legit tx score {t.fraud_score} too high"
            assert t.is_fraud is False

    def test_transaction_fields_present(self):
        """All required fields must be present."""
        gen = TransactionGenerator(seed=5)
        tx = gen.generate_one()
        required = [
            "transaction_id", "amount", "country", "merchant_type",
            "device_type", "user_age", "transaction_velocity", "is_night",
            "account_age_days", "amount_zscore", "geo_risk_score",
            "merchant_risk_score", "device_consistency", "is_fraud",
            "fraud_score", "fraud_signals",
        ]
        for field in required:
            assert hasattr(tx, field), f"Missing field: {field}"


# ===========================================================================
# Reward Engine Tests
# ===========================================================================

class TestRewardEngine:
    def setup_method(self):
        self.engine = RewardEngine()

    def test_correct_block_fraud(self):
        r = self.engine.compute("BLOCK", True, 0.0)
        assert r.base_reward == 1.0
        assert r.correctness == "correct"

    def test_miss_fraud_approve(self):
        r = self.engine.compute("APPROVE", True, 0.0)
        assert r.base_reward == -1.0
        assert r.correctness == "incorrect"

    def test_false_block_legit(self):
        r = self.engine.compute("BLOCK", False, 0.0)
        assert r.base_reward == -0.7
        assert r.correctness == "incorrect"

    def test_correct_approve_legit(self):
        r = self.engine.compute("APPROVE", False, 0.0)
        assert r.base_reward == 0.5
        assert r.correctness == "correct"

    def test_flag_fraud_is_partial(self):
        r = self.engine.compute("FLAG", True, 0.0)
        assert r.base_reward == 0.5
        assert r.correctness == "partial"

    def test_flag_legit_is_partial(self):
        r = self.engine.compute("FLAG", False, 0.0)
        assert r.base_reward == -0.3
        assert r.correctness == "partial"

    def test_over_block_penalty_applied(self):
        """Block rate above threshold should incur penalty."""
        r_normal = self.engine.compute("BLOCK", True, 0.10)
        r_excess  = self.engine.compute("BLOCK", True, 0.50)
        assert r_excess.total_reward < r_normal.total_reward
        assert r_excess.over_block_penalty < 0

    def test_total_reward_bounded(self):
        """All reward values must be within [-2, 2]."""
        combos = [
            ("APPROVE", True, 0.0), ("APPROVE", False, 0.80),
            ("FLAG",    True, 0.50), ("FLAG",   False, 0.10),
            ("BLOCK",   True, 0.20), ("BLOCK",  False, 0.90),
        ]
        for decision, is_fraud, block_rate in combos:
            r = self.engine.compute(decision, is_fraud, block_rate)
            assert -2.0 <= r.total_reward <= 2.0


# ===========================================================================
# Environment Tests
# ===========================================================================

class TestFraudEnvironment:
    def test_reset_returns_observation(self):
        env = FraudEnvironment(task="easy")
        obs = env.reset()
        assert isinstance(obs, FraudObservation)
        assert obs.transaction_id != ""
        assert obs.max_steps == 20

    def test_step_returns_correct_types(self):
        env = FraudEnvironment(task="easy")
        env.reset()
        action = FraudAction(decision="APPROVE", confidence=0.8)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, FraudObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_episode_terminates_at_max_steps(self):
        env = FraudEnvironment(task="easy")
        env.reset()
        action = FraudAction(decision="FLAG", confidence=0.5)
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(action)
            steps += 1
            assert steps <= 25, "Episode should have terminated"
        assert steps == 20  # easy task has max_steps=20

    def test_state_tracks_counters(self):
        env = FraudEnvironment(task="easy")
        env.reset()
        a = FraudAction(decision="BLOCK", confidence=1.0)
        env.step(a)
        assert env.state.blocks_issued == 1
        assert env.state.step_count == 1

    def test_grade_returns_score_in_range(self):
        env = FraudEnvironment(task="easy")
        env.reset()
        action = FraudAction(decision="APPROVE")
        done = False
        info = {}
        while not done:
            _, _, done, info = env.step(action)
        grade = info.get("grade", {})
        score = grade.get("score", -1.0)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_deterministic_episodes(self):
        """Same task must always produce the same transaction sequence."""
        env1 = FraudEnvironment(task="medium")
        env2 = FraudEnvironment(task="medium")
        obs1 = env1.reset()
        obs2 = env2.reset()
        assert obs1.transaction_id == obs2.transaction_id
        assert obs1.amount == obs2.amount
        assert obs1.country == obs2.country

    def test_invalid_action_raises(self):
        with pytest.raises((ValueError, Exception)):
            FraudAction(decision="MAYBE")

    def test_all_tasks_run(self):
        for task in ("easy", "medium", "hard"):
            env = FraudEnvironment(task=task)
            obs = env.reset()
            assert obs.task_name != ""
            action = FraudAction(decision="FLAG")
            _, _, _, info = env.step(action)
            assert "correctness" in info


# ===========================================================================
# Grader Tests
# ===========================================================================

def _make_steps(decisions, fraud_labels, rewards=None):
    """Helper: build mock step records."""
    steps = []
    cum = 0.0
    for i, (d, f) in enumerate(zip(decisions, fraud_labels)):
        r = (rewards[i] if rewards else 0.5)
        cum += r
        correctness = (
            "correct" if (d == "BLOCK" and f) or (d == "APPROVE" and not f)
            else "partial" if d == "FLAG"
            else "incorrect"
        )
        steps.append({
            "step": i + 1,
            "agent_action": d,
            "is_fraud": f,
            "step_reward": r,
            "cumulative_reward": cum,
            "correctness": correctness,
        })
    return steps


class TestEasyGrader:
    def test_perfect_score(self):
        steps = _make_steps(
            ["BLOCK"] * 5 + ["APPROVE"] * 5,
            [True]   * 5 + [False]    * 5,
        )
        grade = EasyGrader().grade(steps)
        assert grade["score"] == 1.0

    def test_zero_score(self):
        steps = _make_steps(
            ["APPROVE"] * 5 + ["BLOCK"] * 5,
            [True]      * 5 + [False]   * 5,
        )
        grade = EasyGrader().grade(steps)
        assert grade["score"] == 0.0

    def test_partial_score(self):
        steps = _make_steps(
            ["FLAG"] * 10,
            [True]   * 5 + [False] * 5,
        )
        grade = EasyGrader().grade(steps)
        assert grade["score"] == 0.5

    def test_score_in_range(self):
        steps = _make_steps(
            ["BLOCK", "APPROVE", "FLAG", "BLOCK", "APPROVE"],
            [True, False, True, False, True],
        )
        grade = EasyGrader().grade(steps)
        assert 0.0 <= grade["score"] <= 1.0


class TestMediumGrader:
    def test_fp_penalty_applied(self):
        """High false-positive rate triggers penalty."""
        # All legit transactions blocked
        steps = _make_steps(["BLOCK"] * 10, [False] * 10)
        grade = MediumGrader().grade(steps)
        assert grade["breakdown"]["fp_penalty_applied"]
        assert grade["score"] < 0.5

    def test_no_penalty_low_fp(self):
        steps = _make_steps(
            ["APPROVE"] * 9 + ["BLOCK"],
            [False]     * 9 + [True],
        )
        grade = MediumGrader().grade(steps)
        assert not grade["breakdown"]["fp_penalty_applied"]
        assert grade["score"] >= 0.85


class TestHardGrader:
    def test_high_reward_high_score(self):
        rewards = [1.0] * 20 + [0.5] * 10
        decisions = ["BLOCK"] * 20 + ["APPROVE"] * 10
        labels = [True] * 20 + [False] * 10
        steps = _make_steps(decisions, labels, rewards)
        grade = HardGrader().grade(steps)
        assert grade["score"] >= 0.80

    def test_negative_reward_low_score(self):
        rewards = [-1.0] * 30
        decisions = ["APPROVE"] * 30
        labels = [True] * 30
        steps = _make_steps(decisions, labels, rewards)
        grade = HardGrader().grade(steps)
        # Agent that approves all fraud with catch_rate < 0.2 gets penalized further
        assert grade["score"] < 0.3

    def test_score_bounded(self):
        rewards = [0.0] * 25
        decisions = ["FLAG"] * 25
        labels = [True] * 12 + [False] * 13
        steps = _make_steps(decisions, labels, rewards)
        grade = HardGrader().grade(steps)
        assert 0.0 <= grade["score"] <= 1.0


class TestGradingConvenience:
    def test_grade_episode_easy(self):
        steps = _make_steps(["BLOCK"] * 5, [True] * 5)
        result = grade_episode("easy", steps)
        assert result["score"] == 1.0

    def test_grade_episode_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown difficulty"):
            grade_episode("impossible", [])


# ===========================================================================
# Logger Tests
# ===========================================================================

class TestEpisodeLogger:
    def test_log_and_summary(self):
        logger = EpisodeLogger(task_name="test")
        logger.log_step({
            "step": 1, "is_fraud": True, "agent_action": "BLOCK",
            "step_reward": 1.0, "cumulative_reward": 1.0,
            "correctness": "correct",
        })
        summary = logger.summary()
        assert summary["total_steps"] == 1
        assert summary["total_reward"] == 1.0

    def test_flush_creates_file(self, tmp_path, monkeypatch):
        import server.episode_logger as el
        monkeypatch.setattr(el, "LOG_DIR", tmp_path)
        logger = EpisodeLogger(episode_id="testid", task_name="test")
        logger.log_step({
            "step": 1, "is_fraud": False, "agent_action": "APPROVE",
            "step_reward": 0.5, "cumulative_reward": 0.5,
            "correctness": "correct",
        })
        path = logger.flush()
        assert path.exists()
