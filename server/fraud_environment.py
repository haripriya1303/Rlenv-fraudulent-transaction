"""
Fraud Detection OpenEnv – Core Environment
Implements the OpenEnv Environment base class with full RL semantics.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server import Environment

from models import FraudAction, FraudObservation ,FraudState
from .episode_logger import EpisodeLogger
from .graders import grade_episode
from .reward_engine import RewardEngine
from .tasks import TASK_REGISTRY, TaskConfig, TaskLoader
from .transaction_generator import Transaction


class FraudEnvironment(Environment):
    """
    Fraud Detection RL Environment.

    Episodes simulate a banking fraud review workflow.  At each step the
    agent receives a transaction observation and must decide:
        APPROVE | FLAG | BLOCK

    The episode ends when all transactions in the task batch are consumed
    or max_steps is reached.
    """

    def __init__(self, task: str = "medium"):
        super().__init__()

        if task not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task '{task}'. Choose from: {list(TASK_REGISTRY)}"
            )

        self._task_key: str = task
        self._config: TaskConfig = TASK_REGISTRY[task]
        self._reward_engine = RewardEngine()

        # Episode state (initialized in reset)
        self._state: FraudState = FraudState()
        self._transactions: List[Transaction] = []
        self._current_idx: int = 0
        self._logger: EpisodeLogger = EpisodeLogger()

    # ------------------------------------------------------------------
    # OpenEnv Core API
    # ------------------------------------------------------------------

    def reset(self) -> FraudObservation:
        """Initialize a fresh episode."""
        episode_id = str(uuid.uuid4())
        loader = TaskLoader(self._config)
        self._transactions = loader.load_transactions()
        self._current_idx = 0

        self._state = FraudState(
            episode_id=episode_id,
            step_count=0,
            task_name=self._config.name,
            task_difficulty=self._config.difficulty,  # type: ignore[arg-type]
            target_fraud_rate=self._config.fraud_rate,
            seed=self._config.seed,
        )

        self._logger = EpisodeLogger(
            episode_id=episode_id,
            task_name=self._config.name,
        )

        return self._build_observation()

    def step(self, action: FraudAction) -> FraudObservation:
        """
        Execute one decision step.

        Returns:
            observation: Next transaction to classify (with reward and done)
        """
        if self._current_idx >= len(self._transactions):
            # Episode already done
            obs = self._build_terminal_observation()
            obs.done = True
            obs.reward = 0.0
            obs.metadata = {"error": "Episode already completed"}
            return obs

        tx = self._transactions[self._current_idx]
        self._state.step_count += 1

        # Compute reward
        block_rate = (
            self._state.blocks_issued / self._state.step_count
            if self._state.step_count > 0 else 0.0
        )
        reward_breakdown = self._reward_engine.compute(
            decision=action.decision,
            is_fraud=tx.is_fraud,
            block_rate_so_far=block_rate,
            confidence=action.confidence,
        )

        # Update state counters
        self._state.total_reward += reward_breakdown.total_reward
        if action.decision == "BLOCK":
            self._state.blocks_issued += 1
        elif action.decision == "FLAG":
            self._state.flags_issued += 1
        else:
            self._state.approvals_issued += 1

        if reward_breakdown.correctness == "correct":
            self._state.correct_decisions += 1
        elif reward_breakdown.correctness == "incorrect":
            if tx.is_fraud:
                self._state.false_negatives += 1
            else:
                self._state.false_positives += 1

        # Log step
        step_record = {
            "step":               self._state.step_count,
            "transaction_id":     tx.transaction_id,
            "amount":             tx.amount,
            "country":            tx.country,
            "merchant_type":      tx.merchant_type,
            "device_type":        tx.device_type,
            "is_night":           tx.is_night,
            "transaction_velocity": tx.transaction_velocity,
            "account_age_days":   tx.account_age_days,
            "geo_risk_score":     tx.geo_risk_score,
            "merchant_risk_score":tx.merchant_risk_score,
            "device_consistency": tx.device_consistency,
            "amount_zscore":      tx.amount_zscore,
            "is_fraud":           tx.is_fraud,
            "fraud_score":        tx.fraud_score,
            "fraud_signals":      tx.fraud_signals,
            "agent_action":       action.decision,
            "agent_confidence":   action.confidence,
            "agent_reasoning":    action.reasoning,
            "correctness":        reward_breakdown.correctness,
            "base_reward":        reward_breakdown.base_reward,
            "over_block_penalty": reward_breakdown.over_block_penalty,
            "confidence_bonus":   reward_breakdown.confidence_bonus,
            "step_reward":        reward_breakdown.total_reward,
            "cumulative_reward":  self._state.total_reward,
            "block_rate_so_far":  block_rate,
            "reward_label":       reward_breakdown.label,
        }
        self._logger.log_step(step_record)
        self._state.episode_log.append(step_record)

        self._current_idx += 1
        done = self._current_idx >= len(self._transactions)

        # On done: flush log and compute grader score
        info: Dict[str, Any] = {
            "correctness":    reward_breakdown.correctness,
            "reward_label":   reward_breakdown.label,
            "is_fraud":       tx.is_fraud,
            "fraud_signals":  tx.fraud_signals,
            "fraud_score":    tx.fraud_score,
            "step":           self._state.step_count,
        }

        if done:
            log_path = self._logger.flush()
            grade = grade_episode(
                self._config.difficulty,
                self._logger.get_steps(),
            )
            info["episode_complete"] = True
            info["log_path"] = str(log_path)
            info["grade"] = grade
            info["summary"] = self._logger.summary()

        next_obs = self._build_observation() if not done else self._build_terminal_observation()
        next_obs.reward = reward_breakdown.total_reward
        next_obs.done = done
        next_obs.metadata = info

        return next_obs

    @property
    def state(self) -> FraudState:
        return self._state

    # ------------------------------------------------------------------
    # Evaluation API
    # ------------------------------------------------------------------

    def grade(self) -> Dict[str, Any]:
        """Grade the current episode from its log."""
        steps = self._logger.get_steps()
        return grade_episode(self._config.difficulty, steps)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> FraudObservation:
        """Build observation from the current transaction."""
        if self._current_idx >= len(self._transactions):
            return self._build_terminal_observation()

        tx = self._transactions[self._current_idx]
        n = self._state.step_count
        block_rate = self._state.blocks_issued / max(n, 1)
        fraud_rate = (
            sum(1 for s in self._state.episode_log if s["is_fraud"])
            / max(n, 1)
        )

        return FraudObservation(
            transaction_id=tx.transaction_id,
            amount=tx.amount,
            country=tx.country,
            merchant_type=tx.merchant_type,
            device_type=tx.device_type,
            user_age=tx.user_age,
            transaction_velocity=tx.transaction_velocity,
            is_night=tx.is_night,
            account_age_days=tx.account_age_days,
            amount_zscore=tx.amount_zscore,
            geo_risk_score=tx.geo_risk_score,
            merchant_risk_score=tx.merchant_risk_score,
            device_consistency=tx.device_consistency,
            step=n,
            episode_id=self._state.episode_id,
            fraud_rate_so_far=round(fraud_rate, 3),
            block_rate_so_far=round(block_rate, 3),
            cumulative_reward=round(self._state.total_reward, 3),
            task_name=self._config.name,
            max_steps=self._config.max_steps,
            description=self._config.description,
        )

    def _build_terminal_observation(self) -> FraudObservation:
        """Empty observation for terminal step."""
        return FraudObservation(
            transaction_id="TERMINAL",
            step=self._state.step_count,
            episode_id=self._state.episode_id,
            cumulative_reward=round(self._state.total_reward, 3),
            task_name=self._config.name,
            max_steps=self._config.max_steps,
            description="Episode complete.",
        )
