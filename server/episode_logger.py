"""
Fraud Detection OpenEnv – Episode Logger
Structured per-step logging.  Graders consume these logs directly.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LOG_DIR = Path(os.environ.get("LOG_DIR", "outputs/logs"))


class EpisodeLogger:
    """
    Maintains an in-memory log of every step in an episode and persists
    it to disk on episode completion.

    Log schema per step:
    {
        "step":        int,
        "transaction_id": str,
        "amount":      float,
        "country":     str,
        "merchant_type": str,
        "is_night":    bool,
        "is_fraud":    bool,           # ground truth
        "fraud_score": float,
        "fraud_signals": [str],
        "agent_action": str,           # APPROVE | FLAG | BLOCK
        "agent_confidence": float,
        "correctness": str,            # correct | partial | incorrect
        "base_reward": float,
        "over_block_penalty": float,
        "confidence_bonus": float,
        "step_reward": float,
        "cumulative_reward": float,
        "block_rate_so_far": float,
        "timestamp": str (ISO-8601)
    }
    """

    def __init__(self, episode_id: Optional[str] = None, task_name: str = ""):
        self.episode_id: str = episode_id or str(uuid.uuid4())
        self.task_name: str = task_name
        self.steps: List[Dict[str, Any]] = []
        self.started_at: str = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_step(self, record: Dict[str, Any]) -> None:
        """Append a step record.  record must contain all schema fields."""
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.steps.append(record)

    def flush(self) -> Path:
        """Write episode log to disk and return path."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        filename = LOG_DIR / f"{self.task_name}_{self.episode_id}.json"

        payload = {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "started_at": self.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "total_steps": len(self.steps),
            "steps": self.steps,
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return filename

    # ------------------------------------------------------------------
    # Summary metrics (for state() endpoint)
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        if not self.steps:
            return {}

        total_reward = sum(s["step_reward"] for s in self.steps)
        correct = sum(1 for s in self.steps if s["correctness"] == "correct")
        partial = sum(1 for s in self.steps if s["correctness"] == "partial")
        incorrect = sum(1 for s in self.steps if s["correctness"] == "incorrect")
        fraud_steps = sum(1 for s in self.steps if s["is_fraud"])
        block_count = sum(1 for s in self.steps if s["agent_action"] == "BLOCK")
        flag_count  = sum(1 for s in self.steps if s["agent_action"] == "FLAG")
        approve_count = sum(1 for s in self.steps if s["agent_action"] == "APPROVE")
        n = len(self.steps)

        return {
            "episode_id": self.episode_id,
            "task_name": self.task_name,
            "total_steps": n,
            "total_reward": round(total_reward, 4),
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "accuracy": round((correct + 0.5 * partial) / n, 4) if n else 0.0,
            "fraud_prevalence": round(fraud_steps / n, 4) if n else 0.0,
            "block_rate": round(block_count / n, 4) if n else 0.0,
            "flag_rate":  round(flag_count / n, 4) if n else 0.0,
            "approve_rate": round(approve_count / n, 4) if n else 0.0,
        }

    def get_steps(self) -> List[Dict[str, Any]]:
        return list(self.steps)

    def __len__(self) -> int:
        return len(self.steps)
