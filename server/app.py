"""
Fraud Detection OpenEnv – FastAPI Server
Serves the environment as an HTTP + WebSocket API using openenv-core helpers.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app

from ..models import FraudAction, FraudObservation
from .fraud_environment import FraudEnvironment

# Task is configurable via environment variable (defaults to "medium")
TASK = os.environ.get("FRAUD_TASK", "medium")

env = FraudEnvironment(task=TASK)
app: FastAPI = create_fastapi_app(env, FraudAction, FraudObservation)


# ---------------------------------------------------------------------------
# Extra health + metadata endpoints
# ---------------------------------------------------------------------------

@app.get("/info")
def get_info():
    """Return environment metadata – used by openenv validate."""
    return {
        "name": "openenv-fraud",
        "version": "1.0.0",
        "task": TASK,
        "description": (
            "Fraud Detection & Transaction Risk Review RL Environment. "
            "Agent must classify banking transactions as APPROVE, FLAG, or BLOCK."
        ),
        "action_space":      ["APPROVE", "FLAG", "BLOCK"],
        "observation_fields": [
            "transaction_id", "amount", "country", "merchant_type",
            "device_type", "user_age", "transaction_velocity", "is_night",
            "account_age_days", "amount_zscore", "geo_risk_score",
            "merchant_risk_score", "device_consistency",
            "step", "episode_id", "fraud_rate_so_far",
            "block_rate_so_far", "cumulative_reward",
        ],
    }


@app.get("/grade")
def get_grade():
    """Grade the current episode on demand."""
    return env.grade()


@app.get("/summary")
def get_summary():
    """Return summarized episode statistics."""
    return env._logger.summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
