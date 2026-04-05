"""
Fraud Detection OpenEnv – EnvClient
Client-side wrapper for communicating with the environment server.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import FraudAction, FraudObservation, FraudState


class FraudEnv(EnvClient[FraudAction, FraudObservation, FraudState]):
    """
    Type-safe client for the Fraud Detection environment.

    Usage (async):
        async with FraudEnv(base_url="http://localhost:8000") as client:
            result = await client.reset()
            result = await client.step(FraudAction(decision="BLOCK"))

    Usage (sync):
        with FraudEnv(base_url="http://localhost:8000").sync() as client:
            result = client.reset()
            result = client.step(FraudAction(decision="BLOCK"))
    """

    def _step_payload(self, action: FraudAction) -> dict:
        return {
            "decision":   action.decision,
            "confidence": action.confidence,
            "reasoning":  action.reasoning,
        }

    def _parse_result(self, payload: dict) -> StepResult[FraudObservation]:
        obs_data = payload.get("observation", {})
        obs = FraudObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> FraudState:
        # Strip any extra fields the server might add
        valid_fields = FraudState.__dataclass_fields__.keys()
        filtered = {k: v for k, v in payload.items() if k in valid_fields}
        return FraudState(**filtered)
