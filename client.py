"""
Fraud Detection OpenEnv – Private MCP Client
A custom client that uses the Model Context Protocol (MCP) endpoints for absolute stability.
"""
from __future__ import annotations
import os
import requests
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()
from models import FraudAction, FraudObservation, FraudState

@dataclass
class StepResult:
    observation: FraudObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

class FraudEnv:
    """
    MCP-based HTTP client for the Fraud Detection environment.
    Designed for stability on Private/Hugging Face Space proxies.
    """
    def __init__(self, base_url: str, **kwargs):
        self.base_url = base_url.rstrip("/")
        self.hf_token = os.getenv("HF_TOKEN")
        self._session = requests.Session()
        
        if self.hf_token:
            self._session.headers.update({"Authorization": f"Bearer {self.hf_token}"})
            
        print(f">>> [MCP CLIENT] Connecting to {self.base_url}")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self._session.close()

    def reset(self, **kwargs) -> StepResult:
        """Resets the environment using the MCP reset endpoint."""
        try:
            # We use the standard /mcp prefix that openenv-core 0.2.x uses
            url = f"{self.base_url}/mcp/reset"
            payload = {"data": kwargs} if kwargs else {}
            resp = self._session.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return self._parse_result(data)
        except Exception as e:
            # Fallback check: try root if /mcp fails
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                return self._reset_root(**kwargs)
            raise

    def _reset_root(self, **kwargs) -> StepResult:
        url = f"{self.base_url}/reset"
        resp = self._session.post(url, json={"data": kwargs}, timeout=20)
        resp.raise_for_status()
        return self._parse_result(resp.json().get("data", {}))

    def step(self, action: FraudAction) -> StepResult:
        """Submits an action using the MCP step endpoint."""
        try:
            url = f"{self.base_url}/mcp/step"
            action_payload = {
                "decision":   action.decision,
                "confidence": action.confidence,
                "reasoning":  action.reasoning,
            }
            resp = self._session.post(url, json={"data": action_payload}, timeout=20)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return self._parse_result(data)
        except Exception as e:
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                 return self._step_root(action)
            raise

    def _step_root(self, action: FraudAction) -> StepResult:
        url = f"{self.base_url}/step"
        payload = {"data": {"decision": action.decision, "confidence": action.confidence, "reasoning": action.reasoning}}
        resp = self._session.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        return self._parse_result(resp.json().get("data", {}))

    def get_state(self) -> FraudState:
        """Fetches terminal state."""
        try:
            url = f"{self.base_url}/mcp/state"
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            valid_fields = FraudState.__dataclass_fields__.keys()
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return FraudState(**filtered)
        except Exception:
            # Fallback to root state
            resp = self._session.get(f"{self.base_url}/state", timeout=20)
            data = resp.json().get("data", {})
            valid_fields = FraudState.__dataclass_fields__.keys()
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return FraudState(**filtered)

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        obs = FraudObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            info=payload.get("info", {})
        )

    def sync(self):
        return self
