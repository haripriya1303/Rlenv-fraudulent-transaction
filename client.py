"""
Fraud Detection OpenEnv – Final Stabilized Client
Uses the exact 'action' key required by the server's Pydantic models.
"""
from __future__ import annotations
import os
import requests
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from models import FraudAction, FraudObservation, FraudState

@dataclass
class StepResult:
    observation: FraudObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

class FraudEnv:
    """
    Authenticated HTTP client tailored for Private Hugging Face Spaces.
    Forces communication through standard 'action' and 'observation' keys.
    """
    def __init__(self, base_url: str, **kwargs):
        self.base_url = base_url.rstrip("/")
        self.hf_token = os.getenv("HF_TOKEN")
        self._session = requests.Session()
        
        if self.hf_token:
            self._session.headers.update({"Authorization": f"Bearer {self.hf_token}"})
            
        print(f">>> [FINAL CLIENT] Connecting to {self.base_url}")

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def close(self): self._session.close()

    def reset(self, task: Optional[str] = None, **kwargs) -> StepResult:
        """Resets the environment, optionally switching the task."""
        try:
            url = f"{self.base_url}/reset"
            payload = {"task": task} if task else {}
            resp = self._session.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            
            # Server returns observation in 'data.observation' or 'observation'
            raw = resp.json()
            data = raw.get("data", raw)
            return self._parse_result(data)
        except Exception as e:
            print(f">>> [ERROR] Reset failed: {e}")
            raise

    def step(self, action: FraudAction) -> StepResult:
        """Submits an action using the 'action' key (exactly as the server expects)."""
        try:
            url = f"{self.base_url}/step"
            
            # 🛡️ THE FIX: Use the specific 'action' key required by Pydantic
            payload = {
                "action": {
                    "decision":   action.decision,
                    "confidence": float(action.confidence),
                    "reasoning":  str(action.reasoning)
                }
            }
            
            resp = self._session.post(url, json=payload, timeout=20)
            resp.raise_for_status()
            
            raw = resp.json()
            data = raw.get("data", raw)
            return self._parse_result(data)
        except Exception as e:
            print(f">>> [ERROR] Step failed: {e}")
            raise

    def get_state(self) -> FraudState:
        """Fetches terminal state."""
        try:
            url = f"{self.base_url}/state"
            resp = self._session.get(url, timeout=20)
            resp.raise_for_status()
            raw = resp.json()
            data = raw.get("data", raw)
            valid_fields = FraudState.__dataclass_fields__.keys()
            filtered = {k: v for k, v in data.items() if k in valid_fields}
            return FraudState(**filtered)
        except Exception as e:
            print(f">>> [ERROR] State fetch failed: {e}")
            raise

    def _parse_result(self, payload: dict) -> StepResult:
        # Extract observation (handles both flat and nested responses)
        obs_data = payload.get("observation", payload)
        if not isinstance(obs_data, dict):
             obs_data = {}
             
        obs = FraudObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            info=payload.get("info", {})
        )

    def sync(self):
        return self
