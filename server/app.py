"""
Fraud Detection OpenEnv – FastAPI Server
"""
from __future__ import annotations

import os
from fastapi import FastAPI
from openenv.core.env_server import create_fastapi_app

from models import FraudAction, FraudObservation
from .fraud_environment import FraudEnvironment

TASK = os.environ.get("FRAUD_TASK", "medium")

app = create_fastapi_app(
    lambda: FraudEnvironment(task=TASK),
    FraudAction,
    FraudObservation
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
