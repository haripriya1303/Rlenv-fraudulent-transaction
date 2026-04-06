"""
Fraud Detection OpenEnv – FastAPI Server
Enhanced with direct HTTP routes for stable hackathon submission.
"""
from __future__ import annotations
import os
from fastapi import FastAPI, Request
from openenv.core.env_server import create_fastapi_app

from models import FraudAction, FraudObservation
from .fraud_environment import FraudEnvironment

# 1. Initialize the Environment
TASK = os.environ.get("FRAUD_TASK", "medium")
env_instance = FraudEnvironment(task=TASK)

# 2. Create the standard OpenEnv app
app = create_fastapi_app(
    lambda: env_instance,
    FraudAction,
    FraudObservation
)

# 3. ADD DIRECT STABILITY ROUTES
# These ensure that simple POST /reset and POST /step work even if MCP routing fails
@app.post("/reset")
async def manual_reset(request: Request):
    """Direct HTTP route for resetting the environment."""
    body = await request.json()
    config = body.get("data", body) if body else {}
    obs = env_instance.reset()
    return {"status": "success", "data": {"observation": obs}}

@app.post("/step")
async def manual_step(request: Request):
    """Direct HTTP route for taking a step."""
    body = await request.json()
    action_data = body.get("data", body)
    
    # Reconstruct Action model
    action = FraudAction(
        decision=action_data.get("decision"),
        confidence=float(action_data.get("confidence", 1.0)),
        reasoning=action_data.get("reasoning", "")
    )
    
    # Execute step (returns FraudObservation with reward/done attached)
    res_obs = env_instance.step(action)
    
    return {
        "status": "success", 
        "data": {
            "observation": res_obs,
            "reward": getattr(res_obs, "reward", 0.0),
            "done": getattr(res_obs, "done", False),
            "info": getattr(res_obs, "metadata", {})
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "task": TASK}

def main():
    import uvicorn
    # Use 7860 as the internal port for Hugging Face compatibility
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
