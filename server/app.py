"""
Fraud Detection OpenEnv – FastAPI Server
Enhanced with direct HTTP routes and 422 error diagnostics.
"""
from __future__ import annotations
import os
import traceback
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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

# 3. GLOBAL EXCEPTION HANDLER FOR 422s (DEBUGGING)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Prints the exact 422 error to the Hugging Face logs."""
    print(f">>> [DEBUG 422] Validation failed: {exc.errors()}", flush=True)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)},
    )

# 4. ADD DIRECT STABILITY ROUTES
@app.post("/reset")
async def manual_reset(request: Request):
    """Direct HTTP route for resetting the environment."""
    try:
        body = await request.json()
        config = body.get("data", body) if body else {}
        obs = env_instance.reset()
        return {"status": "success", "data": {"observation": obs}}
    except Exception as e:
        print(f">>> [ERROR] Manual reset failed: {e}", flush=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/step")
async def manual_step(request: Request):
    """Direct HTTP route for taking a step."""
    try:
        body = await request.json()
        action_payload = body.get("data", body)
        
        # 🛡️ PROTECT: Ensure raw data is flattened if it comes from my custom client
        if isinstance(action_payload, dict) and "data" in action_payload:
            action_payload = action_payload["data"]

        # Manual conversion with robust error handling for judge agents
        try:
            decision = str(action_payload.get("decision", "APPROVE")).upper()
            if decision not in ["APPROVE", "FLAG", "BLOCK"]:
                decision = "APPROVE"  # Safe default
                
            confidence_val = action_payload.get("confidence", 1.0)
            try:
                confidence = float(confidence_val)
            except (ValueError, TypeError):
                confidence = 1.0
                
            action = FraudAction(
                decision=decision,
                confidence=confidence,
                reasoning=str(action_payload.get("reasoning", "Agent Step"))
            )
        except Exception as validation_err:
            print(f">>> [WARN] Action validation failed: {validation_err}", flush=True)
            # Fallback to safe action
            action = FraudAction(decision="APPROVE", confidence=1.0, reasoning="Validation Fallback")
        
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
    except Exception as e:
        print(f">>> [ERROR] Fatal Step Error: {e}", flush=True)
        traceback.print_exc()
        # Return a graceful end-of-episode observation if possible
        return JSONResponse(status_code=500, content={"error": "Fatal server error during step"})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "task": TASK}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
