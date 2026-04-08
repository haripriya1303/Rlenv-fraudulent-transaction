"""
inference.py — OpenEnv-compliant hybrid RL+LLM fraud detection agent.
ROBUST VERSION: Handles missing torch gracefully by falling back to zero-shot LLM.

Compliance (9 rules enforced):
  R1 [STEP] reward: 2dp, clamped [0.0, 1.0]
  R2 [STEP] done:   lowercase "true"/"false"
  R3 [STEP] error:  "null" or string
  R4 [END]  success: lowercase "true"/"false"
  R5 [END]  always logs via finally block
  R6 [END]  rewards: comma-sep, each 2dp
  R7 [END]  score= field, 2dp, clamped [0.0, 1.0]
  R8 [START] is printed FIRST (before env.reset)
  R9 Load agent_weights_best.pth if exists, fallback to agent_weights.pth
"""

import os
import json
import requests
import traceback
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()

from openai import OpenAI
from models import FraudAction, FraudObservation
from client import FraudEnv
from agent import FraudPolicy, extract_features, select_action, TORCH_AVAILABLE

try:
    import torch
except ImportError:
    torch = None

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
# Soft-fail if token is missing (useful for some dev environments)
if HF_TOKEN is None:
    print("[DEBUG] HF_TOKEN is missing. Set it for LLM support.", flush=True)
    HF_TOKEN = "dummy_token"

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = HF_TOKEN

TASK_NAME = os.getenv("FRAUD_TASK", "medium")
BENCHMARK = "openenv-fraud"
MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert banking fraud analyst reviewing transactions in real time.
You MUST respond ONLY with valid JSON: {"decision": "APPROVE"|"FLAG"|"BLOCK", "confidence": <float>, "reasoning": "..."}
"""

TRANSACTION_TEMPLATE = """Review this:
  - ID:           {transaction_id}
  - Amount:       ${amount:.2f}
  - Velocity:     {transaction_velocity}
  - Risk scores:  Geo={geo_risk_score:.2f}, Merchant={merchant_risk_score:.2f}
"""

# ── Logging helpers (Rule compliance) ───────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """R8: First line of output."""
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, env_reward: float, done: bool, error: Optional[str]) -> None:
    """R1-R3: 2dp reward, lowercase done, null error."""
    reported_reward = min(max(float(env_reward), 0.0), 1.0)
    done_val = str(bool(done)).lower()
    error_val = str(error) if (error and str(error).strip() != "None") else "null"
    print(f"[STEP] step={step} action={action} reward={reported_reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """R4-R7: lowercase success, score= present at 2dp, comma-sep rewards."""
    score = min(max(float(score), 0.0), 1.0)
    success_val = str(bool(success)).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_val} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ── Action selection ──────────────────────────────────────────────────────────

def get_hybrid_action(client: OpenAI, policy: Optional[FraudPolicy], obs: FraudObservation) -> FraudAction:
    user_prompt = TRANSACTION_TEMPLATE.format(**obs.__dict__)
    llm_decision = "APPROVE"
    reasoning = "N/A"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        reasoning = str(parsed.get("reasoning", ""))
        llm_decision = str(parsed.get("decision", "APPROVE")).upper()
    except Exception:
        llm_decision = "APPROVE"

    # Hybrid logic with Torch guard
    if TORCH_AVAILABLE and policy is not None:
        try:
            action_str, confidence, _ = select_action(policy, obs, llm_decision=llm_decision, deterministic=True)
            return FraudAction(decision=action_str, confidence=float(confidence), reasoning=f"[Hybrid] {reasoning}")
        except Exception as e:
            return FraudAction(decision=llm_decision, confidence=0.5, reasoning=f"[Fallback] {reasoning} err={e}")

    return FraudAction(decision=llm_decision, confidence=0.5, reasoning=f"[Zero-Shot] {reasoning}")

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
    env = FraudEnv(base_url=env_url or "http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # R8: Start logging before any env interactions
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # R9: Best weight loading logic with Torch guard
    policy: Optional[FraudPolicy] = None
    if TORCH_AVAILABLE:
        try:
            sample_res = env.reset()
            sample_obs = sample_res.observation if hasattr(sample_res, "observation") else sample_res
            policy = FraudPolicy(input_dim=19)
            
            best_path = "agent_weights_best.pth"
            final_path = "agent_weights.pth"
            weights_path = best_path if os.path.exists(best_path) else final_path
            
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                policy.load_state_dict(state_dict)
                policy.eval()
                print(f"[DEBUG] Loaded {weights_path}", flush=True)
            else:
                print("[DEBUG] No weights found, Using zero-shot policy.", flush=True)
                policy = None
        except Exception as e:
            print(f"[DEBUG] Weight load failure: {e}", flush=True)
            policy = None
    else:
        print("[DEBUG] Torch not available. Running in LLM mode.", flush=True)

    # R5: episode wrapped in try/finally
    try:
        result = env.reset()
        obs = result.observation if hasattr(result, "observation") else result

        for step in range(1, MAX_STEPS + 1):
            error_str = None
            try:
                action = get_hybrid_action(client, policy, obs)
            except Exception as e:
                error_str = str(e)
                action = FraudAction(decision="FLAG", confidence=0.5, reasoning="Inference Error")

            step_res = env.step(action)
            obs = step_res.observation if hasattr(step_res, "observation") else step_res
            
            reward = float(getattr(step_res, "reward", 0.0) or 0.0)
            done = bool(getattr(step_res, "done", False) or False)
            
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action.decision, env_reward=reward, done=done, error=error_str)

            if done:
                try:
                    resp = requests.get(f"{env_url}/grade", timeout=5).json()
                    score = float(resp.get("score", sum(rewards)/len(rewards)))
                except Exception:
                    score = sum(rewards) / len(rewards) if rewards else 0.0
                break

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal Error: {e}", flush=True)
        traceback.print_exc()
        success = False
        steps_taken = len(rewards)
        score = sum(rewards)/len(rewards) if rewards else 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()