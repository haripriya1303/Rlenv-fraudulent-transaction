"""
inference.py — OpenEnv-compliant hybrid RL+LLM fraud detection agent.
Multi-task enabled (Easy, Medium, Hard) to satisfy validator requirements.
"""

import os
import json
import math
import requests
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from models import FraudAction, FraudObservation
from client import FraudEnv
from agent import FraudPolicy, extract_features, select_action

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# ── Config (guidelines-compliant) ────────────────────────────────────────────
# API_BASE_URL and MODEL_NAME MUST have defaults (per guidelines)
# HF_TOKEN MUST NOT have a default (per guidelines)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK               = "openenv-fraud"
MAX_STEPS               = 50
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert banking fraud analyst reviewing transactions in real time.
Your task is to classify each transaction as exactly one of:
- APPROVE: Transaction appears legitimate. Allow it.
- FLAG: Transaction is suspicious. Send for human review.
- BLOCK: Transaction is high-confidence fraud. Reject immediately.

You MUST respond ONLY with a valid JSON object:
{"decision": "APPROVE"|"FLAG"|"BLOCK", "confidence": <float>, "reasoning": "<brief explanation>"}
"""

TRANSACTION_TEMPLATE = """Review this transaction and decide: APPROVE, FLAG, or BLOCK.

Transaction Details:
  - ID:           {transaction_id}
  - Amount:       ${amount:.2f} (z-score: {amount_zscore:+.2f} vs user baseline)
  - Country:      {country} (geo risk score: {geo_risk_score:.2f})
  - Merchant:     {merchant_type} (merchant risk: {merchant_risk_score:.2f})
  - Device:       {device_type} (consistency: {device_consistency:.2f})
  - User age:     {user_age} years
  - Velocity:     {transaction_velocity} transactions in last 24h
  - Night tx:     {is_night}
  - Account age:  {account_age_days} days

Episode Context:
  - Step:              {step}/{max_steps}
  - Fraud rate so far: {fraud_rate_so_far:.1%}
  - Block rate so far: {block_rate_so_far:.1%}
  - Cumulative reward: {cumulative_reward:.3f}
"""

# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    reported_reward = min(max(float(reward), 0.001), 0.999)
    done_val = str(bool(done)).lower()
    error_val = str(error) if (error and str(error).strip() not in ("None", "")) else "null"
    print(f"[STEP] step={step} action={action} reward={reported_reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    score = min(max(float(score), 0.01), 0.99)
    rewards_str = ",".join(f"{min(max(r, 0.01), 0.99):.2f}" for r in rewards)
    print(f"[END] success={str(bool(success)).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ── Action selection ──────────────────────────────────────────────────────────

def get_hybrid_action(client: OpenAI, policy, obs: FraudObservation) -> FraudAction:
    user_prompt = TRANSACTION_TEMPLATE.format(**obs.__dict__)
    llm_decision = "APPROVE"
    reasoning = "N/A"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )
        content = response.choices[0].message.content
        if "{" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        parsed = json.loads(content)
        reasoning    = str(parsed.get("reasoning", "N/A"))
        llm_decision = str(parsed.get("decision", "APPROVE")).upper()
    except Exception as e:
        print(f"[DEBUG] CRITICAL LLM ERROR: {e}", flush=True)
        llm_decision = "APPROVE"

    if TORCH_AVAILABLE and policy is not None:
        try:
            action_str, confidence, _ = select_action(
                policy, obs, llm_decision=llm_decision, deterministic=True
            )
            return FraudAction(decision=action_str, confidence=float(confidence), reasoning=f"[Fusion] {reasoning}")
        except Exception:
            pass

    return FraudAction(decision=llm_decision, confidence=0.5, reasoning=f"[LLM] {reasoning}")

# ── Simulation Engine ─────────────────────────────────────────────────────────

def run_simulation(client: OpenAI, env: FraudEnv, policy: Optional[FraudPolicy], task_name: str) -> None:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = env.reset(task=task_name)
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

            raw_reward = float(getattr(step_res, "reward", 0.0) or 0.0)
            normalized_reward = 1 / (1 + math.exp(-raw_reward))
            final_reward = min(max(normalized_reward, 0.001), 0.999)

            done = bool(getattr(step_res, "done", False) or False)
            rewards.append(final_reward)
            steps_taken = step

            log_step(step=step, action=action.decision, reward=final_reward, done=done, error=error_str)

            if done:
                try:
                    resp = requests.get(f"{env.base_url}/grade", timeout=5).json()
                    remote_score = resp.get("score")
                    score = float(remote_score) if remote_score is not None else sum(rewards) / len(rewards)
                except Exception:
                    score = sum(rewards) / len(rewards) if rewards else 0.0
                break

        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal Task Error ({task_name}): {e}", flush=True)
        success = False
        steps_taken = len(rewards)
        score = 0.01
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Initialized using guidelines-compliant variables
    # Validator overrides API_BASE_URL with its proxy at runtime
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    env_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
    env = FraudEnv(base_url=env_url)

    # Load RL policy once
    policy = None
    if TORCH_AVAILABLE:
        try:
            sample_res = env.reset(task="medium")
            sample_obs = sample_res.observation if hasattr(sample_res, "observation") else sample_res
            input_dim = extract_features(sample_obs).shape[-1]
            policy = FraudPolicy(input_dim=input_dim)
            best_path = "agent_weights_best.pth"
            if os.path.exists(best_path):
                policy.load_state_dict(torch.load(best_path, map_location="cpu"))
                policy.eval()
                print(f"[DEBUG] Loaded Weights: {best_path}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Policy failed: {e}", flush=True)
            policy = None

    # Run all three tasks required by validator
    for target_task in ["easy", "medium", "hard"]:
        try:
            run_simulation(client, env, policy, target_task)
        except Exception as task_err:
            print(f"[DEBUG] Simulation failed for {target_task}: {task_err}", flush=True)

if __name__ == "__main__":
    main()