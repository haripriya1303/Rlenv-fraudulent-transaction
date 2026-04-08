"""
inference.py — OpenEnv-compliant hybrid RL+LLM fraud detection agent.
FIXED: Removed load_dotenv(), fixed API_KEY to use injected credentials,
       fixed [START] log order per R8.
"""

import os
import json
import math
import requests
import traceback
from typing import List, Optional

# ── CRITICAL: NO load_dotenv() here — validator injects env vars directly ──

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

# ── Config ────────────────────────────────────────────────────────────────────
# PRE-SUBMISSION CHECKLIST VERBATIM
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = os.getenv("API_KEY")

# Choose the active credentials
# Platform injections: os.environ["API_BASE_URL"] and os.environ["API_KEY"]
ACTIVE_KEY = API_KEY or HF_TOKEN
ACTIVE_URL = os.environ.get("API_BASE_URL", API_BASE_URL)

TASK_NAME = os.getenv("FRAUD_TASK", "medium")
BENCHMARK = "openenv-fraud"
MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert banking fraud analyst reviewing transactions in real time.
Your task is to classify each transaction as exactly one of:
- APPROVE: Transaction appears legitimate. Allow it.
- FLAG: Transaction is suspicious. Send for human review.
- BLOCK: Transaction is high-confidence fraud. Reject immediately.

Guidelines:
- BLOCK only when you have strong multi-signal evidence of fraud.
- FLAG when signals are mixed or moderate risk.
- APPROVE clearly legitimate transactions to maintain customer experience.
- Systematic over-blocking will incur operational penalties.

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
    """R8: Must be the FIRST line of output."""
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """R1-R3: 2dp reward clamped [0,1], lowercase done, null error."""
    reported_reward = min(max(float(reward), 0.0), 1.0)
    done_val = str(bool(done)).lower()
    error_val = str(error) if (error and str(error).strip() not in ("None", "")) else "null"
    print(f"[STEP] step={step} action={action} reward={reported_reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """R4-R7: lowercase success, score= at 2dp, comma-sep rewards."""
    score = min(max(float(score), 0.0), 1.0)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
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
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        content = response.choices[0].message.content
        if "{" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        parsed = json.loads(content)
        reasoning = str(parsed.get("reasoning", "N/A"))
        llm_decision = str(parsed.get("decision", "APPROVE")).upper()
    except Exception as e:
        print(f"[DEBUG] CRITICAL LLM ERROR: {e}", flush=True)
        traceback.print_exc() # This will show the full error in the dashboard logs
        llm_decision = "APPROVE"

    if TORCH_AVAILABLE and policy is not None:
        try:
            action_str, confidence, _ = select_action(
                policy, obs, llm_decision=llm_decision, deterministic=True
            )
            return FraudAction(decision=action_str, confidence=float(confidence), reasoning=f"[Fusion] {reasoning}")
        except Exception as e:
            return FraudAction(decision=llm_decision, confidence=0.5, reasoning=f"[Fallback] {reasoning} err={e}")

    return FraudAction(decision=llm_decision, confidence=0.5, reasoning=f"[LLM] {reasoning}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # R8: [START] MUST be printed before anything else, including env.reset()
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    # 🔗 LITELLM PROXY CONNECTION
    # We use os.environ directly to ensure we catch the platform's injected values
    effective_url = os.environ.get("API_BASE_URL", API_BASE_URL)
    effective_key = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN")

    print(f">>> [INIT] Base URL: {effective_url}", flush=True)
    print(f">>> [INIT] API Key Provided: {'Yes' if effective_key else 'No'}", flush=True)

    client = OpenAI(
        base_url=effective_url,
        api_key=effective_key,
    )

    env_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
    env = FraudEnv(base_url=env_url)

    rewards: List[float] = []
    raw_rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Load policy
    policy = None
    if TORCH_AVAILABLE:
        try:
            sample_res = env.reset()
            sample_obs = sample_res.observation if hasattr(sample_res, "observation") else sample_res
            input_dim = extract_features(sample_obs).shape[-1]
            policy = FraudPolicy(input_dim=input_dim)

            best_path = "agent_weights_best.pth"
            fallback_path = "agent_weights.pth"
            weights_path = best_path if os.path.exists(best_path) else fallback_path

            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                if state_dict["input_layer.weight"].shape[1] == input_dim:
                    policy.load_state_dict(state_dict)
                    policy.eval()
                    print(f"[DEBUG] Loaded weights from {weights_path}", flush=True)
                else:
                    print("[DEBUG] Weight dimension mismatch, using zero-shot.", flush=True)
                    policy = None
            else:
                print("[DEBUG] No weights found, using zero-shot.", flush=True)
                policy = None
        except Exception as e:
            print(f"[DEBUG] Policy load error: {e}", flush=True)
            policy = None
    else:
        print("[DEBUG] Torch unavailable. LLM-only mode.", flush=True)

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

            raw_reward = float(getattr(step_res, "reward", 0.0) or 0.0)
            # Sigmoid normalization for stable gradients, then clamp for OpenEnv compliance
            normalized_reward = 1 / (1 + math.exp(-raw_reward))
            final_reward = min(max(normalized_reward, 0.0), 1.0)

            done = bool(getattr(step_res, "done", False) or False)
            raw_rewards.append(raw_reward)
            rewards.append(final_reward)
            steps_taken = step

            log_step(step=step, action=action.decision, reward=final_reward, done=done, error=error_str)

            if done:
                try:
                    resp = requests.get(f"{env_url}/grade", timeout=5).json()
                    remote_score = resp.get("score")
                    score = float(remote_score) if remote_score is not None else sum(rewards) / len(rewards)
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
        score = sum(rewards) / len(rewards) if rewards else 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()