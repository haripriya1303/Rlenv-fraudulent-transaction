import asyncio
import os
import textwrap
import json
import time
from dotenv import load_dotenv

load_dotenv()
from typing import List, Optional

from openai import OpenAI

from models import FraudAction, FraudObservation
from client import FraudEnv

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("FRAUD_TASK", "medium")
BENCHMARK = "openenv-fraud"

MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score

SYSTEM_PROMPT = """You are an expert banking fraud analyst reviewing transactions in real time.

Your task is to classify each transaction as exactly one of:
  - APPROVE: Transaction appears legitimate. Allow it.
  - FLAG:    Transaction is suspicious. Send for human review.
  - BLOCK:   Transaction is high-confidence fraud. Reject immediately.

Guidelines:
- BLOCK only when you have strong multi-signal evidence of fraud.
- FLAG when signals are mixed or moderate risk.
- APPROVE clearly legitimate transactions to maintain customer experience.
- Systematic over-blocking will incur operational penalties.

You MUST respond ONLY with a valid JSON object in this exact format:
{
  "decision": "APPROVE" | "FLAG" | "BLOCK",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<brief explanation>"
}
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

Risk interpretation guide:
  - Geo risk   > 0.5 = high-risk country
  - Merchant risk > 0.5 = high-risk merchant (crypto, gambling, wire transfer)
  - Device consistency = 0.0 means brand-new device (potential takeover)
  - Amount z-score > 2.0 = significant anomaly vs user average
  - Velocity > 15 = unusual transaction frequency
"""

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def get_action(client: OpenAI, obs_dict: dict) -> FraudAction:
    user_prompt = TRANSACTION_TEMPLATE.format(**obs_dict)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"},
            )

            parsed = json.loads(response.choices[0].message.content)

            decision = parsed.get("decision", "APPROVE").upper()
            if decision not in ("APPROVE", "FLAG", "BLOCK"):
                decision = "APPROVE"

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return FraudAction(
                decision=decision,
                confidence=confidence,
                reasoning=str(parsed.get("reasoning", "")),
            )

        except Exception:
            time.sleep(1)

    # fallback
    return FraudAction(decision="FLAG", confidence=0.3, reasoning="fallback")


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
    env = FraudEnv(base_url=env_url)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset() # OpenEnv standard uses reset_res.observation
        obs = result.observation if hasattr(result, "observation") else result

        for step in range(1, MAX_STEPS + 1):
            # Extract features manually to be safe with model types
            obs_dict = {k: v for k, v in obs.__dict__.items() if not k.startswith("_")}
            
            action = get_action(client, obs_dict)

            step_res = env.step(action)
            obs = step_res.observation if hasattr(step_res, "observation") else step_res

            reward = getattr(step_res, "reward", 0.0)
            if reward is None: reward = 0.0
            done = getattr(step_res, "done", False)
            if done is None: done = False
            error = None

            rewards.append(float(reward))
            steps_taken = step

            log_step(step=step, action=action.decision, reward=reward, done=done, error=error)

            if done:
                try:
                    import requests
                    resp = requests.get(f"{env_url}/grade").json()
                    score = resp.get("score", 0.0)
                except Exception:
                    score = sum(rewards) / (step * 1.5)
                break

        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except:
            pass

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())