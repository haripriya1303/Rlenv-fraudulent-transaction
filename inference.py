import asyncio
import os
import json
import time
from typing import List, Optional
from openai import OpenAI

from env.models import FraudAction
from env.environment import FraudEnvironment

# ===== ENV CONFIG =====
IMAGE_NAME = os.getenv("IMAGE_NAME", "openenv-fraud")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK_NAME = os.getenv("FRAUD_TASK", "medium")
BENCHMARK = "openenv-fraud"

MAX_STEPS = 20
SUCCESS_THRESHOLD = 0.5

# ===== PROMPTS (FROM BASELINE) =====

SYSTEM_PROMPT = """You are an expert banking fraud analyst reviewing transactions in real time.

Classify each transaction as:
- APPROVE
- FLAG
- BLOCK

Respond ONLY in JSON:
{
  "decision": "...",
  "confidence": 0.0-1.0,
  "reasoning": "short"
}
"""

TRANSACTION_TEMPLATE = """Transaction:
ID: {transaction_id}
Amount: ${amount:.2f} (z={amount_zscore:+.2f})
Country: {country} (risk={geo_risk_score:.2f})
Merchant: {merchant_type} (risk={merchant_risk_score:.2f})
Device: {device_type} (consistency={device_consistency:.2f})
Velocity: {transaction_velocity}
Night: {is_night}
Account age: {account_age_days}

Step {step}/{max_steps}
Fraud rate: {fraud_rate_so_far:.2f}
Block rate: {block_rate_so_far:.2f}
Reward: {cumulative_reward:.2f}
"""

# ===== LOGGING =====

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ===== AGENT (FROM BASELINE) =====

def get_action(client, obs):
    obs_dict = {
        "transaction_id": obs.transaction_id,
        "amount": obs.amount,
        "amount_zscore": obs.amount_zscore,
        "country": obs.country,
        "geo_risk_score": obs.geo_risk_score,
        "merchant_type": obs.merchant_type,
        "merchant_risk_score": obs.merchant_risk_score,
        "device_type": obs.device_type,
        "device_consistency": obs.device_consistency,
        "transaction_velocity": obs.transaction_velocity,
        "is_night": obs.is_night,
        "account_age_days": obs.account_age_days,
        "step": obs.step,
        "max_steps": obs.max_steps,
        "fraud_rate_so_far": obs.fraud_rate_so_far,
        "block_rate_so_far": obs.block_rate_so_far,
        "cumulative_reward": obs.cumulative_reward,
    }

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


# ===== MAIN =====

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = await FraudEnvironment.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = get_action(client, obs)

            result = await env.step(action)

            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(step, action.decision, reward, done, None)

            if done:
                break

        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        success = avg_reward >= SUCCESS_THRESHOLD

    finally:
        try:
            await env.close()
        except:
            pass

        log_end(success, steps_taken, rewards)


if __name__ == "__main__":
    asyncio.run(main())