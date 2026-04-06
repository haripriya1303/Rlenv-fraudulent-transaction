---
title: RL Fraud Detection Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# 🏦 OpenEnv Fraud Detection Environment

> **Meta PyTorch Hackathon — Round 1: OpenEnv Environment**

A production-grade Reinforcement Learning environment that simulates a real banking fraud decision pipeline. An AI agent must sequentially review transactions and decide: **APPROVE**, **FLAG**, or **BLOCK** — balancing fraud prevention against customer experience.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-green)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-Spaces-yellow)](https://huggingface.co/spaces)

---

## 🎯 Overview

This environment models the full operational reality of a bank's fraud team:

| Decision | Correct on Fraud | Correct on Legit |
|----------|------------------|-----------------|
| **BLOCK**  | +1.00 ✅ | -0.70 ❌ |
| **FLAG**   | +0.50 ⚠️  | -0.30 ⚠️  |
| **APPROVE**| -1.00 ❌ | +0.50 ✅ |

**Plus**: A cascade over-blocking penalty when block rate exceeds 35%.

---

## 🚀 Quick Start

### Local (No Docker)

```bash
# 1. Install dependencies
pip install openenv-core
pip install -e .

# 2. Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 3. Open web interface
open http://localhost:8000/web
```

### Using the Client

```python
from fraud_env import FraudEnv, FraudAction

# Async usage
import asyncio

async def main():
    async with FraudEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        print(f"First transaction: ${result.observation.amount:.2f}")
        print(f"Country: {result.observation.country}")
        print(f"Merchant: {result.observation.merchant_type}")
        
        # Make a decision
        action = FraudAction(decision="FLAG", confidence=0.75)
        result = await client.step(action)
        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")

asyncio.run(main())
```

```python
# Sync usage
from fraud_env import FraudEnv, FraudAction

with FraudEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    while not result.done:
        action = FraudAction(decision="FLAG", confidence=0.5)
        result = client.step(action)
    print(f"Episode complete. Final reward: {result.reward}")
```

---

## 🐳 Docker

```bash
# Build
docker build -t openenv-fraud .

# Run (default task: medium)
docker run -p 8000:8000 openenv-fraud

# Run with different task or API key
docker run -p 8000:8000 \
  -e FRAUD_TASK=hard \
  -e OPENAI_API_KEY=sk-... \
  openenv-fraud
```

---

## 📋 Tasks

### 🟢 Easy — Obvious Fraud Detection
- **Steps**: 20
- **Fraud rate**: ~40%
- **Pattern**: Single high-confidence signal per transaction
  - Crypto at 3am from Nigeria
  - 15× amount spike + new device
  - Extreme velocity spike (25+ tx/day)
- **Grader**: Accuracy-based score ∈ [0, 1]
- **Seed**: `1001` (deterministic)

### 🟡 Medium — Multi-Feature Reasoning
- **Steps**: 30
- **Fraud rate**: ~40%
- **Pattern**: 2–3 moderate signals requiring joint reasoning
  - Gift card chain with elevated velocity
  - Account takeover: new device + foreign + suspicious merchant
  - Domestic gambling with night + medium risk country
- **Grader**: Partial credit (correct=1.0, partial=0.5, incorrect=0.0) + FP penalty
- **Seed**: `2002` (deterministic)

### 🔴 Hard — Long-Horizon Optimization
- **Steps**: 50
- **Fraud rate**: ~35%
- **Pattern**: Adversarial, subtle fraud designed to evade simple rules
  - Low-amount probing attacks
  - Behavioral drift patterns
  - Domestic device anomalies
- **Grader**: Normalized cumulative reward ∈ [0, 1]
- **Seed**: `3003` (deterministic)

---

## 🧠 Observation Space

Each step the agent receives:

| Feature | Type | Description |
|---------|------|-------------|
| `transaction_id` | str | Unique transaction ID |
| `amount` | float | Transaction amount (USD) |
| `country` | str | ISO-2 country code |
| `merchant_type` | str | Merchant category |
| `device_type` | str | Device: mobile/desktop/tablet/unknown |
| `user_age` | int | Customer age |
| `transaction_velocity` | int | Transactions in last 24h |
| `is_night` | bool | True if 23:00–05:00 |
| `account_age_days` | int | Account age in days |
| `amount_zscore` | float | Z-score vs user average |
| `geo_risk_score` | float | Country risk [0–1] |
| `merchant_risk_score` | float | Merchant risk [0–1] |
| `device_consistency` | float | 1.0=known device, 0.0=new |
| `step` | int | Current step in episode |
| `fraud_rate_so_far` | float | Episode fraud prevalence so far |
| `block_rate_so_far` | float | Agent's block rate so far |
| `cumulative_reward` | float | Running reward total |

---

## 🤖 Baseline Agent

```bash
# Set your OpenAI key
export OPENAI_API_KEY=sk-...

# Run all three tasks
python baseline.py

# Run a single task
python baseline.py --task easy

# Use a different model
python baseline.py --model gpt-4o
```

Results are saved to `outputs/evals/baseline_results.json`.

---

## 🧪 Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Submit an action |
| `/state` | GET | Get current episode state |
| `/health` | GET | Health check |
| `/info` | GET | Environment metadata |
| `/grade` | GET | Grade current episode |
| `/summary` | GET | Episode statistics |
| `/web` | GET | Interactive web UI |

---

## 🏗️ Architecture

```
openenv-fraud/
├── models.py                     # FraudAction, FraudObservation, FraudState
├── client.py                     # FraudEnv (EnvClient subclass)
├── __init__.py
├── openenv.yaml                  # Environment manifest
├── pyproject.toml
├── baseline.py                   # GPT-powered baseline agent
├── Dockerfile
└── server/
    ├── app.py                    # FastAPI server
    ├── fraud_environment.py      # Core RL environment
    ├── transaction_generator.py  # Hybrid probabilistic fraud synthesis
    ├── reward_engine.py          # Dense reward computation
    ├── episode_logger.py         # Structured per-step logging
    ├── tasks/__init__.py         # Task configs (easy/medium/hard)
    ├── graders/__init__.py       # Deterministic graders
    └── requirements.txt
```

---

## 🌐 Hugging Face Deployment

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Set SDK to **Docker**
3. Push this repository
4. Set secret: `OPENAI_API_KEY` (optional, for baseline agent)
5. Set variable: `FRAUD_TASK=medium` (or easy/hard)

Add the `openenv` tag to make it discoverable by the OpenEnv ecosystem.

---

## 📐 Fraud Signal Reference

| Signal | Risk Level | Description |
|--------|-----------|-------------|
| Crypto/gambling merchant | 🔴 High | 75–80% base risk |
| Wire transfer | 🟠 Medium-High | 55% base risk |
| High-risk country (NG, RU, KP…) | 🔴 High | 0.65–0.95 geo score |
| New device (consistency=0.0) | 🟠 Medium | Potential account takeover |
| Velocity > 20 tx/day | 🟠 Medium | Unusual transaction frequency |
| Amount z-score > 3 | 🟠 Medium | Significant anomaly vs baseline |
| Night transaction (23:00–05:00) | 🟡 Low-Medium | +7% score contribution |
| New account (<30 days) | 🟡 Low-Medium | +12% score contribution |

---

## 📝 License

MIT License — see [LICENSE](LICENSE).
