# 🏦 FraudGuard RL — Hybrid Strategic Fraud Detection

[![OpenEnv Spec v1.0](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://hf.co/spaces/openenv)
[![HF Space](https://img.shields.io/badge/🤗_Space-Live-green)](https://harihari1906-rlenv-fraudulent-transaction-blocker.hf.space)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://github.com/haripriya1303/Rlenv-fraudulent-transaction)
[![Tasks](https://img.shields.io/badge/Tasks-Easy_|_Medium_|_Hard-orange)]()

> **The only OpenEnv environment where a neural network overrides a frontier LLM — because sometimes the smartest move is knowing when *not* to trust yourself.**

---

## 🎯 Why This Matters

Every day, fraud analysts at banks make thousands of decisions: approve, flag, or block. Get it wrong in either direction and you either let fraudsters through or you anger legitimate customers who just want to buy groceries abroad.

**This is not a toy problem.** FraudGuard RL simulates the exact operational constraints a real financial institution faces:

- A fraud analyst can only block so many transactions before triggering customer complaints
- Flagging too much overwhelms the human review team
- Missing real fraud is catastrophic

Standard LLM agents fail here. They are semantically smart but **operationally blind** — they don't know they're about to hit a block quota that will bankrupt the episode score. FraudGuard RL exists to solve this gap.

---

## 🧠 The Core Innovation: Dual-Brain Architecture

Most hackathon environments test one thing. This one tests two simultaneously — and the tension between them is the point.

```
Transaction Data
      │
      ▼
┌─────────────────────┐
│   Frontier LLM      │  ← Semantic reasoning: "This transaction pattern
│  (Qwen 2.5 72B)     │    looks like card testing fraud"
└─────────┬───────────┘
          │  LLM Decision + Confidence
          ▼
┌─────────────────────┐
│   RL Policy         │  ← Strategic override: "Yes, but we've blocked
│  (PyTorch 19-dim)   │    18% of transactions already. Flag instead."
└─────────┬───────────┘
          │  Final Action
          ▼
    APPROVE / FLAG / BLOCK
```

The LLM sees the transaction. The RL agent sees the episode. Together they outperform either alone.

**Benchmark results (seed 42):**

| Task | Pure LLM | FraudGuard Hybrid | Improvement |
|:-----|:--------:|:-----------------:|:-----------:|
| Easy | 0.5750 | **0.7842** | +36% |
| Medium | 0.4392 | **0.6760** | +54% |
| Hard | 0.0058 | **0.1245** | +2047% |

The Hard task delta tells the real story — an LLM with no quota awareness self-destructs. The hybrid agent survives.

---

## 🌍 Environment Design

### Three Tasks, Genuine Difficulty Progression

| Task | Grader | What Makes It Hard |
|:-----|:-------|:-------------------|
| 🟢 **Easy** | `easy_accuracy` | Deterministic risk profiles. Tests basic signal reading. |
| 🟡 **Medium** | `medium_partial_credit` | Introduces FLAG rewards and False Positive penalties. Pure BLOCK strategies collapse here. |
| 🔴 **Hard** | `hard_utility_regime` | Dynamic adversarial patterns + strict block quotas. Requires long-horizon strategic thinking. |

### Observation Space (19 Dimensions)

```
Core Transaction Signals (8)
  amount_zscore       — deviation from user's historical baseline
  transaction_velocity — transactions in last 24h
  geo_risk_score      — location-based risk
  merchant_risk_score — merchant category risk
  device_consistency  — does device match user history?
  user_age            — account holder age
  account_age_days    — how old is this account?
  is_night            — temporal risk signal

Episode Context (3)
  step_progress       — where are we in the episode (0→1)
  block_rate_so_far   — have we been too aggressive?
  danger_signal       — binary flag when block rate > 20%

LLM Reasoning (3)
  llm_approve         — one-hot: LLM said APPROVE
  llm_flag            — one-hot: LLM said FLAG
  llm_block           — one-hot: LLM said BLOCK

Device Metadata (4)
  mobile / desktop / tablet / unknown   — device type one-hots

Performance Signal (1)
  avg_reward_so_far   — running episode quality
```

### Anti-Exploit Design

Three mechanisms prevent degenerate strategies that game the score without solving the real problem:

- **Entropy Injection** — rare "spike" transactions that look fraudulent but aren't, punishing pure amount-threshold strategies
- **Dynamic Quotas** — exponential penalty (`scale = -4.0`) if block rate exceeds threshold, making "always block" fatal
- **Reasoning Consistency** — graders verify action confidence matches decision strength

---

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/haripriya1303/Rlenv-fraudulent-transaction
cd Rlenv-fraudulent-transaction
pip install -r requirements.txt
```

### 2. Configure

```bash
# Required
export HF_TOKEN=your_hf_token_here

# Optional — defaults provided
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_BASE_URL=http://localhost:8000
```

### 3. Start the Environment Server

```bash
docker build -t openenv-fraud ./server
docker run -p 8000:8000 openenv-fraud
```

### 4. Run Inference

```bash
python inference.py
```

### 5. Reproduce Baseline

```bash
python baseline.py
```

---

## 📋 Pre-Submission Validation

Run the validator before every submission:

```bash
chmod +x scripts/validate-submission.sh
./scripts/validate-submission.sh https://harihari1906-rlenv-fraudulent-transaction-blocker.hf.space .
```

Expected output:
```
[HH:MM:SS] PASSED -- HF Space is live and responds to /reset
[HH:MM:SS] PASSED -- Docker build succeeded
[HH:MM:SS] PASSED -- openenv validate passed
========================================
  All 3/3 checks passed!
  Your submission is ready to submit.
========================================
```

---

## 📁 Project Structure

```
.
├── inference.py          # OpenEnv-compliant agent (root, required)
├── baseline.py           # Zero-shot LLM baseline for comparison
├── agent.py              # FraudPolicy (PyTorch) + feature extractor
├── client.py             # FraudEnv HTTP client
├── models.py             # Pydantic typed models
├── openenv.yaml          # OpenEnv spec manifest
├── requirements.txt
├── Dockerfile
├── server/
│   ├── app.py            # FastAPI environment server
│   ├── graders.py        # easy / medium / hard graders
│   ├── reward_engine.py  # Reward shaping + quota management
│   └── transaction_generator.py
├── tests/
│   └── test_environment.py
└── scripts/
    └── validate-submission.sh
```

---

## ⚙️ Environment API

The server exposes three endpoints per OpenEnv spec:

```
POST /reset          → { observation: FraudObservation }
POST /step           → { observation, reward, done, info }
GET  /state          → FraudState (terminal summary)
GET  /grade          → { score: float }   # 0.0–1.0
```

All rewards are normalized to `[0.0, 1.0]`. All graders are deterministic given the same seed.

---

## 🔧 OpenEnv Spec Compliance

```yaml
# openenv.yaml
name: openenv-fraud
version: "1.0"
tasks:
  - easy
  - medium
  - hard
endpoints:
  reset: /reset
  step: /step
  state: /state
```

Validated with `openenv validate` ✅

---

## 📊 Stdout Log Format

```
[START] task=easy env=openenv-fraud model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=APPROVE reward=0.73 done=false error=null
[STEP] step=2 action=FLAG reward=0.62 done=false error=null
[STEP] step=3 action=BLOCK reward=0.88 done=false error=null
...
[END] success=true steps=25 score=0.78 rewards=0.73,0.62,0.88,...
```

---

## 🏗️ Hardware Requirements

Runs within hackathon constraints (2 vCPU, 8 GB RAM):

| Component | Footprint |
|:----------|:----------|
| RL Policy | ~50KB (19→128→128→3) |
| Environment server | ~180MB RAM |
| LLM calls | External API, zero local GPU |
| Inference runtime | < 8 minutes for all 3 tasks |

---

## 🤝 Spec Links

- HF Space: [harihari1906/Rlenv-fraudulent-transaction-blocker](https://huggingface.co/spaces/harihari1906/Rlenv-fraudulent-transaction-blocker)
- GitHub: [haripriya1303/Rlenv-fraudulent-transaction](https://github.com/haripriya1303/Rlenv-fraudulent-transaction)