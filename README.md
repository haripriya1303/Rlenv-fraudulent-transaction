---
title: RL Fraud Detection Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: server/app.py
app_port: 7860
pinned: false
---

# 🏦 FraudGuard RL: A Strategic Hybrid Architecture for Real-World Transaction Defense

[![OpenEnv Spec v1.0](https://img.shields.io/badge/OpenEnv-v1.0-blue)](https://hf.co/spaces/openenv)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Deployed-orange)](https://harihari1906-rlenv-fraudulent-transaction-blocker.hf.space)

## 🌍 Environment Overview
**FraudGuard RL** is a high-fidelity simulator for modern banking fraud operations. Unlike simple classification datasets, this environment models the **Operational Constraints** of a real financial institution. Agents must not only catch fraud but manage **Customer Experience (UX)** and **Investigative Capacity**.

### Real-World Utility (30% Scoring Category)
This environment simulates a genuine task performed by thousands of fraud analysts daily. It replicates:
- **Transaction Velocity:** Detecting rapid-fire purchases.
- **Geographic Risk Analysis:** Calculating risk scores based on merchant location.
- **Customer Churn Risk:** Penalizing over-aggressive blocking of legitimate users.
- **Account Takeover (ATO):** Identifying brand-new device IDs that mismatch user history.

---

## 🛠️ Task & Grader Specification
We provide three standardized tasks with increasing complexity to evaluate agent reasoning and long-term utility management.

| Task | Difficulty | Grader Logic | Description |
| :--- | :--- | :--- | :--- |
| **Easy** | 🟢 Easy | **`easy_accuracy`** | Deterministic risk profiles; tests basic rule-following. |
| **Medium** | 🟡 Medium | **`medium_partial_credit`** | Introduces "Flag" rewards and False Positive penalties. |
| **Hard** | 🔴 Hard | **`hard_utility_regime`** | Dynamic adversarial patterns; requires strategic quota management. |

*All graders return a normalized score strictly in the range **[0.0, 1.0]** as per Hackathon requirements.*

---

## 🧠 Action & Observation Spaces

### Observation Space (19 Dimensions)
The agent receives a unified 19-dimensional feature vector:
1.  **Core Signals (8):** Amount (z-score), Velocity, Geo-Risk, Merchant-Risk, Device Consistency, User/Account Age.
2.  **Strategic Context (3):** Episode progress (0-1), current block rate (0-1), danger signal.
3.  **LLM Reasoning (3):** One-hot encoded decision mapping from the Frontier LLM.
4.  **Metadata (5):** Device type one-hots (mobile/desktop/tablet/unknown), cumulative reward tracker.

### Action Space (Discrete: 3)
- **`APPROVE` (0):** Maximize UX; high penalty if transaction is fraudulent.
- **`FLAG` (1):** Partial credit; sends for review; minor user friction.
- **`BLOCK` (2):** Immediate rejection; high reward if fraud; severe UX penalty if legitimate.

---

## 🚀 Unique Strategy: Hybrid Strategic Fusion
**FraudGuard RL** stands out by using a **Dual-Brain Architecture**:
- **The Reasoning Engine (LLM):** Uses a Frontier model (Qwen 2.5 72B) to interpret semantic risk from the transaction details.
- **The Strategic Controller (RL):** A lightweight PyTorch neural network that "filters" the LLM's recommendation. The RL agent learns that when it is near its "Block Quota," it should override an aggressive LLM to avoid a bankrupting penalty.

This solves the "Hallucination" and "Quota Ignorance" problems of standard LLM-only agents.

---

## ⚙️ Setup & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a .env file with your credentials:

```bash
HF_TOKEN=your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
FRAUD_TASK=medium
```
### 3. Run Inference
The inference.py script follows the mandatory OpenEnv logging format ([START], [STEP], [END]):

```bash
python inference.py
```
### 4. Reproduce Baseline Results
To compare against a zero-shot LLM baseline:

```bash
python baseline.py
```

### 5. Docker Support
The environment is fully containerized and ready for local or HF Space deployment:

```bash
docker build -t openenv-fraud .
docker run -p 8000:8000 openenv-fraud
```

---

## 📊 Baseline Benchmarks (Deterministic Seed 42)
| Metrics | Pure LLM Baseline | **FraudGuard Hybrid Agent** |
| :--- | :--- | :--- |
| **Easy Task Score** | 0.5750 | **0.7842** |
| **Medium Task Score** | 0.4392 | **0.6760** |
| **Hard Task Score** | 0.0058 | **0.1245** |

*Our Hybrid Agent consistently outperforms the pure LLM baseline by ~20% in medium/hard tasks by intelligently managing operational block quotas.*

---

## 🛡️ Anti-Exploit Design
To ensure fair and realistic evaluation, the environment includes:
1. **Entropy Injection:** Rare but valid "spike" transactions to prevent the agent from over-fitting to simple amount rules.
2. **Dynamic Quotas:** The environment monitors the agent's behavior. If an agent tries to "Always Block" or "Always Flag" to protect its score, the `RewardEngine` applies an exponential penalty (`penalty_scale = -4.0`).
3. **Reasoning Consistency:** Graders verify that the agent's actions match its internal reasoning confidence.

## 🤝 Spec Compliance Verification
This environment passes all **`openenv validate`** checks.
- **Manifest:** `openenv.yaml`
- **Models:** Optimized Pydantic models in `models.py`
- **Inference:** `inference.py` (Standardized Stdout Logging)
