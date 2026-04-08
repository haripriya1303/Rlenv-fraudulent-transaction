import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import argparse
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
from typing import List, Any
print(">>> DEBUG: SCRIPT START")

from client import FraudEnv
from models import FraudAction
from agent import FraudPolicy, select_action, extract_features

# Hyperparameters (Upgraded for Stability)
LR = 5e-4            
GAMMA = 0.99         
CLIP_GRAD = 1.0      
ENV_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# OpenAI Setup for Training
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

def get_llm_signal(client: OpenAI, obs: Any) -> str:
    """Gets the LLM's classification signal as a feature."""
    try:
        prompt = f"Transaction: Amount={obs.amount}, Country={obs.country}. Decision (APPROVE/FLAG/BLOCK) JSON."
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content).get("decision", "APPROVE").upper()
    except:
        return "APPROVE"

def train(episodes: int = 500):
    print(">>> DEBUG: ENTERING TRAIN", flush=True)
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env_client = FraudEnv(base_url=ENV_URL)
    
    # 1. Dynamic Dimension Auto-Detection
    try:
        sample_res = env_client.reset()
        sample_obs = sample_res.observation if hasattr(sample_res, "observation") else sample_res
        input_dim = extract_features(sample_obs).shape[-1]
        print(f"✅ AUTO-DETECTED FEATURE DIM: {input_dim}", flush=True)
    except Exception as e:
        print(f"⚠️ Failed to auto-detect dimension: {e}. Defaulting to 19.")
        input_dim = 19

    policy = FraudPolicy(input_dim=input_dim) 
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    print(f"🚀 [START] Strategic Fusion Training: {episodes} episodes", flush=True)
    
    for ep in range(1, episodes + 1):
        try:
            res = env_client.reset()
            obs = res.observation if hasattr(res, "observation") else res
            
            log_probs = []
            rewards = []
            done = False
            
            while not done:
                # 1. Get LLM Signal
                llm_decision = get_llm_signal(client, obs)

                # 2. RL Action with exploration
                action_str, confidence, log_prob = select_action(
                    policy, obs, llm_decision=llm_decision, deterministic=False
                )
                
                action = FraudAction(
                    decision=action_str,
                    confidence=float(confidence),
                    reasoning=f"Hybrid strategic update"
                )
                
                step_res = env_client.step(action)
                obs = step_res.observation if hasattr(step_res, "observation") else step_res
                
                # 3. Process Reward
                env_reward = float(getattr(step_res, "reward", 0.0) or 0.0)
                
                # Apply Sigmoid Normalization (Consistency with inference.py)
                normalized_reward = 1 / (1 + math.exp(-env_reward))
                reward = round(normalized_reward, 4)
                
                # Internal Reward Shaping: Strategic bonus for catching fraud near quotas
                if obs.block_rate_so_far > 0.20 and env_reward > 0:
                    reward += 0.15 
                
                log_probs.append(log_prob)
                rewards.append(reward)
                done = bool(getattr(step_res, "done", False) or False)
                
            # --- POLICY UPDATE ---
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)
            
            returns = torch.tensor(returns)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            policy_loss = []
            for lp, G in zip(log_probs, returns):
                policy_loss.append(-lp * G)
                
            optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP_GRAD)
            optimizer.step()
            
            if ep % 5 == 0 or ep == 1:
                print(f"Episode {ep:03d} | Reward: {sum(rewards):+.2f} | Loss: {loss.item():.4f}", flush=True)

        except Exception as e:
            print(f"⚠️ Episode {ep} error: {e}", flush=True)
            continue

    torch.save(policy.state_dict(), "agent_weights.pth")
    print(f"\n✅ [DONE] Strategy upgraded. Weights: agent_weights.pth", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    train(episodes=args.episodes)
