import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from typing import List

from client import FraudEnv
from models import FraudAction
from agent import FraudPolicy, select_action

# Default Hyperparameters
LR = 1.6e-3          
GAMMA = 0.99         
ENV_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

def train(episodes: int = 500):
    # 1. Initialize environment (Standard Sync Client)
    env_client = FraudEnv(base_url=ENV_URL)
    policy = FraudPolicy(input_dim=18)
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    
    print(f"🚀 [START] Strategic RL Training: {episodes} episodes", flush=True)
    
    # 2. Training Loop
    for ep in range(1, episodes + 1):
        # Episode lifecycle: open context if required, or direct use
        try:
            res = env_client.reset()
            obs = res.observation if hasattr(res, "observation") else res
            
            log_probs = []
            rewards = []
            done = False
            
            while not done:
                action_str, confidence, log_prob = select_action(policy, obs)
                
                action = FraudAction(
                    decision=action_str,
                    confidence=float(confidence),
                    reasoning="Strategic learning step"
                )
                
                step_res = env_client.step(action)
                obs = step_res.observation if hasattr(step_res, "observation") else step_res
                
                reward = getattr(step_res, "reward", 0.0)
                done = getattr(step_res, "done", False)
                
                log_probs.append(log_prob)
                rewards.append(reward)
                
            # Policy Update (REINFORCE)
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
            optimizer.step()
            
            if ep % 10 == 0 or ep == 1:
                print(f"Episode {ep:03d} | Reward: {sum(rewards):+.3f} | Loss: {loss.item():.4f}", flush=True)

        except Exception as e:
            print(f"⚠️ Episode {ep} failed: {e}", flush=True)
            continue

    torch.save(policy.state_dict(), "agent_weights.pth")
    print(f"\n✅ [DONE] Strategy upgraded. Weights: agent_weights.pth", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    train(episodes=args.episodes)
