import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple

class FraudPolicy(nn.Module):
    """
    Pure PyTorch Policy Network for Fraud Detection.
    Architecture: Simple MLP (Multi-Layer Perceptron)
    Input: Numerical features from FraudObservation
    Output: Action probabilities for [APPROVE, FLAG, BLOCK]
    """
    def __init__(self, input_dim: int = 18, hidden_dim: int = 64):
        super(FraudPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 actions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns categorical distribution (probabilities)
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

def extract_features(obs: Any) -> torch.Tensor:
    """
    Manually convert FraudObservation to a numerical tensor.
    Matches the input_dim=18 in FraudPolicy.
    """
    # 1. Continuous Features (Normalizing where possible)
    amount = min(obs.amount / 5000.0, 5.0)  # capped normalization
    user_age = obs.user_age / 100.0
    velocity = obs.transaction_velocity / 50.0
    account_age = min(obs.account_age_days / 3650.0, 1.0)
    z_score = obs.amount_zscore / 10.0
    geo_risk = obs.geo_risk_score
    merchant_risk = obs.merchant_risk_score
    device_cons = obs.device_consistency
    night = 1.0 if obs.is_night else 0.0
    
    # 2. Episode Progress
    progress = obs.step / max(obs.max_steps, 1)
    fraud_so_far = obs.fraud_rate_so_far
    block_so_far = obs.block_rate_so_far
    # Signal if we are over the 25% threshold
    danger_zone = 1.0 if block_so_far > 0.25 else 0.0
    
    # 3. Categorical: Device Type (mobile, desktop, tablet, unknown)
    device_one_hot = [0.0, 0.0, 0.0, 0.0]
    dt = str(obs.device_type).lower()
    if dt == "mobile": device_one_hot[0] = 1.0
    elif dt == "desktop": device_one_hot[1] = 1.0
    elif dt == "tablet": device_one_hot[2] = 1.0
    else: device_one_hot[3] = 1.0

    features = [
        amount, user_age, velocity, account_age, z_score,
        geo_risk, merchant_risk, device_cons, night,
        progress, fraud_so_far, block_so_far, danger_zone,
        *device_one_hot,
        obs.cumulative_reward / (obs.step + 1)
    ]
    
    return torch.FloatTensor(features)

def select_action(policy: FraudPolicy, obs: Any) -> Tuple[str, float, torch.Tensor]:
    """
    Sample an action from the policy given an observation.
    Returns: (action_str, log_prob, log_prob_tensor)
    """
    state_tensor = extract_features(obs)
    probs = policy(state_tensor)
    
    # Create a categorical distribution
    m = torch.distributions.Categorical(probs)
    action_idx = m.sample()
    log_prob = m.log_prob(action_idx)
    
    actions = ["APPROVE", "FLAG", "BLOCK"]
    return actions[action_idx.item()], probs[action_idx.item()].item(), log_prob
