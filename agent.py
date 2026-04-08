try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn:
        class Module: pass
    F = None

from typing import List, Dict, Any, Tuple, Optional

class FraudPolicy(nn.Module if TORCH_AVAILABLE else object):
    """
    STABLE RESOURCE-EFFICIENT BRAIN:
    - Input: 19 Features (Synchronized with current extractor)
    """
    def __init__(self, input_dim: int = 19, hidden_dim: int = 128):
        if not TORCH_AVAILABLE:
            return
        super(FraudPolicy, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 3) 
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available")
        # Standardize for both vector and batch input
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.norm1(self.input_layer(x)))
        x = F.relu(self.norm2(self.hidden_layer(x)))
        logits = self.output_head(x)
        return F.softmax(logits, dim=-1)

def extract_features(obs: Any, llm_decision: str = "APPROVE") -> 'torch.Tensor':
    """
    EXTRACTS EXACTLY 19 FEATURES:
    [8 tx signals] + [3 episode signals] + [3 llm one-hots] + [4 device one-hots] + [1 reward signal]
    """
    if not TORCH_AVAILABLE:
        return None
        
    # 8 Base signals
    amount = torch.tanh(torch.tensor([getattr(obs, 'amount', 0.0) / 1000.0]))
    user_age = torch.tensor([getattr(obs, 'user_age', 30) / 100.0])
    velocity = torch.tensor([min(getattr(obs, 'transaction_velocity', 1) / 20.0, 1.0)])
    acc_age = torch.tensor([min(getattr(obs, 'account_age_days', 365) / 3650.0, 1.0)])
    geo_risk = torch.tensor([getattr(obs, 'geo_risk_score', 0.0)])
    m_risk = torch.tensor([getattr(obs, 'merchant_risk_score', 0.0)])
    device_cons = torch.tensor([getattr(obs, 'device_consistency', 1.0)])
    z_score = torch.tensor([max(-5.0, min(5.0, getattr(obs, 'amount_zscore', 0.0))) / 5.0])
    
    # 3 Episode signals
    p = getattr(obs, 'step', 0) / max(getattr(obs, 'max_steps', 50), 1)
    progress = torch.tensor([p])
    block_rate = torch.tensor([getattr(obs, 'block_rate_so_far', 0.0)])
    danger = torch.tensor([1.0 if getattr(obs, 'block_rate_so_far', 0.0) > 0.20 else 0.0])
    
    # 3 LLM One-Hots
    llm_map = {"APPROVE": [1.,0.,0.], "FLAG": [0.,1.,0.], "BLOCK": [0.,0.,1.]}
    llm_feat = torch.tensor(llm_map.get(str(llm_decision).upper(), [1.,0.,0.]))

    # 4 Device One-Hots
    dt_map = {"mobile": [1,0,0,0], "desktop": [0,1,0,0], "tablet": [0,0,1,0], "unknown": [0,0,0,1]}
    device_feat = torch.tensor(dt_map.get(str(getattr(obs, 'device_type', 'unknown')).lower(), [0,0,0,1]), dtype=torch.float32)

    # 1 Performance Signal
    avg_reward = torch.tensor([getattr(obs, 'cumulative_reward', 0.0) / (getattr(obs, 'step', 0) + 1)])

    # Total = 19
    return torch.cat([
        amount, user_age, velocity, acc_age, geo_risk, m_risk, device_cons, z_score,
        progress, block_rate, danger, llm_feat, device_feat, avg_reward
    ]).float()

def select_action(policy: FraudPolicy, obs: Any, llm_decision: str, deterministic: bool = False) -> Tuple[str, float, Any]:
    if not TORCH_AVAILABLE or policy is None:
        # Fallback to pure LLM decision with high confidence
        return llm_decision.upper(), 0.95, None
        
    state_tensor = extract_features(obs, llm_decision)
    probs = policy(state_tensor)
    
    if deterministic:
        action_idx = torch.argmax(probs).item()
        log_prob = torch.log(probs[0, action_idx] + 1e-8)
    else:
        m = torch.distributions.Categorical(probs)
        action_idx = m.sample()
        log_prob = m.log_prob(action_idx)
        action_idx = action_idx.item()
    
    actions = ["APPROVE", "FLAG", "BLOCK"]
    return actions[action_idx], float(probs[0, action_idx].item()), log_prob
