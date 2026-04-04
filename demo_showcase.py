import time
import json
from server.fraud_environment import FraudEnvironment
from models import FraudAction

def run_showcase():
    print("\n" + "="*60)
    print("🚀 OPENENV FRAUD DETECTION DEMO SHOWCASE 🚀")
    print("="*60 + "\n")

    env = FraudEnvironment(task="easy")

    for ep in range(1, 3):
        print(f"\n[ EPISODE {ep} STARTING ]")
        print("-" * 40)
        
        obs = env.reset()
        done = False
        step = 0
        
        # Limit to 3 steps per episode for demo readability
        while not done and step < 3:
            step += 1
            
            # 1. Show Transaction Generated
            print(f"🔹 Transaction Generated: ID={obs.transaction_id} | Amount=${obs.amount:.2f} | Country={obs.country} | Merchant={obs.merchant_type} | Velocity={obs.transaction_velocity}")
            
            # Simple heuristic for dummy agent
            decision = "APPROVE"
            if obs.geo_risk_score > 0.6 or obs.amount_zscore > 2.0:
                decision = "FLAG"
            if obs.device_consistency < 0.2:
                decision = "BLOCK"
                
            action = FraudAction(decision=decision, confidence=0.85, reasoning="Demo heuristic rules.")
            
            # 2. Show Action Taken
            print(f"🔸 Action Taken: {action.decision} (Confidence: {action.confidence})")
            
            # Step the environment
            obs, reward, done, info = env.step(action)
            
            # 3. Show Reward
            correctness_flag = "✅" if info['correctness'] == "correct" else "❌"
            print(f"💰 Reward: {reward:+.3f} ({correctness_flag} Truth context: {info['reward_label']})")
            print("-" * 40)
            time.sleep(0.5)
            
        print(f"🏁 Episode {ep} completed. Cumulative Reward: {env.state.total_reward:.3f}\n")

if __name__ == "__main__":
    run_showcase()
