"""
Fraud Detection OpenEnv – Baseline Agent
Uses OpenAI API to drive a GPT-based agent through all three tasks.
Reads OPENAI_API_KEY from environment.  Produces deterministic results via seed.

Usage:
    OPENAI_API_KEY=sk-... python baseline.py
    OPENAI_API_KEY=sk-... python baseline.py --task easy
    OPENAI_API_KEY=sk-... python baseline.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Load environment variables
load_dotenv()

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent))

from server.fraud_environment import FraudEnvironment
from server.graders import grade_episode

console = Console()

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FraudBaselineAgent:
    """
    GPT-powered fraud detection agent.
    Uses deterministic sampling (seed, temperature=0) for reproducibility.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-72B-Instruct",
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        api_key = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        if not api_key:
            raise EnvironmentError(
                "API Key is not set. "
                "Export it or add it to a .env file: HF_TOKEN=hf_... or OPENAI_API_KEY=hf_..."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._call_count = 0

    def decide(self, observation_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Make a decision given an observation dict."""
        user_message = TRANSACTION_TEMPLATE.format(**observation_dict)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=self.temperature,
                    max_tokens=256,
                    seed=42,    # Deterministic completions
                    response_format={"type": "json_object"},
                )
                self._call_count += 1
                raw = response.choices[0].message.content
                parsed = json.loads(raw)

                # Validate
                decision = parsed.get("decision", "APPROVE").upper()
                if decision not in ("APPROVE", "FLAG", "BLOCK"):
                    decision = "APPROVE"

                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                return {
                    "decision":   decision,
                    "confidence": confidence,
                    "reasoning":  parsed.get("reasoning", ""),
                }

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(1.0)
                    continue
                # Fallback: safe default
                return {"decision": "FLAG", "confidence": 0.3, "reasoning": f"Parse error: {e}"}

            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2.0 ** attempt)
                    continue
                return {"decision": "FLAG", "confidence": 0.3, "reasoning": f"API error: {e}"}

        return {"decision": "FLAG", "confidence": 0.3, "reasoning": "Max retries exceeded"}

    @property
    def total_api_calls(self) -> int:
        return self._call_count


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_task(agent: FraudBaselineAgent, task: str) -> Dict[str, Any]:
    """Run the agent through a single task and return results."""
    console.print(f"\n[bold cyan]━━━ Task: {task.upper()} ━━━[/bold cyan]")

    env = FraudEnvironment(task=task)
    obs = env.reset()

    step_results = []
    cumulative_reward = 0.0
    done = False
    info = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task_progress = progress.add_task(
            f"[yellow]Running {task} task...", total=env._config.max_steps
        )

        while not done:
            obs_dict = {
                "transaction_id":     obs.transaction_id,
                "amount":             obs.amount,
                "amount_zscore":      obs.amount_zscore,
                "country":            obs.country,
                "geo_risk_score":     obs.geo_risk_score,
                "merchant_type":      obs.merchant_type,
                "merchant_risk_score":obs.merchant_risk_score,
                "device_type":        obs.device_type,
                "device_consistency": obs.device_consistency,
                "user_age":           obs.user_age,
                "transaction_velocity": obs.transaction_velocity,
                "is_night":           obs.is_night,
                "account_age_days":   obs.account_age_days,
                "step":               obs.step,
                "max_steps":          obs.max_steps,
                "fraud_rate_so_far":  obs.fraud_rate_so_far,
                "block_rate_so_far":  obs.block_rate_so_far,
                "cumulative_reward":  obs.cumulative_reward,
            }

            action_data = agent.decide(obs_dict)
            from models import FraudAction
            action = FraudAction(
                decision=action_data["decision"],
                confidence=action_data["confidence"],
                reasoning=action_data["reasoning"],
            )

            step_obs = env.step(action)
            # FraudEnvironment returns a single FraudObservation (not a Gym tuple)
            obs    = step_obs
            reward = getattr(step_obs, "reward", 0.0)
            done   = getattr(step_obs, "done", False)
            info   = getattr(step_obs, "metadata", {}) or {}
            cumulative_reward += reward
            step_results.append({
                "decision":   action.decision,
                "reward":     reward,
                "reasoning":  action.reasoning[:60] if action.reasoning else "",
            })

            progress.update(task_progress, advance=1)

    # Final grade
    grade = info.get("grade", {})
    summary = info.get("summary", {})

    return {
        "task":               task,
        "steps":              len(step_results),
        "cumulative_reward":  round(cumulative_reward, 4),
        "grade":              grade,
        "summary":            summary,
        "actions_taken":      step_results,
    }


def print_results(results: List[Dict[str, Any]]) -> None:
    """Pretty-print a summary table of all task results."""
    console.print("\n")
    console.print(Panel.fit("🏦  Fraud Detection Baseline Agent – Evaluation Results", style="bold green"))

    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Task",         style="cyan",  min_width=10)
    table.add_column("Steps",        justify="right")
    table.add_column("Cum. Reward",  justify="right")
    table.add_column("Score [0-1]",  justify="right", style="bold")
    table.add_column("Grader",       style="dim")
    table.add_column("Accuracy",     justify="right")
    table.add_column("Block Rate",   justify="right")
    table.add_column("Details",      style="dim", min_width=30)

    for r in results:
        grade = r["grade"]
        summary = r["summary"]
        score   = grade.get("score", 0.0)
        grader  = grade.get("grader", "-")
        details = grade.get("details", "")[:50]
        accuracy   = f"{summary.get('accuracy', 0.0):.1%}" if summary else "-"
        block_rate = f"{summary.get('block_rate', 0.0):.1%}" if summary else "-"

        score_str = f"{score:.4f}"
        score_color = "green" if score >= 0.7 else "yellow" if score >= 0.4 else "red"

        table.add_row(
            r["task"].upper(),
            str(r["steps"]),
            f"{r['cumulative_reward']:+.3f}",
            f"[{score_color}]{score_str}[/{score_color}]",
            grader,
            accuracy,
            block_rate,
            details,
        )

    console.print(table)

    # Summary panel
    avg_score = np.mean([r["grade"].get("score", 0.0) for r in results])
    console.print(
        Panel(
            f"[bold white]Overall Average Score:[/bold white] "
            f"[bold {'green' if avg_score >= 0.7 else 'yellow'}]{avg_score:.4f}[/bold {'green' if avg_score >= 0.7 else 'yellow'}]"
            f"\n[dim]Scores are deterministic (seed=42) and reproducible.[/dim]",
            title="📊 Final Verdict",
            expand=False,
        )
    )

    # Save results to file
    out_path = Path("outputs/evals/baseline_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[dim]Full results saved to: {out_path}[/dim]\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fraud Detection OpenEnv – Baseline Agent"
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Run a single task (default: run all three)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-72B-Instruct)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Explicitly run all three tasks",
    )
    args = parser.parse_args()

    tasks_to_run: List[str]
    if args.task:
        tasks_to_run = [args.task]
    else:
        tasks_to_run = ["easy", "medium", "hard"]

    console.print(Panel.fit(
        f"[bold]🏦 Fraud Detection OpenEnv – Baseline Agent[/bold]\n"
        f"Model: [cyan]{args.model}[/cyan] | Tasks: [yellow]{', '.join(tasks_to_run)}[/yellow]",
        style="blue"
    ))

    try:
        agent = FraudBaselineAgent(model=args.model)
    except EnvironmentError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)

    all_results = []
    for task in tasks_to_run:
        try:
            result = run_task(agent, task)
            all_results.append(result)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Task {task} failed: {e}[/red]")
            all_results.append({
                "task": task,
                "error": str(e),
                "grade": {"score": 0.0},
                "summary": {},
                "steps": 0,
                "cumulative_reward": 0.0,
                "actions_taken": [],
            })

    if all_results:
        print_results(all_results)
        console.print(f"[dim]Total API calls: {agent.total_api_calls}[/dim]")


if __name__ == "__main__":
    main()
