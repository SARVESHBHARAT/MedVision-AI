"""
MedVision AI — DQN Triage Standalone Evaluator
Loads dqn_model.pth and runs benchmark evaluations across simulated clinical test cases.

Usage:
    python evaluate_dqn.py [--episodes 1000]
"""

import argparse
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dqn import DQNEvaluator, DQN_MODEL_PATH, DQN_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedVision AI DQN Triage Model")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of benchmark test episodes")
    parser.add_argument("--model-path", type=str, default=str(DQN_MODEL_PATH), help="Path to dqn_model.pth")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" [MedVision AI] DQN BENCHMARK EVALUATION")
    print("=" * 60)
    print(f"  Model Path: {args.model_path}")
    print(f"  Episodes:   {args.episodes}")
    print("=" * 60 + "\n")

    evaluator = DQNEvaluator(model_path=Path(args.model_path))
    results = evaluator.run_benchmark(num_episodes=args.episodes)
    evaluator.generate_plots(benchmark_results=results, output_dir=DQN_OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(" [SUMMARY] EVALUATION BENCHMARK METRICS")
    print("=" * 60)
    print(f"  Episodes:                       {results['num_episodes']}")
    print(f"  Accuracy (Optimal Action):      {results['accuracy']*100:.2f}%")
    print(f"  DQN Mean Reward:                {results['dqn_mean_reward']:.3f} ± {results['dqn_reward_std']:.3f}")
    print(f"  Optimal Policy Mean Reward:     {results['optimal_mean_reward']:.3f}")
    print(f"  Random Policy Mean Reward:      {results['random_mean_reward']:.3f}")
    print(f"  Critical Safety Violation Rate: {results['critical_safety_violation_rate']*100:.2f}%")
    print(f"  Under-Triage Rate:              {results['under_triage_rate']*100:.2f}%")
    print(f"  Over-Triage Rate:               {results['over_triage_rate']*100:.2f}%")
    print(f"  Action Distribution:            {results['action_distribution']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
