"""
MedVision AI — DQN Triage Training Script
Trains the Deep Q-Network for clinical symptom triage, evaluates convergence and safety metrics,
saves dqn_model.pth and dqn_meta.json, and generates research evaluation figures.

Usage:
    python train_dqn.py [--episodes 2500] [--lr 1e-3] [--seed 42]
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

from dqn import (
    DQNTrainer,
    DQNEvaluator,
    DEFAULT_TRAIN_CONFIG,
    DQN_MODEL_PATH,
    DQN_META_PATH,
    DQN_OUTPUT_DIR,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedVision AI DQN Triage Model")
    parser.add_argument("--episodes", type=int, default=2500, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor gamma")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--target-update", type=int, default=10, help="Target network update frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--eval-episodes", type=int, default=500, help="Validation episodes")
    return parser.parse_args()


def main():
    args = parse_args()

    config = dict(DEFAULT_TRAIN_CONFIG)
    config.update({
        "episodes": args.episodes,
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "target_update_freq": args.target_update,
        "random_seed": args.seed,
        "eval_episodes": args.eval_episodes,
    })

    print("\n" + "=" * 60)
    print(" [MedVision AI] REINFORCEMENT LEARNING DQN TRIAGE TRAINER")
    print("=" * 60)
    print(f"  Configuration:")
    print(f"    - Episodes:           {config['episodes']}")
    print(f"    - State Dimension:    {config['state_dim']}")
    print(f"    - Action Dimension:   {config['action_dim']} (Priority 1, 2, 3)")
    print(f"    - Learning Rate:      {config['learning_rate']}")
    print(f"    - Discount (Gamma):   {config['gamma']}")
    print(f"    - Target Update:      Every {config['target_update_freq']} episodes")
    print(f"    - Random Seed:        {config['random_seed']}")
    print(f"    - Environment:        {config['environment_version']} (Simulated)")
    print("=" * 60 + "\n")

    trainer = DQNTrainer(config=config)
    eval_metrics = trainer.train()

    # Save final verified model and metadata
    trainer.save_checkpoint(
        model_path=DQN_MODEL_PATH,
        meta_path=DQN_META_PATH,
        eval_metrics=eval_metrics,
    )

    # Run in-depth evaluation and generate plots
    print("\n[DQN Pipeline] Running 1,000-episode diagnostic benchmark...")
    evaluator = DQNEvaluator(model=trainer.policy_net, device=trainer.device)
    benchmark_results = evaluator.run_benchmark(num_episodes=1000)
    evaluator.generate_plots(
        benchmark_results=benchmark_results,
        training_history=trainer.training_history,
        output_dir=DQN_OUTPUT_DIR,
    )

    print("\n" + "=" * 60)
    print(" [OK] DQN TRAINING & BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"  Checkpoint:        {DQN_MODEL_PATH}")
    print(f"  Metadata:          {DQN_META_PATH}")
    print(f"  Diagnostic Output: {DQN_OUTPUT_DIR}")
    print(f"  Accuracy:          {benchmark_results['accuracy']*100:.2f}%")
    print(f"  DQN Mean Reward:   {benchmark_results['dqn_mean_reward']:.3f} (Optimal: {benchmark_results['optimal_mean_reward']:.3f}, Random: {benchmark_results['random_mean_reward']:.3f})")
    print(f"  Critical Safety Violation Rate: {benchmark_results['critical_safety_violation_rate']*100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
