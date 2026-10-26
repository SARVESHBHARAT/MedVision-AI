"""
DQN Evaluation & Diagnostic Plotting
Evaluates the trained policy against clinical safety metrics and Bayes-optimal baselines.
Generates research artifacts in dqn_outputs/.
"""

import json
import numpy as np
import torch
try:
    import matplotlib
    matplotlib.use("Agg")  # Headless backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    matplotlib = None
    plt = None
from pathlib import Path
from typing import Dict, Any, Optional

from .config import (
    STATE_DIM,
    ACTION_PRIORITY_MAP,
    ACTION_NAMES,
    CLINICAL_REWARD_MATRIX,
    DQN_OUTPUT_DIR,
    DQN_MODEL_PATH,
    DQN_META_PATH,
)
from .model import DQNNetwork
from .environment import ClinicalTriageEnv


class DQNEvaluator:
    """
    Comprehensive RL evaluator for DQN Clinical Triage.
    """
    def __init__(
        self,
        model: Optional[DQNNetwork] = None,
        model_path: Optional[Path] = DQN_MODEL_PATH,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model
        else:
            if not model_path.exists():
                raise FileNotFoundError(f"DQN checkpoint not found at: {model_path}")
            ckpt = torch.load(model_path, map_location=self.device)
            state_dim = ckpt.get("state_dim", STATE_DIM)
            action_dim = ckpt.get("action_dim", 3)
            hidden_dims = ckpt.get("hidden_dims", [128, 128, 64])
            dropout = ckpt.get("dropout", 0.2)

            self.model = DQNNetwork(
                input_dim=state_dim,
                output_dim=action_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            ).to(self.device)

            state_dict = ckpt["policy_net"] if "policy_net" in ckpt else ckpt
            self.model.load_state_dict(state_dict)

        self.model.eval()
        self.env = ClinicalTriageEnv(state_dim=self.model.input_dim, seed=999)

    def run_benchmark(self, num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Runs comprehensive benchmark comparing DQN policy vs Bayes-Optimal vs Random baseline.
        """
        rewards_dqn, rewards_opt, rewards_rand = [], [], []
        confusion = np.zeros((3, 3), dtype=np.int64)  # [true_priority - 1, pred_priority - 1]
        action_counts = {0: 0, 1: 0, 2: 0}
        q_values_by_priority = {1: [], 2: [], 3: []}

        critical_under_triage = 0
        under_triage = 0
        over_triage = 0
        correct_count = 0

        for ep in range(num_episodes):
            s = self.env.reset()
            true_urgency = self.env.current_urgency  # 1, 2, 3
            opt_action = self.env.get_optimal_action(s)
            rand_action = self.env.rng.randint(0, 3)

            # DQN action & Q-values
            q_vals = self.model.predict_q_values(s, device=self.device)[0]
            dqn_action = int(np.argmax(q_vals))
            action_counts[dqn_action] += 1

            dqn_priority = ACTION_PRIORITY_MAP[dqn_action]
            q_values_by_priority[true_urgency].append(float(q_vals[dqn_action]))

            confusion[true_urgency - 1, dqn_priority - 1] += 1

            # Step DQN
            _, r_dqn, _, info = self.env.step(dqn_action)
            rewards_dqn.append(r_dqn)

            # Compute optimal reward
            r_opt = float(CLINICAL_REWARD_MATRIX[true_urgency][ACTION_PRIORITY_MAP[opt_action]])
            rewards_opt.append(r_opt)

            # Compute random reward
            r_rand = float(CLINICAL_REWARD_MATRIX[true_urgency][ACTION_PRIORITY_MAP[rand_action]])
            rewards_rand.append(r_rand)

            if info["correct"]:
                correct_count += 1
            if info["critical_under_triage"]:
                critical_under_triage += 1
            if info["under_triaged"]:
                under_triage += 1
            if info["over_triaged"]:
                over_triage += 1

        accuracy = correct_count / num_episodes
        safety_violation_rate = critical_under_triage / num_episodes
        under_triage_rate = under_triage / num_episodes
        over_triage_rate = over_triage / num_episodes

        results = {
            "num_episodes": num_episodes,
            "dqn_mean_reward": float(np.mean(rewards_dqn)),
            "dqn_reward_std": float(np.std(rewards_dqn)),
            "optimal_mean_reward": float(np.mean(rewards_opt)),
            "random_mean_reward": float(np.mean(rewards_rand)),
            "accuracy": float(accuracy),
            "critical_safety_violations": int(critical_under_triage),
            "critical_safety_violation_rate": float(safety_violation_rate),
            "under_triage_rate": float(under_triage_rate),
            "over_triage_rate": float(over_triage_rate),
            "action_distribution": {
                ACTION_NAMES[a]: int(count) for a, count in action_counts.items()
            },
            "confusion_matrix": confusion.tolist(),
            "mean_q_by_true_urgency": {
                f"Priority {k}": float(np.mean(v)) if len(v) > 0 else 0.0
                for k, v in q_values_by_priority.items()
            },
        }

        return results

    def generate_plots(
        self,
        benchmark_results: Dict[str, Any],
        training_history: Optional[Dict[str, Any]] = None,
        output_dir: Path = DQN_OUTPUT_DIR,
    ):
        """Generates research evaluation figures in dqn_outputs/."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Confusion Matrix Plot
        if HAS_MATPLOTLIB and plt is not None:
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = np.array(benchmark_results["confusion_matrix"])
            im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
            fig.colorbar(im, ax=ax)

            classes = ["Priority 1\n(Emergency)", "Priority 2\n(Urgent)", "Priority 3\n(Non-urgent)"]
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes, fontsize=9)
            ax.set_yticklabels(classes, fontsize=9)
            ax.set_xlabel("DQN Predicted Triage Priority", fontsize=11, fontweight="bold")
            ax.set_ylabel("True Clinical Urgency", fontsize=11, fontweight="bold")
            ax.set_title("DQN Clinical Triage Decision Matrix (1,000 Cases)", fontsize=11, fontweight="bold")

            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(
                        j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2.0 else "black",
                        fontsize=12, fontweight="bold",
                    )
            plt.tight_layout()
            cm_path = output_dir / "dqn_confusion_matrix.png"
            plt.savefig(cm_path, dpi=200)
            plt.close()
            print(f"[DQN Evaluator] Saved confusion matrix to: {cm_path}")

            # 2. Training Convergence Curves (if available)
            if training_history and "reward" in training_history and len(training_history["reward"]) > 0:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
                episodes = training_history.get("episode", list(range(len(training_history["reward"]))))

                ax1.plot(episodes, training_history["reward"], alpha=0.3, color="#0099ff", label="Episode Reward")
                if "moving_avg_reward" in training_history:
                    ax1.plot(episodes, training_history["moving_avg_reward"], color="#00d4ff", linewidth=2.0, label="100-Ep Moving Average")
                ax1.set_ylabel("Clinical Reward", fontsize=10, fontweight="bold")
                ax1.set_title("DQN Clinical Triage Learning Convergence", fontsize=12, fontweight="bold")
                ax1.grid(True, linestyle="--", alpha=0.5)
                ax1.legend(loc="lower right")

                if "loss" in training_history:
                    ax2.plot(episodes, training_history["loss"], color="#ff8800", linewidth=1.5, label="TD Huber Loss")
                    ax2.set_ylabel("Loss", fontsize=10, fontweight="bold")
                    ax2.set_xlabel("Training Episodes", fontsize=10, fontweight="bold")
                    ax2.set_yscale("log")
                    ax2.grid(True, linestyle="--", alpha=0.5)
                    ax2.legend(loc="upper right")

                plt.tight_layout()
                curve_path = output_dir / "dqn_training_curves.png"
                plt.savefig(curve_path, dpi=200)
                plt.close()
                print(f"[DQN Evaluator] Saved training curves to: {curve_path}")

        # 3. Save benchmark summary JSON
        summary_path = output_dir / "dqn_evaluation_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)
        print(f"[DQN Evaluator] Saved evaluation metrics to: {summary_path}")
