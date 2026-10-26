"""
DQN Training Pipeline
Implements Double DQN / Standard DQN training with target networks,
experience replay, temporal difference learning, and validation tracking.
"""

import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from .config import (
    STATE_DIM,
    NUM_ACTIONS,
    ACTION_PRIORITY_MAP,
    ACTION_NAMES,
    CANONICAL_SYMPTOMS,
    DEFAULT_TRAIN_CONFIG,
    DQN_MODEL_PATH,
    DQN_META_PATH,
)
from .model import DQNNetwork
from .replay_buffer import ReplayBuffer
from .environment import ClinicalTriageEnv


class DQNTrainer:
    """
    DQN Trainer for Medical Symptom Triage.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = dict(DEFAULT_TRAIN_CONFIG)
        if config:
            self.config.update(config)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set reproducible random seeds
        seed = self.config.get("random_seed", 42)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Initialize Environment
        self.env = ClinicalTriageEnv(state_dim=self.config["state_dim"], seed=seed)
        self.eval_env = ClinicalTriageEnv(state_dim=self.config["state_dim"], seed=seed + 1000)

        # Policy & Target Networks
        self.policy_net = DQNNetwork(
            input_dim=self.config["state_dim"],
            output_dim=self.config["action_dim"],
            hidden_dims=self.config["hidden_dims"],
            dropout=self.config["dropout"],
        ).to(self.device)

        self.target_net = DQNNetwork(
            input_dim=self.config["state_dim"],
            output_dim=self.config["action_dim"],
            hidden_dims=self.config["hidden_dims"],
            dropout=self.config["dropout"],
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer & Loss Function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config["learning_rate"])
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stable gradient steps

        # Replay Buffer
        self.buffer = ReplayBuffer(
            capacity=self.config["buffer_size"],
            state_dim=self.config["state_dim"],
        )

        self.epsilon = self.config["epsilon_start"]
        self.training_history: Dict[str, List[float]] = {
            "episode": [],
            "reward": [],
            "moving_avg_reward": [],
            "loss": [],
            "epsilon": [],
            "accuracy": [],
            "safety_violations": [],
        }

    def fill_buffer(self, min_samples: int):
        """Populates the replay buffer with initial exploratory interactions."""
        print(f"[DQN Trainer] Pre-filling replay buffer with {min_samples} transitions...")
        while len(self.buffer) < min_samples:
            s = self.env.reset()
            a = self.env.rng.randint(0, self.config["action_dim"])
            next_s, r, done, _ = self.env.step(a)
            self.buffer.add(s, a, r, next_s, done)
        print(f"[DQN Trainer] Replay buffer initialized: {len(self.buffer)} samples.")

    def train_step(self) -> float:
        """Executes a single mini-batch Bellman TD update."""
        if len(self.buffer) < self.config["batch_size"]:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.config["batch_size"], device=self.device
        )

        self.policy_net.train()
        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN target computation
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.config["gamma"] * next_q_values

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        return float(loss.item())

    def evaluate(self, num_episodes: int = 200) -> Dict[str, float]:
        """Evaluates policy deterministically (epsilon=0) over validation episodes."""
        self.policy_net.eval()
        rewards = []
        correct_count = 0
        safety_violation_count = 0
        under_triage_count = 0
        over_triage_count = 0

        for _ in range(num_episodes):
            s = self.eval_env.reset()
            a = self.policy_net.select_action(s, epsilon=0.0, device=self.device)
            _, r, _, info = self.eval_env.step(a)

            rewards.append(r)
            if info["correct"]:
                correct_count += 1
            if info["critical_under_triage"]:
                safety_violation_count += 1
            if info["under_triaged"]:
                under_triage_count += 1
            if info["over_triaged"]:
                over_triage_count += 1

        return {
            "eval_reward_mean": float(np.mean(rewards)),
            "eval_reward_std": float(np.std(rewards)),
            "eval_accuracy": float(correct_count / num_episodes),
            "eval_safety_violations": float(safety_violation_count / num_episodes),
            "eval_under_triage_rate": float(under_triage_count / num_episodes),
            "eval_over_triage_rate": float(over_triage_count / num_episodes),
        }

    def train(self) -> Dict[str, Any]:
        """Runs the complete DQN training and validation loop."""
        print(f"\n==================================================")
        print(f"  Starting DQN Clinical Triage Training")
        print(f"  Episodes: {self.config['episodes']} | Device: {self.device}")
        print(f"  State Dim: {self.config['state_dim']} | Actions: {self.config['action_dim']}")
        print(f"==================================================\n")

        self.fill_buffer(self.config["min_buffer_size"])

        episodes = self.config["episodes"]
        target_update_freq = self.config["target_update_freq"]
        recent_rewards = []
        best_eval_reward = -float("inf")
        best_state_dict = None

        t0 = time.time()

        for ep in range(1, episodes + 1):
            s = self.env.reset()
            a = self.policy_net.select_action(s, epsilon=self.epsilon, device=self.device)
            next_s, r, done, info = self.env.step(a)

            self.buffer.add(s, a, r, next_s, done)
            loss = self.train_step()

            # Epsilon decay
            self.epsilon = max(self.config["epsilon_end"], self.epsilon * self.config["epsilon_decay"])

            # Sync Target Network
            if ep % target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            recent_rewards.append(r)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)

            mov_avg = float(np.mean(recent_rewards))

            # Record history
            if ep % 10 == 0 or ep == episodes:
                self.training_history["episode"].append(ep)
                self.training_history["reward"].append(float(r))
                self.training_history["moving_avg_reward"].append(mov_avg)
                self.training_history["loss"].append(float(loss))
                self.training_history["epsilon"].append(float(self.epsilon))

            # Periodic logging & validation
            if ep % 250 == 0 or ep == episodes:
                eval_metrics = self.evaluate(self.config.get("eval_episodes", 300))
                elapsed = time.time() - t0
                print(
                    f"Episode {ep:4d}/{episodes} | "
                    f"MovAvg Reward: {mov_avg:6.2f} | "
                    f"Eval Reward: {eval_metrics['eval_reward_mean']:6.2f} (±{eval_metrics['eval_reward_std']:.2f}) | "
                    f"Accuracy: {eval_metrics['eval_accuracy']*100:5.1f}% | "
                    f"Safety Violations: {eval_metrics['eval_safety_violations']*100:4.1f}% | "
                    f"Eps: {self.epsilon:.3f} | "
                    f"Time: {elapsed:.1f}s"
                )

                if eval_metrics["eval_reward_mean"] > best_eval_reward:
                    best_eval_reward = eval_metrics["eval_reward_mean"]
                    best_state_dict = self.policy_net.state_dict().copy()

        # Load best policy
        if best_state_dict is not None:
            self.policy_net.load_state_dict(best_state_dict)

        # Final comprehensive evaluation
        final_eval = self.evaluate(1000)
        print(f"\n[DQN Trainer] Training complete! Final 1000-episode evaluation:")
        print(f"  Mean Reward: {final_eval['eval_reward_mean']:.3f} ± {final_eval['eval_reward_std']:.3f}")
        print(f"  Triage Accuracy: {final_eval['eval_accuracy']*100:.2f}%")
        print(f"  Critical Safety Violation Rate: {final_eval['eval_safety_violations']*100:.2f}%")
        print(f"  Under-Triage Rate: {final_eval['eval_under_triage_rate']*100:.2f}%")
        print(f"  Over-Triage Rate: {final_eval['eval_over_triage_rate']*100:.2f}%")

        return final_eval

    def save_checkpoint(
        self,
        model_path: Path = DQN_MODEL_PATH,
        meta_path: Path = DQN_META_PATH,
        eval_metrics: Optional[Dict[str, float]] = None,
    ):
        """Saves dqn_model.pth checkpoint and dqn_meta.json metadata."""
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model checkpoint
        checkpoint = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.config["state_dim"],
            "action_dim": self.config["action_dim"],
            "hidden_dims": self.config["hidden_dims"],
            "dropout": self.config["dropout"],
            "version": "1.0.0",
        }
        torch.save(checkpoint, model_path)
        print(f"[DQN Trainer] Checkpoint successfully saved to: {model_path}")

        # Save metadata
        metadata = {
            "model_name": "dqn_triage",
            "version": "1.0.0",
            "state_dimension": self.config["state_dim"],
            "action_dimension": self.config["action_dim"],
            "canonical_symptoms_count": len(CANONICAL_SYMPTOMS),
            "canonical_symptoms": CANONICAL_SYMPTOMS,
            "actions": ACTION_NAMES,
            "action_priority_map": {str(k): v for k, v in ACTION_PRIORITY_MAP.items()},
            "training_config": {
                "episodes": self.config["episodes"],
                "learning_rate": self.config["learning_rate"],
                "gamma": self.config["gamma"],
                "batch_size": self.config["batch_size"],
                "buffer_size": self.config["buffer_size"],
                "target_update_freq": self.config["target_update_freq"],
                "random_seed": self.config["random_seed"],
            },
            "evaluation_metrics": eval_metrics or {},
            "environment": {
                "type": "simulated_clinical_triage_mdp",
                "is_synthetic": True,
                "taxonomy_source": "symptoms2.csv (389 symptom descriptors)",
                "urgency_framework": "Emergency Severity Index (ESI) Principles",
            },
            "disclaimer": "Research prototype — simulated clinical triage MDP. Not clinically validated for hospital deployment.",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[DQN Trainer] Metadata successfully saved to: {meta_path}")
