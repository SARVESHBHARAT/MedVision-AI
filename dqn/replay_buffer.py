"""
Experience Replay Buffer for DQN
Stores transitions (s, a, r, s', done) and samples mini-batches for temporal-difference updates.
"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer using continuous numpy arrays for optimal performance.
    """
    def __init__(self, capacity: int = 50000, state_dim: int = 50):
        self.capacity = capacity
        self.state_dim = state_dim
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add transition to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions uniformly at random."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        states_b = torch.as_tensor(self.states[idxs], device=device, dtype=torch.float32)
        actions_b = torch.as_tensor(self.actions[idxs], device=device, dtype=torch.long)
        rewards_b = torch.as_tensor(self.rewards[idxs], device=device, dtype=torch.float32)
        next_states_b = torch.as_tensor(self.next_states[idxs], device=device, dtype=torch.float32)
        dones_b = torch.as_tensor(self.dones[idxs], device=device, dtype=torch.float32)

        return states_b, actions_b, rewards_b, next_states_b, dones_b

    def __len__(self) -> int:
        return self.size
