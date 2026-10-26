"""
Deep Q-Network (DQN) Neural Architecture
PyTorch implementation of the Q-Value approximation network for medical triage.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, List


class DQNNetwork(nn.Module):
    """
    Multi-Layer Perceptron Q-Network mapping patient state vector to action Q-values.
    Q(s, a): Expected cumulative discounted clinical reward for triage action a in state s.
    """
    def __init__(
        self,
        input_dim: int = 50,
        output_dim: int = 3,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final hidden layer before output
        last_h_dim = hidden_dims[-1]
        layers.append(nn.Linear(in_dim, last_h_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(last_h_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns Q-values for all discrete triage actions [batch_size, num_actions]."""
        return self.network(x)

    @torch.no_grad()
    def predict_q_values(self, state: Union[np.ndarray, torch.Tensor], device: torch.device = None) -> np.ndarray:
        """Evaluates Q-values deterministically in eval mode."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        if isinstance(state, np.ndarray):
            state_t = torch.FloatTensor(state).to(device)
        else:
            state_t = state.to(device)

        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)

        q_vals = self.forward(state_t)
        return q_vals.cpu().numpy()

    @torch.no_grad()
    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        epsilon: float = 0.0,
        device: torch.device = None
    ) -> int:
        """Epsilon-greedy action selection."""
        if epsilon > 0.0 and np.random.rand() < epsilon:
            return int(np.random.randint(0, self.output_dim))

        q_vals = self.predict_q_values(state, device=device)
        return int(np.argmax(q_vals[0]))
