"""Centralized twin Q-critics for CTDE Discrete SAC.

Each critic receives:
  - local latent ``z_i``,
  - chosen action ``a_i`` (one-hot or index),
  - pooled graph context ``c = mean(z)`` (centralized information).

Returns Q-values for all discrete actions (no action input needed for
discrete SAC — the critic outputs Q(s, ·) for every action).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QNetwork(nn.Module):
    """Single Q-network: maps (z_i, context) → Q-values for all actions.

    Parameters
    ----------
    latent_dim : int
        Per-agent embedding dimension.
    context_dim : int
        Dimension of the pooled graph context.
    num_actions : int
        Number of discrete actions.
    hidden_dim : int
        Width of hidden layers.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        context_dim: int = 64,
        num_actions: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, z_local: Tensor, context: Tensor) -> Tensor:
        """
        Parameters
        ----------
        z_local : Tensor  ``[..., latent_dim]``
        context : Tensor  ``[..., context_dim]``

        Returns
        -------
        q_values : Tensor  ``[..., num_actions]``
        """
        x = torch.cat([z_local, context], dim=-1)
        return self.net(x)


class CentralizedTwinCritic(nn.Module):
    """Twin Q-networks with mean-pooling centralized context.

    Parameters
    ----------
    latent_dim : int
        Per-agent latent dimension (output of ``GraphEncoder``).
    num_actions : int
        Number of discrete actions.
    hidden_dim : int
        Hidden width for each Q-network.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_actions: int = 4,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        context_dim = latent_dim  # pooled graph context has same dim

        self.q1 = QNetwork(latent_dim, context_dim, num_actions, hidden_dim)
        self.q2 = QNetwork(latent_dim, context_dim, num_actions, hidden_dim)

    def forward(
        self, z: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute twin Q-values for all agents.

        Parameters
        ----------
        z : Tensor  ``[n_agents, latent_dim]`` or ``[B, n_agents, latent_dim]``

        Returns
        -------
        q1_values, q2_values : Tensor  ``[..., n_agents, num_actions]``
        """
        context = self._pool(z)  # [..., latent_dim]
        # Expand context to per-agent: [..., n_agents, latent_dim]
        if z.dim() == 2:
            # [n_agents, latent_dim]
            context = context.unsqueeze(0).expand_as(z)
        else:
            # [B, n_agents, latent_dim]
            context = context.unsqueeze(-2).expand_as(z)

        q1 = self.q1(z, context)
        q2 = self.q2(z, context)
        return q1, q2

    @staticmethod
    def _pool(z: Tensor) -> Tensor:
        """Mean-pool over the agent dimension.

        Parameters
        ----------
        z : Tensor  ``[n_agents, d]`` or ``[B, n_agents, d]``

        Returns
        -------
        pooled : Tensor  ``[d]`` or ``[B, d]``
        """
        if z.dim() == 2:
            return z.mean(dim=0)
        return z.mean(dim=-2)
