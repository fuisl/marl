"""MARL Discrete SAC agent — combines encoder, actor, critics.

This module glues together the graph encoder, shared actor, and
centralized twin critics into a single agent interface that is
compatible with TorchRL's TensorDict-based loss computation.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor


from models.actor import SharedDiscreteActor
from models.critic import CentralizedTwinCritic
from models.graph_encoder import GraphEncoder


class MARLDiscreteSAC(nn.Module):
    """Full MARL Discrete SAC agent.

    Owns:
        - ``encoder``       : GATv2-based graph encoder
        - ``actor``         : shared discrete policy
        - ``critic``        : twin Q-networks (online)
        - ``target_critic`` : twin Q-networks (target, EMA-updated)
        - ``log_alpha``     : learnable entropy temperature

    Parameters
    ----------
    obs_dim : int
        Per-agent raw observation dimension.
    num_actions : int
        Number of discrete phase actions.
    encoder_cfg : dict
        Kwargs for :class:`GraphEncoder`.
    actor_cfg : dict
        Kwargs for :class:`SharedDiscreteActor`.
    critic_cfg : dict
        Kwargs for :class:`CentralizedTwinCritic`.
    init_alpha : float
        Initial entropy coefficient (before log).
    tau : float
        Soft-update coefficient for target critic.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        encoder_cfg: dict | None = None,
        actor_cfg: dict | None = None,
        critic_cfg: dict | None = None,
        init_alpha: float = 0.2,
        tau: float = 0.005,
    ) -> None:
        super().__init__()
        encoder_cfg = encoder_cfg or {}
        actor_cfg = actor_cfg or {}
        critic_cfg = critic_cfg or {}

        latent_dim = encoder_cfg.get("out_dim", 64)

        # --- Encoder ---
        self.encoder = GraphEncoder(in_dim=obs_dim, **encoder_cfg)

        # --- Actor ---
        self.actor = SharedDiscreteActor(
            latent_dim=latent_dim, num_actions=num_actions, **actor_cfg
        )

        # --- Critics ---
        self.critic = CentralizedTwinCritic(
            latent_dim=latent_dim, num_actions=num_actions, **critic_cfg
        )
        self.target_critic = copy.deepcopy(self.critic)
        # Freeze target parameters
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # --- Entropy temperature ---
        self.log_alpha = nn.Parameter(
            torch.tensor(init_alpha, dtype=torch.float32).log()
        )
        self.target_entropy = -float(num_actions) * 0.98  # heuristic

        self.tau = tau

    # ==================================================================
    # Forward helpers
    # ==================================================================
    def encode(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Run encoder. ``obs`` shape: ``[n_agents, obs_dim]``."""
        return self.encoder(obs, edge_index, edge_attr)

    def select_action(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """End-to-end: obs → encoder → actor → (action, log_prob)."""
        z = self.encode(obs, edge_index, edge_attr)
        return self.actor.get_action(z, action_mask, deterministic)

    def get_action_probs(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(z, action_probs, log_action_probs)``."""
        z = self.encode(obs, edge_index, edge_attr)
        action_probs, log_action_probs = self.actor.get_action_probs(z, action_mask)
        return z, action_probs, log_action_probs

    def critic_values(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute online twin Q-values: (q1, q2) each ``[n_agents, num_actions]``."""
        z = self.encode(obs, edge_index, edge_attr)
        return self.critic(z)

    def target_critic_values(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute target twin Q-values (no grad)."""
        with torch.no_grad():
            z = self.encode(obs, edge_index, edge_attr)
            return self.target_critic(z)

    # ==================================================================
    # Target update
    # ==================================================================
    @torch.no_grad()
    def soft_update_target(self) -> None:
        """Polyak-average update of target critic parameters."""
        for tp, op in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)

    # ==================================================================
    # Property helpers
    # ==================================================================
    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()
