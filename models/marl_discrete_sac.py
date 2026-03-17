"""MARL Discrete SAC agent — combines encoder, actor, critics.

This module glues together the graph encoder, shared actor, and
centralized twin critics into a single agent interface that is
compatible with TorchRL's TensorDict-based loss computation.
"""

from __future__ import annotations

import copy
import math

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
        # Discrete SAC target entropy should be near log(|A|), not negative.
        self.target_entropy = 0.98 * math.log(float(max(num_actions, 2)))

        self.tau = tau

    # ==================================================================
    # Forward helpers
    # ==================================================================
    @staticmethod
    def _pool_agent_latents(
        z_nodes: Tensor,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> Tensor:
        """Pool node embeddings into per-agent embeddings.

        If no mapping is provided, this is an identity map and assumes one
        graph node per RL agent.
        """
        if agent_node_indices is None:
            return z_nodes

        if agent_node_mask is None:
            agent_node_mask = agent_node_indices >= 0

        if z_nodes.dim() == 2:
            z_nodes = z_nodes.unsqueeze(0)
            squeeze_result = True
        else:
            squeeze_result = False

        if agent_node_indices.dim() == 2:
            agent_node_indices = agent_node_indices.unsqueeze(0).expand(
                z_nodes.shape[0], -1, -1
            )
        if agent_node_mask.dim() == 2:
            agent_node_mask = agent_node_mask.unsqueeze(0).expand(
                z_nodes.shape[0], -1, -1
            )

        safe_indices = agent_node_indices.clamp_min(0)
        num_agents = safe_indices.shape[1]
        latent_dim = z_nodes.shape[-1]

        expanded_nodes = z_nodes.unsqueeze(1).expand(-1, num_agents, -1, -1)
        gather_idx = safe_indices.unsqueeze(-1).expand(-1, -1, -1, latent_dim)
        selected = torch.gather(expanded_nodes, 2, gather_idx)

        weights = agent_node_mask.unsqueeze(-1).to(selected.dtype)
        pooled = (selected * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)

        if squeeze_result:
            return pooled[0]
        return pooled

    def encode(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> Tensor:
        """Run encoder and optionally pool graph nodes back to RL agents."""
        z_nodes = self.encoder(obs, edge_index, edge_attr)
        return self._pool_agent_latents(z_nodes, agent_node_indices, agent_node_mask)

    def select_action(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        action_mask: Tensor | None = None,
        deterministic: bool = False,
        *,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """End-to-end: obs → encoder → actor → (action, log_prob)."""
        z = self.encode(
            obs,
            edge_index,
            edge_attr,
            agent_node_indices=agent_node_indices,
            agent_node_mask=agent_node_mask,
        )
        return self.actor.get_action(z, action_mask, deterministic)

    def get_action_probs(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        action_mask: Tensor | None = None,
        *,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(z, action_probs, log_action_probs)``."""
        z = self.encode(
            obs,
            edge_index,
            edge_attr,
            agent_node_indices=agent_node_indices,
            agent_node_mask=agent_node_mask,
        )
        action_probs, log_action_probs = self.actor.get_action_probs(z, action_mask)
        return z, action_probs, log_action_probs

    def critic_values(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute online twin Q-values: (q1, q2) each ``[n_agents, num_actions]``."""
        z = self.encode(
            obs,
            edge_index,
            edge_attr,
            agent_node_indices=agent_node_indices,
            agent_node_mask=agent_node_mask,
        )
        return self.critic(z)

    def target_critic_values(
        self,
        obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute target twin Q-values (no grad)."""
        with torch.no_grad():
            z = self.encode(
                obs,
                edge_index,
                edge_attr,
                agent_node_indices=agent_node_indices,
                agent_node_mask=agent_node_mask,
            )
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
