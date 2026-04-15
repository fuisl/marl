"""Local-Neighbor GAT Discrete SAC agent — combines local encoding with graph neighbor encoding.

This module implements the architecture from the research diagram: local observations
are encoded via a small MLP, neighbor information is aggregated via GATv2, then the
two representations are fused before feeding to the shared actor and centralized critics.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv

from models.actor import SharedDiscreteActor
from models.critic import CentralizedTwinCritic


class LocalEncoder(nn.Module):
    """Local observation encoder: MLP that processes only the agent's own observation.
    
    Parameters
    ----------
    in_dim : int
        Per-agent observation dimension.
    hidden_dim : int
        Hidden width.
    out_dim : int
        Output latent dimension.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 48,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ELU(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Encode local observations.
        
        Parameters
        ----------
        x : Tensor  ``[n_agents, in_dim]`` or ``[B, n_agents, in_dim]``
        
        Returns
        -------
        z_local : Tensor  ``[..., out_dim]``
        """
        return self.net(x)


class NeighborEncoder(nn.Module):
    """Neighbor aggregation encoder: GATv2 over the road graph.
    
    Parameters
    ----------
    in_dim : int
        Per-node feature dimension.
    hidden_dim : int
        Hidden channel width.
    out_dim : int
        Output latent dimension per node.
    heads : int
        Number of attention heads.
    edge_dim : int | None
        Dimension of edge attributes.
    dropout : float
        Dropout in attention.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 48,
        heads: int = 2,
        edge_dim: int | None = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        self.conv1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,  # output: [N, hidden_dim]
        )
        
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ELU()
    
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Encode neighbor information via graph attention.
        
        Parameters
        ----------
        x : Tensor  ``[n_agents, in_dim]``
        edge_index : Tensor  ``[2, E]``
        edge_attr : Tensor | None  ``[E, edge_dim]``
        
        Returns
        -------
        z_neighbor : Tensor  ``[n_agents, out_dim]``
        """
        h = self.act(self.input_proj(x))  # [N, hidden]
        h = self.act(self.conv1(h, edge_index, edge_attr))  # [N, hidden]
        z = self.output_proj(h)  # [N, out_dim]
        return z


class FusionMLP(nn.Module):
    """Fusion MLP: combine local and neighbor latents.
    
    Parameters
    ----------
    latent_dim : int
        Dimension of each input (local and neighbor).
    hidden_dim : int
        Hidden width.
    out_dim : int
        Output latent dimension (fed to actor/critic).
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 96,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ELU(),
        )
    
    def forward(self, z_local: Tensor, z_neighbor: Tensor) -> Tensor:
        """Fuse local and neighbor representations.
        
        Parameters
        ----------
        z_local : Tensor  ``[n_agents, latent_dim]``
        z_neighbor : Tensor  ``[n_agents, latent_dim]``
        
        Returns
        -------
        z : Tensor  ``[n_agents, out_dim]``
        """
        combined = torch.cat([z_local, z_neighbor], dim=-1)
        return self.net(combined)


class LocalNeighborGATDiscreteSAC(nn.Module):
    """Local-Neighbor GAT Discrete SAC agent.
    
    Architecture: local observation → local encoder, graph structure → neighbor encoder,
    fuse both → shared actor + centralized twin critics.
    
    Parameters
    ----------
    obs_dim : int
        Per-agent observation dimension.
    num_actions : int
        Number of discrete actions.
    local_encoder_cfg : dict
        Kwargs for LocalEncoder.
    neighbor_encoder_cfg : dict
        Kwargs for NeighborEncoder.
    fusion_cfg : dict
        Kwargs for FusionMLP.
    actor_cfg : dict
        Kwargs for SharedDiscreteActor.
    critic_cfg : dict
        Kwargs for CentralizedTwinCritic.
    init_alpha : float
        Initial entropy coefficient.
    tau : float
        Soft-update coefficient for target critic.
    """
    
    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        local_encoder_cfg: dict | None = None,
        neighbor_encoder_cfg: dict | None = None,
        fusion_cfg: dict | None = None,
        actor_cfg: dict | None = None,
        critic_cfg: dict | None = None,
        init_alpha: float = 0.2,
        tau: float = 0.005,
    ) -> None:
        super().__init__()
        
        local_encoder_cfg = local_encoder_cfg or {}
        neighbor_encoder_cfg = neighbor_encoder_cfg or {}
        fusion_cfg = fusion_cfg or {}
        actor_cfg = actor_cfg or {}
        critic_cfg = critic_cfg or {}
        
        # Encoder stages
        local_out_dim = local_encoder_cfg.get("out_dim", 48)
        neighbor_out_dim = neighbor_encoder_cfg.get("out_dim", 48)
        fusion_out_dim = fusion_cfg.get("out_dim", 64)
        
        self.local_encoder = LocalEncoder(in_dim=obs_dim, **local_encoder_cfg)
        self.neighbor_encoder = NeighborEncoder(in_dim=obs_dim, **neighbor_encoder_cfg)
        self.fusion = FusionMLP(
            latent_dim=max(local_out_dim, neighbor_out_dim),
            **fusion_cfg
        )
        
        # Actor and critic
        self.actor = SharedDiscreteActor(
            latent_dim=fusion_out_dim, num_actions=num_actions, **actor_cfg
        )
        self.critic = CentralizedTwinCritic(
            latent_dim=fusion_out_dim, num_actions=num_actions, **critic_cfg
        )
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        
        # Entropy temperature
        self.log_alpha = nn.Parameter(
            torch.tensor(init_alpha, dtype=torch.float32).log()
        )
        self.target_entropy = 0.98 * math.log(float(max(num_actions, 2)))
        
        self.tau = tau
    
    @staticmethod
    def _pool_agent_latents(
        z_nodes: Tensor,
        agent_node_indices: Tensor | None = None,
        agent_node_mask: Tensor | None = None,
    ) -> Tensor:
        """Pool node embeddings into per-agent embeddings if needed."""
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
        """Encode observations via local + neighbor pathway."""
        z_local = self.local_encoder(obs)
        z_neighbor = self.neighbor_encoder(obs, edge_index, edge_attr)
        
        # Fusion
        z = self.fusion(z_local, z_neighbor)
        
        # Pool if needed
        return self._pool_agent_latents(z, agent_node_indices, agent_node_mask)
    
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
        """End-to-end: obs → encode → actor → (action, log_prob)."""
        z = self.encode(
            obs, edge_index, edge_attr,
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
        """Return (z, action_probs, log_action_probs)."""
        z = self.encode(
            obs, edge_index, edge_attr,
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
        """Compute online critic Q-values."""
        z = self.encode(
            obs, edge_index, edge_attr,
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
        """Compute target critic Q-values (no grad)."""
        with torch.no_grad():
            z = self.encode(
                obs, edge_index, edge_attr,
                agent_node_indices=agent_node_indices,
                agent_node_mask=agent_node_mask,
            )
            return self.target_critic(z)
    
    @torch.no_grad()
    def soft_update_target(self) -> None:
        """Polyak-average update of target critic."""
        for tp, op in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)
    
    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()
