"""Discrete SAC loss computation for MARL.

Implements the three SAC losses (critic, actor, alpha) using the
graph-based encoder for state representation.  Designed to work with
raw custom training loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict
from torch import Tensor

from models.local_neighbor_gat_discrete_sac import LocalNeighborGATDiscreteSAC


@dataclass
class SACLossOutput:
    """Container for the three SAC losses and diagnostics."""

    critic_loss: Tensor
    actor_loss: Tensor
    alpha_loss: Tensor
    # Diagnostics
    q1_mean: float = 0.0
    q2_mean: float = 0.0
    entropy: float = 0.0
    alpha: float = 0.0


class DiscreteSACLossComputer:
    """Compute Discrete SAC losses from a batch of transitions.

    Follows the Discrete SAC formulation where:
      - Critics output Q(s, ·) for all actions.
      - Actor loss uses expectation over action probabilities.
      - Alpha is auto-tuned against a target entropy.

    Parameters
    ----------
    agent : LocalNeighborGATDiscreteSAC
        The full MARL agent.
    gamma : float
        Discount factor.
    """

    def __init__(
        self,
        agent: LocalNeighborGATDiscreteSAC,
        gamma: float = 0.99,
        *,
        use_huber_loss: bool = False,
        huber_delta: float = 1.0,
        clip_target_q: bool = False,
        target_q_min: float = -1_000.0,
        target_q_max: float = 1_000.0,
    ) -> None:
        self.agent = agent
        self.gamma = gamma
        self.use_huber_loss = bool(use_huber_loss)
        self.huber_delta = float(huber_delta)
        self.clip_target_q = bool(clip_target_q)
        self.target_q_min = float(target_q_min)
        self.target_q_max = float(target_q_max)

    def __call__(self, batch: TensorDict) -> SACLossOutput:
        return self.compute(batch)

    def compute(self, batch: TensorDict) -> SACLossOutput:
        """Compute all three SAC losses from a sampled batch.

        Parameters
        ----------
        batch : TensorDict
            A batch sampled from the replay buffer with keys:
            ``agents/observation``, ``agents/action``, ``agents/reward``,
            ``agents/done``, ``agents/action_mask``,
            ``next/agents/observation``, ``next/agents/action_mask``,
            ``edge_index``, optionally ``edge_attr``.
        """
        # --- Unpack batch ---
        obs = batch["agents", "observation"]          # [B, n_agents, obs_dim]
        actions = batch["agents", "action"]            # [B, n_agents]
        rewards = batch["agents", "reward"]            # [B, n_agents, 1]
        dones = batch["agents", "done"].float()        # [B, n_agents, 1]
        action_mask = batch["agents", "action_mask"]   # [B, n_agents, num_actions]

        next_obs = batch["next", "agents", "observation"]    # [B, n_agents, obs_dim]
        next_mask = batch["next", "agents", "action_mask"]   # [B, n_agents, num_actions]
        graph_obs = batch.get("graph_observation", obs)
        next_graph_obs = batch["next"].get("graph_observation", next_obs)

        edge_index = batch["edge_index"]               # [B, 2, E] or [2, E]
        edge_attr = batch.get("edge_attr", None)
        agent_node_indices = batch.get("agent_node_indices", None)
        agent_node_mask = batch.get("agent_node_mask", None)

        B, N, _ = obs.shape

        # --- Critic loss ---
        critic_loss, q1_mean, q2_mean = self._critic_loss(
            graph_obs,
            actions,
            rewards,
            dones,
            next_graph_obs,
            next_mask,
            edge_index,
            edge_attr,
            agent_node_indices,
            agent_node_mask,
            B,
        )

        # --- Actor loss ---
        actor_loss, entropy = self._actor_loss(
            graph_obs,
            action_mask,
            edge_index,
            edge_attr,
            agent_node_indices,
            agent_node_mask,
            B,
        )

        # --- Alpha loss ---
        alpha_loss = self._alpha_loss(entropy)

        return SACLossOutput(
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            alpha_loss=alpha_loss,
            q1_mean=q1_mean,
            q2_mean=q2_mean,
            entropy=entropy,
            alpha=self.agent.alpha.item(),
        )

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------
    def _critic_loss(
        self,
        graph_obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        dones: Tensor,
        next_graph_obs: Tensor,
        next_mask: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        agent_node_indices: Tensor | None,
        agent_node_mask: Tensor | None,
        B: int,
    ) -> tuple[Tensor, float, float]:
        """Compute clipped double-Q critic loss."""
        alpha = self.agent.alpha.detach()

        # Current Q-values
        q1_all, q2_all = self._batch_critic(
            graph_obs,
            edge_index,
            edge_attr,
            agent_node_indices,
            agent_node_mask,
            B,
        )
        # Gather Q-values for taken actions: [B, N, 1]
        actions_idx = actions.long().unsqueeze(-1)
        q1 = q1_all.gather(-1, actions_idx)
        q2 = q2_all.gather(-1, actions_idx)

        # Target Q-values (no grad)
        with torch.no_grad():
            # Next action distribution from current policy
            next_z = self._batch_encode(
                next_graph_obs,
                edge_index,
                edge_attr,
                agent_node_indices,
                agent_node_mask,
                B,
            )
            next_probs, next_log_probs = self.agent.actor.get_action_probs(
                next_z, next_mask
            )

            # Target Q-values
            tq1_all, tq2_all = self._batch_target_critic(
                next_graph_obs,
                edge_index,
                edge_attr,
                agent_node_indices,
                agent_node_mask,
                B,
            )
            tq_min = torch.min(tq1_all, tq2_all)

            # V(s') = E_a[Q(s',a) - alpha * log pi(a|s')]
            v_next = (next_probs * (tq_min - alpha * next_log_probs)).sum(dim=-1, keepdim=True)

            target = rewards + (1.0 - dones) * self.gamma * v_next
            if self.clip_target_q:
                target = target.clamp(min=self.target_q_min, max=self.target_q_max)

        if self.use_huber_loss:
            loss_q1 = torch.nn.functional.huber_loss(
                q1,
                target,
                delta=self.huber_delta,
            )
            loss_q2 = torch.nn.functional.huber_loss(
                q2,
                target,
                delta=self.huber_delta,
            )
        else:
            loss_q1 = torch.nn.functional.mse_loss(q1, target)
            loss_q2 = torch.nn.functional.mse_loss(q2, target)

        loss = 0.5 * (loss_q1 + loss_q2)
        return loss, q1.mean().item(), q2.mean().item()

    def _actor_loss(
        self,
        graph_obs: Tensor,
        action_mask: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        agent_node_indices: Tensor | None,
        agent_node_mask: Tensor | None,
        B: int,
    ) -> tuple[Tensor, float]:
        """Compute actor loss: minimize E[alpha * log pi - Q]."""
        alpha = self.agent.alpha.detach()

        z = self._batch_encode(
            graph_obs,
            edge_index,
            edge_attr,
            agent_node_indices,
            agent_node_mask,
            B,
        )
        action_probs, log_action_probs = self.agent.actor.get_action_probs(
            z, action_mask
        )

        # Q-values from online critic (detach to not update critic via actor loss)
        with torch.no_grad():
            q1_all, q2_all = self._batch_critic(
                graph_obs,
                edge_index,
                edge_attr,
                agent_node_indices,
                agent_node_mask,
                B,
            )
        q_min = torch.min(q1_all, q2_all)

        # actor loss = E_a[alpha * log pi(a|s) - Q(s, a)]
        loss = (action_probs * (alpha * log_action_probs - q_min)).sum(dim=-1).mean()

        entropy = -(action_probs * log_action_probs).sum(dim=-1).mean().item()
        return loss, entropy

    def _alpha_loss(self, entropy: float) -> Tensor:
        """Adaptive temperature loss."""
        return (
            self.agent.log_alpha * (entropy - self.agent.target_entropy)
        ).mean()

    # ------------------------------------------------------------------
    # Batched encode / critic helpers
    # ------------------------------------------------------------------
    def _batch_encode(
        self,
        graph_obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        agent_node_indices: Tensor | None,
        agent_node_mask: Tensor | None,
        B: int,
    ) -> Tensor:
        """Encode a batch of observations via local+neighbor+fusion blocks."""
        num_graph_nodes = graph_obs.shape[1]
        obs_flat = graph_obs.reshape(B * num_graph_nodes, -1)  # [B*N_graph, obs_dim]

        # Replicate edge_index for each batch element
        if edge_index.dim() == 3:
            ei = edge_index[0]  # assume shared topology: [2, E]
        else:
            ei = edge_index

        # Build batch-diagonal edge_index for PyG
        ei_batch = _batch_edge_index(ei, B, num_graph_nodes)
        ea_batch = None
        if edge_attr is not None:
            ea = edge_attr[0] if edge_attr.dim() == 3 else edge_attr
            ea_batch = ea.repeat(B, 1)

        z_local = self.agent.local_encoder(obs_flat)
        z_neighbor = self.agent.neighbor_encoder(obs_flat, ei_batch, ea_batch)
        z_flat = self.agent.fusion(z_local, z_neighbor)
        z_nodes = z_flat.reshape(B, num_graph_nodes, -1)
        return self.agent._pool_agent_latents(
            z_nodes,
            agent_node_indices=agent_node_indices,
            agent_node_mask=agent_node_mask,
        )

    def _batch_critic(
        self,
        graph_obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        agent_node_indices: Tensor | None,
        agent_node_mask: Tensor | None,
        B: int,
    ) -> tuple[Tensor, Tensor]:
        z = self._batch_encode(
            graph_obs,
            edge_index,
            edge_attr,
            agent_node_indices,
            agent_node_mask,
            B,
        )
        return self.agent.critic(z)

    def _batch_target_critic(
        self,
        graph_obs: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        agent_node_indices: Tensor | None,
        agent_node_mask: Tensor | None,
        B: int,
    ) -> tuple[Tensor, Tensor]:
        z = self._batch_encode(
            graph_obs,
            edge_index,
            edge_attr,
            agent_node_indices,
            agent_node_mask,
            B,
        )
        return self.agent.target_critic(z)


def _batch_edge_index(edge_index: Tensor, B: int, N: int) -> Tensor:
    """Replicate a single graph's edge_index into a batch-diagonal format.

    Parameters
    ----------
    edge_index : Tensor [2, E]
    B : int  batch size
    N : int  number of nodes per graph

    Returns
    -------
    Tensor [2, B*E]
    """
    E = edge_index.shape[1]
    offsets = torch.arange(B, device=edge_index.device).unsqueeze(1) * N  # [B, 1]
    ei = edge_index.unsqueeze(0).expand(B, 2, E)  # [B, 2, E]
    ei = ei + offsets.unsqueeze(1)  # broadcast: [B, 2, E]
    return ei.reshape(2, B * E)
