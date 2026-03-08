"""Shared discrete actor for traffic-signal phase selection.

One actor shared across all intersections (parameter sharing).
Input:  per-agent graph latent ``z_i`` from the encoder.
Output: logits over discrete phase actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SharedDiscreteActor(nn.Module):
    """Shared-parameter discrete policy head.

    Parameters
    ----------
    latent_dim : int
        Dimension of encoder output ``z_i``.
    num_actions : int
        Number of discrete phase actions (max across agents; pad if needed).
    hidden_dim : int
        Width of the hidden layer.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        num_actions: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, z: Tensor, action_mask: Tensor | None = None) -> Tensor:
        """Return action logits.

        Parameters
        ----------
        z : Tensor  ``[n_agents, latent_dim]`` or ``[B, n_agents, latent_dim]``
        action_mask : Tensor | None  ``[..., num_actions]``  (True = allowed)

        Returns
        -------
        logits : Tensor  ``[..., num_actions]``
        """
        logits = self.net(z)
        if action_mask is not None:
            # Mask out illegal actions with large negative value
            logits = logits.masked_fill(~action_mask, -1e8)
        return logits

    def get_action(
        self, z: Tensor, action_mask: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Sample action and return ``(action, log_prob)``.

        Parameters
        ----------
        z : Tensor  ``[n_agents, latent_dim]``
        action_mask : Tensor | None
        deterministic : bool
            If ``True``, take argmax instead of sampling.

        Returns
        -------
        action : Tensor  ``[n_agents]``  (int64)
        log_prob : Tensor  ``[n_agents]``
        """
        logits = self.forward(z, action_mask)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_log_prob_entropy(
        self, z: Tensor, actions: Tensor, action_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute log-probabilities and entropy for given actions.

        Used by SAC loss computation.
        """
        logits = self.forward(z, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy

    def get_action_probs(
        self, z: Tensor, action_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Return ``(action_probs, log_action_probs)`` for DiscreteSAC.

        Parameters
        ----------
        z : Tensor  ``[..., latent_dim]``
        action_mask : Tensor | None

        Returns
        -------
        action_probs : Tensor  ``[..., num_actions]``
        log_action_probs : Tensor  ``[..., num_actions]``
        """
        logits = self.forward(z, action_mask)
        action_probs = torch.softmax(logits, dim=-1)
        # Clamp for numerical stability in log
        log_action_probs = torch.log(action_probs.clamp(min=1e-8))
        return action_probs, log_action_probs
