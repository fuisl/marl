"""GATv2-based graph encoder for intersection embeddings.

Takes per-intersection node features and the static road-network topology,
returns a latent embedding per controlled intersection.

Uses PyG exclusively — no RL logic here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv


class GraphEncoder(nn.Module):
    """Two-layer GATv2Conv encoder.

    Parameters
    ----------
    in_dim : int
        Raw node feature dimension (observation_dim from env).
    hidden_dim : int
        Hidden channel width.
    out_dim : int
        Output latent dimension per node (``z_i``).
    heads : int
        Number of attention heads in each GATv2Conv layer.
    edge_dim : int | None
        Dimension of edge attributes.  ``None`` to ignore edge features.
    dropout : float
        Dropout applied inside attention.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 64,
        heads: int = 4,
        edge_dim: int | None = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.conv1 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,           # output dim = (hidden_dim // heads) * heads = hidden_dim
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
        )

        self.output_proj = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ELU()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Encode intersection features.

        Parameters
        ----------
        x : Tensor  ``[n_agents, in_dim]``
        edge_index : Tensor  ``[2, E]``
        edge_attr : Tensor | None  ``[E, edge_dim]``

        Returns
        -------
        z : Tensor  ``[n_agents, out_dim]``
        """
        h = self.act(self.input_proj(x))              # [N, hidden]
        h = self.act(self.conv1(h, edge_index, edge_attr))  # [N, hidden]
        h = self.act(self.conv2(h, edge_index, edge_attr))  # [N, hidden]
        z = self.output_proj(h)                        # [N, out_dim]
        return z
