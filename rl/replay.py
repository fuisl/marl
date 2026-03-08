"""TensorDictReplayBuffer setup.

Stores MARL transitions in TorchRL's native replay buffer so that
sampling returns properly-shaped ``TensorDict`` batches ready for
loss computation.
"""

from __future__ import annotations

from torchrl.data import (
    LazyTensorStorage,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import RandomSampler


def make_replay_buffer(
    capacity: int = 100_000,
    batch_size: int = 256,
    prefetch: int = 4,
) -> TensorDictReplayBuffer:
    """Create a ``TensorDictReplayBuffer`` with lazy storage.

    The buffer stores full transition ``TensorDict`` objects with the
    following expected keys::

        root
        ├── agents/
        │   ├── observation   [n_agents, obs_dim]
        │   ├── action        [n_agents]
        │   ├── action_mask   [n_agents, num_actions]
        │   ├── reward        [n_agents, 1]
        │   └── done          [n_agents, 1]
        ├── next/
        │   └── agents/
        │       ├── observation   [n_agents, obs_dim]
        │       ├── action_mask   [n_agents, num_actions]
        │       └── done          [n_agents, 1]
        ├── edge_index        [2, E]
        └── edge_attr         [E, d_edge]   (optional)

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    batch_size : int
        Number of transitions per sample.
    prefetch : int
        Number of batches to prefetch (0 to disable).
    """
    storage = LazyTensorStorage(max_size=capacity)
    sampler = RandomSampler()

    buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=sampler,
        batch_size=batch_size,
        prefetch=prefetch,
    )
    return buffer
