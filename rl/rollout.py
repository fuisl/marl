"""Rollout worker — collects transitions from the SUMO environment.

Keeps simulation interaction separate from optimization so training loops
can remain simple and easy to extend.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import Tensor

from marl_env.observation_adapter import ObservationAdapter
from marl_env.sumo_env import TrafficSignalEnv
from models.marl_discrete_sac import MARLDiscreteSAC


class RolloutWorker:
    """Collect transitions from the environment using the current policy.

    Parameters
    ----------
    env : TrafficSignalEnv
        The multi-agent SUMO environment.
    agent : MARLDiscreteSAC
        The MARL agent (encoder + actor + critics).
    device : torch.device
        Device for inference.
    """

    def __init__(
        self,
        env: TrafficSignalEnv,
        agent: MARLDiscreteSAC,
        device: torch.device | str = "cpu",
        *,
        feature_mode: str = "wave",
    ) -> None:
        self.env = env
        self.agent = agent
        self.device = torch.device(device)
        self.feature_mode = feature_mode
        self.adapter: ObservationAdapter | None = None

    def _ensure_adapter(self) -> ObservationAdapter:
        if self.adapter is None:
            self.adapter = ObservationAdapter(
                signal_specs=self.env.get_signal_specs(),
                tl_ids=self.env.tl_ids,
                layout=self.env.observation_layout,
                graph_metadata=self.env.get_graph_metadata(),
            )
        return self.adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_steps(
        self,
        n_steps: int,
        deterministic: bool = False,
    ) -> list[TensorDict]:
        """Collect ``n_steps`` transitions and return a list of TensorDicts.

        Each returned TensorDict is a single transition with the standard
        MARL replay layout (see ``rl.replay``).
        """
        transitions: list[TensorDict] = []
        td = self.env.reset()
        td = td.to(self.device)
        adapter = self._ensure_adapter()
        graph_metadata = self.env.get_graph_metadata()

        for _ in range(n_steps):
            obs = adapter.graph_features(
                td["agents", "observation"],
                feature_mode=self.feature_mode,
            )
            edge_index = graph_metadata.edge_index.to(self.device)
            edge_attr = (
                None if graph_metadata.edge_attr is None else graph_metadata.edge_attr.to(self.device)
            )
            action_mask = td["agents", "action_mask"]
            agent_node_indices = graph_metadata.agent_node_indices.to(self.device)
            agent_node_mask = graph_metadata.agent_node_mask.to(self.device)

            actions, _ = self.agent.select_action(
                obs,
                edge_index,
                edge_attr,
                action_mask,
                deterministic,
                agent_node_indices=agent_node_indices,
                agent_node_mask=agent_node_mask,
            )

            next_td = self.env.step(actions.cpu())
            next_td = next_td.to(self.device)
            next_obs = adapter.graph_features(
                next_td["agents", "observation"],
                feature_mode=self.feature_mode,
            )

            transition = self._pack_transition(
                td,
                actions,
                next_td,
                obs,
                next_obs,
                edge_index,
                edge_attr,
                agent_node_indices,
                agent_node_mask,
            )
            transitions.append(transition.cpu())

            # Check done
            if next_td["done"].item():
                td = self.env.reset().to(self.device)
            else:
                td = next_td

        return transitions

    @torch.no_grad()
    def collect_episode(
        self, deterministic: bool = False
    ) -> tuple[list[TensorDict], dict[str, float]]:
        """Run one full episode. Return transitions and summary metrics."""
        transitions: list[TensorDict] = []
        td = self.env.reset().to(self.device)
        adapter = self._ensure_adapter()
        graph_metadata = self.env.get_graph_metadata()

        total_reward = 0.0
        steps = 0

        while True:
            obs = adapter.graph_features(
                td["agents", "observation"],
                feature_mode=self.feature_mode,
            )
            edge_index = graph_metadata.edge_index.to(self.device)
            edge_attr = (
                None if graph_metadata.edge_attr is None else graph_metadata.edge_attr.to(self.device)
            )
            action_mask = td["agents", "action_mask"]
            agent_node_indices = graph_metadata.agent_node_indices.to(self.device)
            agent_node_mask = graph_metadata.agent_node_mask.to(self.device)

            actions, _ = self.agent.select_action(
                obs,
                edge_index,
                edge_attr,
                action_mask,
                deterministic,
                agent_node_indices=agent_node_indices,
                agent_node_mask=agent_node_mask,
            )

            next_td = self.env.step(actions.cpu()).to(self.device)
            next_obs = adapter.graph_features(
                next_td["agents", "observation"],
                feature_mode=self.feature_mode,
            )

            transition = self._pack_transition(
                td,
                actions,
                next_td,
                obs,
                next_obs,
                edge_index,
                edge_attr,
                agent_node_indices,
                agent_node_mask,
            )
            transitions.append(transition.cpu())

            total_reward += next_td["agents", "reward"].sum().item()
            steps += 1

            if next_td["done"].item():
                break
            td = next_td

        info = {
            "episode_return": total_reward,
            "episode_length": float(steps),
        }
        return transitions, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_transition(
        td: TensorDict,
        actions: Tensor,
        next_td: TensorDict,
        graph_observation: Tensor,
        next_graph_observation: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None,
        agent_node_indices: Tensor,
        agent_node_mask: Tensor,
    ) -> TensorDict:
        """Pack current obs, action, reward, next obs into a flat transition."""
        transition = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": td["agents", "observation"],
                        "action": actions,
                        "action_mask": td["agents", "action_mask"],
                        "reward": next_td["agents", "reward"],
                        "done": next_td["agents", "done"],
                    },
                    batch_size=td["agents"].batch_size,
                ),
                "next": TensorDict(
                    {
                        "agents": TensorDict(
                            {
                                "observation": next_td["agents", "observation"],
                                "action_mask": next_td["agents", "action_mask"],
                                "done": next_td["agents", "done"],
                            },
                            batch_size=td["agents"].batch_size,
                        ),
                        "graph_observation": next_graph_observation,
                    },
                    batch_size=[],
                ),
                "graph_observation": graph_observation,
                "agent_node_indices": agent_node_indices,
                "agent_node_mask": agent_node_mask,
                "edge_index": edge_index,
            },
            batch_size=[],
        )
        if edge_attr is not None:
            transition["edge_attr"] = edge_attr
        return transition
