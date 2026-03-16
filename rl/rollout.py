"""Rollout worker — collects transitions from the SUMO environment.

Keeps the rollout loop separate from the training loop so that Lightning
only handles optimization, not simulation interaction.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import Tensor

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
    ) -> None:
        self.env = env
        self.agent = agent
        self.device = torch.device(device)

    @staticmethod
    def _graph_inputs(td: TensorDict) -> tuple[Tensor, Tensor | None, Tensor | None]:
        graph_obs = td.get("graph_observation", td["agents", "observation"])
        agent_node_indices = td.get("agent_node_indices", None)
        agent_node_mask = td.get("agent_node_mask", None)
        return graph_obs, agent_node_indices, agent_node_mask

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

        for _ in range(n_steps):
            obs, agent_node_indices, agent_node_mask = self._graph_inputs(td)
            edge_index = td["edge_index"]
            edge_attr = td.get("edge_attr", None)
            action_mask = td["agents", "action_mask"]

            actions, log_probs = self.agent.select_action(
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

            transition = self._pack_transition(td, actions, next_td)
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

        total_reward = 0.0
        steps = 0

        while True:
            obs, agent_node_indices, agent_node_mask = self._graph_inputs(td)
            edge_index = td["edge_index"]
            edge_attr = td.get("edge_attr", None)
            action_mask = td["agents", "action_mask"]

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

            transition = self._pack_transition(td, actions, next_td)
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
                        "graph_observation": next_td.get(
                            "graph_observation",
                            next_td["agents", "observation"],
                        ),
                    },
                    batch_size=[],
                ),
                "graph_observation": td.get("graph_observation", td["agents", "observation"]),
                "agent_node_indices": td["agent_node_indices"],
                "agent_node_mask": td["agent_node_mask"],
                "edge_index": td["edge_index"],
            },
            batch_size=[],
        )
        if "edge_attr" in td.keys():
            transition["edge_attr"] = td["edge_attr"]
        return transition
