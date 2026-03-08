"""Action constraints and phase-transition logic.

Handles yellow / all-red transition rules and legal-action masking so that
the RL agent never issues an unsafe phase change.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class ActionConstraints:
    """Per-intersection action masking and yellow-phase enforcement.

    Parameters
    ----------
    yellow_duration : int
        Number of SUMO seconds for yellow clearance phase.
    all_red_duration : int
        Number of SUMO seconds for all-red clearance phase after yellow.
    min_green_duration : int
        Minimum green time before a phase switch is allowed.
    """

    def __init__(
        self,
        yellow_duration: int = 3,
        all_red_duration: int = 1,
        min_green_duration: int = 5,
    ) -> None:
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration
        self.min_green_duration = min_green_duration

        # Per-agent state: tracks transition progress
        self._agent_state: dict[str, _AgentTransitionState] = {}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def register_agent(
        self,
        tl_id: str,
        num_phases: int,
        green_phase_indices: list[int],
    ) -> None:
        """Register an intersection's phase structure."""
        self._agent_state[tl_id] = _AgentTransitionState(
            num_phases=num_phases,
            green_phase_indices=green_phase_indices,
        )

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------
    def get_action_mask(self, tl_id: str, current_phase: int, elapsed_green: float) -> Tensor:
        """Return boolean mask ``[num_green_phases]``.

        ``True`` = action allowed, ``False`` = blocked.
        """
        state = self._agent_state[tl_id]
        mask = torch.ones(len(state.green_phase_indices), dtype=torch.bool)

        # During transition, only allow "keep current destination phase"
        if state.in_transition:
            mask[:] = False
            dest_idx = state.green_phase_indices.index(state.transition_dest_phase)
            mask[dest_idx] = True
            return mask

        # Min-green constraint: if not elapsed, only allow keep-current
        if elapsed_green < self.min_green_duration:
            mask[:] = False
            if current_phase in state.green_phase_indices:
                keep_idx = state.green_phase_indices.index(current_phase)
                mask[keep_idx] = True
        return mask

    # ------------------------------------------------------------------
    # Transition management
    # ------------------------------------------------------------------
    def needs_transition(self, tl_id: str, current_green: int, desired_green: int) -> bool:
        """Whether switching from *current_green* to *desired_green* requires yellow."""
        return current_green != desired_green

    def start_transition(self, tl_id: str, dest_green_phase: int) -> None:
        state = self._agent_state[tl_id]
        state.in_transition = True
        state.transition_dest_phase = dest_green_phase
        state.transition_timer = self.yellow_duration + self.all_red_duration

    def tick_transition(self, tl_id: str, seconds: int = 1) -> bool:
        """Advance the transition timer. Returns ``True`` when transition is complete."""
        state = self._agent_state[tl_id]
        if not state.in_transition:
            return True
        state.transition_timer -= seconds
        if state.transition_timer <= 0:
            state.in_transition = False
            state.transition_timer = 0
            return True
        return False

    def get_transition_phase_idx(self, tl_id: str) -> int | None:
        """Return the destination green phase index if in transition, else None."""
        state = self._agent_state[tl_id]
        if state.in_transition:
            return state.transition_dest_phase
        return None


class _AgentTransitionState:
    """Internal book-keeping for one intersection's transition state."""

    __slots__ = (
        "num_phases",
        "green_phase_indices",
        "in_transition",
        "transition_dest_phase",
        "transition_timer",
    )

    def __init__(self, num_phases: int, green_phase_indices: list[int]) -> None:
        self.num_phases = num_phases
        self.green_phase_indices = green_phase_indices
        self.in_transition = False
        self.transition_dest_phase = -1
        self.transition_timer = 0
