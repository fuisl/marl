"""Action constraints and phase-transition logic.

Implements a finite-state transition controller that governs legal-action
masking and yellow / all-red clearance for every controlled intersection.

Two index spaces coexist and are kept strictly separate:

* **action_idx** — RL output in ``[0, G-1]`` where *G* is the number of
  controllable green phases.
* **green_phase_idx** — actual SUMO phase index inside the traffic-light
  program (may be any non-negative integer).

The controller answers the one question the environment must ask every
simulation second: *"What SUMO phase should I apply right now?"*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor


# ======================================================================
# Transition plan — explicit FSM for one phase switch
# ======================================================================
@dataclass
class TransitionPlan:
    """Describes a pending green-to-green transition.

    Fields
    ------
    from_green_phase : int
        SUMO phase index of the departing green.
    to_green_phase : int
        SUMO phase index of the destination green.
    yellow_phase : int | None
        SUMO phase index to display during yellow clearance.
        ``None`` means skip yellow (direct switch).
    all_red_phase : int | None
        SUMO phase index to display during all-red clearance.
        ``None`` means skip all-red.
    stage : str
        Current FSM state: ``"idle"`` | ``"yellow"`` | ``"all_red"``.
    timer : int
        Remaining seconds in the current stage.
    """

    from_green_phase: int
    to_green_phase: int
    yellow_phase: int | None = None
    all_red_phase: int | None = None
    stage: str = "idle"
    timer: int = 0


# ======================================================================
# Per-agent bookkeeping
# ======================================================================
class _AgentState:
    """Internal state for one controlled intersection."""

    __slots__ = (
        "num_phases",
        "green_phase_indices",
        "action_to_green",
        "green_to_action",
        "yellow_phase_map",
        "all_red_phase_map",
        "transition",
    )

    def __init__(
        self,
        num_phases: int,
        green_phase_indices: list[int],
        action_to_green: dict[int, int],
        green_to_action: dict[int, int],
        yellow_phase_map: dict[tuple[int, int], int],
        all_red_phase_map: dict[tuple[int, int], int],
    ) -> None:
        self.num_phases = num_phases
        self.green_phase_indices = green_phase_indices
        self.action_to_green = action_to_green
        self.green_to_action = green_to_action
        self.yellow_phase_map = yellow_phase_map
        self.all_red_phase_map = all_red_phase_map
        self.transition: TransitionPlan | None = None


# ======================================================================
# Main controller
# ======================================================================
class ActionConstraints:
    """Finite-state transition controller for traffic-light phase switching.

    Parameters
    ----------
    yellow_duration : int
        Seconds of yellow clearance.
    all_red_duration : int
        Seconds of all-red clearance (after yellow).
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

        self._agent_state: dict[str, _AgentState] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_agent(
        self,
        tl_id: str,
        num_phases: int,
        green_phase_indices: list[int],
        yellow_phase_map: dict[tuple[int, int], int] | None = None,
        all_red_phase_map: dict[tuple[int, int], int] | None = None,
    ) -> None:
        """Register an intersection's phase structure.

        Parameters
        ----------
        tl_id : str
            SUMO traffic-light ID.
        num_phases : int
            Total number of phases in the SUMO program.
        green_phase_indices : list[int]
            SUMO phase indices that correspond to controllable greens.
        yellow_phase_map : dict[(from_green, to_green), yellow_phase] | None
            Per-transition yellow phases. Missing entries → skip yellow.
        all_red_phase_map : dict[(from_green, to_green), all_red_phase] | None
            Per-transition all-red phases. Missing entries → skip all-red.
        """
        if not green_phase_indices:
            raise ValueError(f"{tl_id}: green_phase_indices must not be empty.")
        if len(set(green_phase_indices)) != len(green_phase_indices):
            raise ValueError(f"{tl_id}: green_phase_indices must be unique.")
        if min(green_phase_indices) < 0 or max(green_phase_indices) >= num_phases:
            raise ValueError(
                f"{tl_id}: green_phase_indices out of range "
                f"[0, {num_phases - 1}]: {green_phase_indices}"
            )

        action_to_green = {i: p for i, p in enumerate(green_phase_indices)}
        green_to_action = {p: i for i, p in enumerate(green_phase_indices)}

        self._agent_state[tl_id] = _AgentState(
            num_phases=num_phases,
            green_phase_indices=green_phase_indices,
            action_to_green=action_to_green,
            green_to_action=green_to_action,
            yellow_phase_map=yellow_phase_map or {},
            all_red_phase_map=all_red_phase_map or {},
        )

    # ------------------------------------------------------------------
    # Index conversion
    # ------------------------------------------------------------------
    def action_to_green_phase(self, tl_id: str, action_idx: int) -> int:
        """Convert RL action index → SUMO green phase index."""
        return self._agent_state[tl_id].action_to_green[action_idx]

    def green_phase_to_action(self, tl_id: str, green_phase: int) -> int:
        """Convert SUMO green phase index → RL action index."""
        return self._agent_state[tl_id].green_to_action[green_phase]

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agent_state)

    def num_actions(self, tl_id: str) -> int:
        """Number of controllable green phases for this intersection."""
        return len(self._agent_state[tl_id].green_phase_indices)

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------
    def get_action_mask(
        self, tl_id: str, current_green_phase: int, elapsed_green: float
    ) -> Tensor:
        """Return boolean mask ``[num_green_phases]``.

        Parameters
        ----------
        tl_id : str
            Traffic-light ID.
        current_green_phase : int
            **SUMO phase index** of the current controllable green.
        elapsed_green : float
            Seconds spent in the current green phase.

        Returns
        -------
        Tensor
            ``True`` = action allowed, ``False`` = blocked.
        """
        state = self._agent_state[tl_id]
        n_actions = len(state.green_phase_indices)
        mask = torch.ones(n_actions, dtype=torch.bool)

        # During transition → only allow "keep destination"
        if state.transition is not None:
            mask[:] = False
            keep_action = state.green_to_action[state.transition.to_green_phase]
            mask[keep_action] = True
            return mask

        # Min-green constraint → only allow "stay on current green"
        if elapsed_green < self.min_green_duration:
            mask[:] = False
            if current_green_phase in state.green_to_action:
                keep_action = state.green_to_action[current_green_phase]
                mask[keep_action] = True

        return mask

    # ------------------------------------------------------------------
    # Transition lifecycle
    # ------------------------------------------------------------------
    def in_transition(self, tl_id: str) -> bool:
        """Whether this intersection is mid-switch."""
        return self._agent_state[tl_id].transition is not None

    def begin_switch(
        self, tl_id: str, current_green_phase: int, target_action: int
    ) -> None:
        """Initiate a phase switch from *current_green_phase* to *target_action*.

        If the target green is the same as current, this is a no-op.

        Parameters
        ----------
        tl_id : str
            Traffic-light ID.
        current_green_phase : int
            **SUMO phase index** of the departing green.
        target_action : int
            **RL action index** of the desired green.
        """
        state = self._agent_state[tl_id]
        target_green_phase = state.action_to_green[target_action]

        # Same phase → nothing to do
        if current_green_phase == target_green_phase:
            state.transition = None
            return

        pair = (current_green_phase, target_green_phase)
        yellow = state.yellow_phase_map.get(pair)
        all_red = state.all_red_phase_map.get(pair)

        plan = TransitionPlan(
            from_green_phase=current_green_phase,
            to_green_phase=target_green_phase,
            yellow_phase=yellow,
            all_red_phase=all_red,
        )

        # Enter first applicable clearance stage
        if yellow is not None:
            plan.stage = "yellow"
            plan.timer = self.yellow_duration
        elif all_red is not None:
            plan.stage = "all_red"
            plan.timer = self.all_red_duration
        else:
            # No clearance phases defined → immediate switch
            plan.stage = "idle"
            plan.timer = 0

        state.transition = plan

    def phase_to_apply(self, tl_id: str) -> int | None:
        """Return the SUMO phase index the environment should set *right now*.

        Returns ``None`` when no transition is active (the environment
        should leave SUMO on the current green).

        During clearance this returns the yellow or all-red phase index.
        When clearance is complete (``stage == "idle"``), returns the
        destination green phase index.
        """
        state = self._agent_state[tl_id]
        plan = state.transition
        if plan is None:
            return None

        if plan.stage == "yellow":
            return plan.yellow_phase
        if plan.stage == "all_red":
            return plan.all_red_phase
        # stage == "idle" → transition plan resolved, apply destination green
        return plan.to_green_phase

    def destination_green(self, tl_id: str) -> int | None:
        """Return the SUMO green phase the current transition is heading to.

        ``None`` if no transition is active.
        """
        plan = self._agent_state[tl_id].transition
        return plan.to_green_phase if plan is not None else None

    def tick(self, tl_id: str, seconds: int = 1) -> bool:
        """Advance the transition FSM by *seconds*.

        Returns ``True`` when the transition is complete (destination green
        should now be applied) or when no transition is active.

        The caller should:

        1. Call :meth:`phase_to_apply` to get what to set in SUMO.
        2. Call :meth:`tick` to advance the timer.
        3. When ``tick`` returns ``True``, finalize the switch.
        """
        state = self._agent_state[tl_id]
        plan = state.transition
        if plan is None:
            return True

        if plan.stage == "idle":
            state.transition = None
            return True

        plan.timer -= seconds
        if plan.timer > 0:
            return False

        # Current stage exhausted → advance to next stage
        if plan.stage == "yellow" and plan.all_red_phase is not None:
            plan.stage = "all_red"
            plan.timer = self.all_red_duration
            return False

        # All clearance stages done
        state.transition = None
        return True
