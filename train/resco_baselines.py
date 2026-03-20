from __future__ import annotations

from itertools import islice, permutations
from math import factorial
from typing import Any


def permutations_without_rotations(values: list[int]) -> list[tuple[int, ...]]:
    if not values:
        return []
    count = factorial(max(len(values) - 1, 0))
    return list(islice(permutations(values), count))


class RescoFixedSignalController:
    def __init__(
        self,
        *,
        num_actions: int,
        fixed_timings: list[int],
        fixed_phase_order_idx: int = 0,
        fixed_offset: int = 0,
    ) -> None:
        phase_orders = permutations_without_rotations(list(range(num_actions)))
        if not phase_orders:
            raise ValueError("RescoFixedSignalController requires at least one action.")

        order_idx = min(max(int(fixed_phase_order_idx), 0), len(phase_orders) - 1)
        self.phase_order = list(phase_orders[order_idx])
        self.plan = [int(value) for value in fixed_timings[:num_actions]]
        if len(self.plan) < num_actions:
            self.plan.extend([0] * (num_actions - len(self.plan)))

        self.active_phase = 0
        self.active_phase_len = 0
        self._apply_offset(int(fixed_offset))

    def _advance_to_next_nonzero_phase(self) -> None:
        if not self.plan:
            return
        start = self.active_phase
        while self.plan[self.active_phase] == 0:
            self.active_phase = (self.active_phase + 1) % len(self.plan)
            if self.active_phase == start:
                break

    def _apply_offset(self, offset: int) -> None:
        if offset <= 0 or not self.plan:
            self._advance_to_next_nonzero_phase()
            return

        self._advance_to_next_nonzero_phase()
        remaining = offset
        while remaining > 0:
            duration = abs(self.plan[self.active_phase])
            if duration == 0:
                self._advance_to_next_nonzero_phase()
                break
            if remaining >= duration:
                remaining -= duration
                self.active_phase = (self.active_phase + 1) % len(self.plan)
                self.active_phase_len = 0
                self._advance_to_next_nonzero_phase()
                continue
            self.active_phase_len = remaining
            remaining = 0

    def act(self) -> int:
        if not self.plan or all(duration == 0 for duration in self.plan):
            return 0

        self._advance_to_next_nonzero_phase()
        current_duration = abs(self.plan[self.active_phase])
        if self.active_phase_len >= current_duration:
            self.active_phase = (self.active_phase + 1) % len(self.plan)
            self._advance_to_next_nonzero_phase()
            self.active_phase_len = 1
        else:
            self.active_phase_len += 1
        return int(self.phase_order[self.active_phase])


def select_phase_pair_action(
    *,
    values_by_direction: dict[str, float],
    phase_pairs: list[list[str]],
) -> int:
    best_action = 0
    best_score = float("-inf")
    for action_idx, pair in enumerate(phase_pairs):
        score = 0.0
        for direction in pair:
            score += float(values_by_direction.get(direction, 0.0))
        if score > best_score:
            best_score = score
            best_action = action_idx
    return best_action


def build_direction_value_map(
    *,
    directions: list[str],
    values: list[float],
) -> dict[str, float]:
    return {
        str(direction): float(values[idx])
        for idx, direction in enumerate(directions)
        if idx < len(values)
    }


def maxwave_actions(
    *,
    signal_specs: dict[str, dict[str, Any]],
    wave_states: dict[str, list[float]],
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for signal_id, spec in signal_specs.items():
        values_by_direction = build_direction_value_map(
            directions=list(spec["directions"]),
            values=list(wave_states.get(signal_id, [])),
        )
        actions[signal_id] = select_phase_pair_action(
            values_by_direction=values_by_direction,
            phase_pairs=list(spec["phase_pairs"]),
        )
    return actions


def maxpressure_actions(
    *,
    signal_specs: dict[str, dict[str, Any]],
    mplight_states: dict[str, list[float]],
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for signal_id, spec in signal_specs.items():
        state = list(mplight_states.get(signal_id, []))
        directional_pressures = state[1:] if state else []
        values_by_direction = build_direction_value_map(
            directions=list(spec["directions"]),
            values=directional_pressures,
        )
        actions[signal_id] = select_phase_pair_action(
            values_by_direction=values_by_direction,
            phase_pairs=list(spec["phase_pairs"]),
        )
    return actions
