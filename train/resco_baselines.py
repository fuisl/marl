from __future__ import annotations

from itertools import islice, permutations
from math import factorial
import random
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
    candidate_action_indices: list[int] | None = None,
) -> int:
    candidates = (
        list(range(len(phase_pairs)))
        if candidate_action_indices is None
        else list(candidate_action_indices)
    )
    if not candidates:
        raise ValueError("At least one candidate action is required.")

    best_action = int(candidates[0])
    best_score = float("-inf")
    for action_idx in candidates:
        pair = phase_pairs[action_idx]
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


def _valid_global_actions(spec: dict[str, Any]) -> list[int]:
    pair_to_act_map = spec.get("pair_to_act_map", {})
    if not isinstance(pair_to_act_map, dict) or not pair_to_act_map:
        raise ValueError("Signal spec is missing pair_to_act_map.")
    return sorted(int(global_idx) for global_idx in pair_to_act_map)


def _select_mapped_local_action(
    *,
    spec: dict[str, Any],
    values_by_direction: dict[str, float],
) -> int:
    phase_pairs = list(spec["phase_pairs"])
    pair_to_act_map = {
        int(global_idx): int(local_idx)
        for global_idx, local_idx in spec["pair_to_act_map"].items()
    }
    best_global_action = select_phase_pair_action(
        values_by_direction=values_by_direction,
        phase_pairs=phase_pairs,
        candidate_action_indices=_valid_global_actions(spec),
    )
    return int(pair_to_act_map[best_global_action])


def stochastic_actions(
    *,
    signal_specs: dict[str, dict[str, Any]],
    rng: random.Random,
) -> dict[str, int]:
    actions: dict[str, int] = {}
    for signal_id, spec in signal_specs.items():
        num_actions = int(spec["local_num_actions"])
        if num_actions <= 0:
            raise ValueError(f"Signal {signal_id!r} exposes no local actions.")
        actions[signal_id] = int(rng.randrange(num_actions))
    return actions


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
        actions[signal_id] = _select_mapped_local_action(
            spec=spec,
            values_by_direction=values_by_direction,
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
        actions[signal_id] = _select_mapped_local_action(
            spec=spec,
            values_by_direction=values_by_direction,
        )
    return actions
