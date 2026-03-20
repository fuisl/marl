from __future__ import annotations

from train.resco_baselines import (
    RescoFixedSignalController,
    maxpressure_actions,
    maxwave_actions,
    permutations_without_rotations,
)


def test_permutations_without_rotations_matches_expected_size() -> None:
    orders = permutations_without_rotations([0, 1, 2, 3])
    assert len(orders) == 6
    assert orders[0] == (0, 1, 2, 3)


def test_fixed_signal_controller_cycles_using_plan_lengths() -> None:
    controller = RescoFixedSignalController(
        num_actions=3,
        fixed_timings=[2, 1, 0],
        fixed_phase_order_idx=0,
        fixed_offset=0,
    )
    actions = [controller.act() for _ in range(6)]
    assert actions == [0, 0, 1, 0, 0, 1]


def test_fixed_signal_controller_applies_offset_before_first_action() -> None:
    controller = RescoFixedSignalController(
        num_actions=2,
        fixed_timings=[2, 1],
        fixed_phase_order_idx=0,
        fixed_offset=1,
    )
    actions = [controller.act() for _ in range(4)]
    assert actions == [0, 1, 0, 0]


def test_maxwave_and_maxpressure_action_selection_uses_phase_pairs() -> None:
    signal_specs = {
        "J1": {
            "directions": ["N-N", "S-S", "W-W"],
            "phase_pairs": [["N-N", "S-S"], ["W-W", "N-N"]],
        }
    }

    assert maxwave_actions(
        signal_specs=signal_specs,
        wave_states={"J1": [3.0, 5.0, 1.0]},
    ) == {"J1": 0}

    assert maxpressure_actions(
        signal_specs=signal_specs,
        mplight_states={"J1": [0.0, 1.0, 2.0, 10.0]},
    ) == {"J1": 1}
