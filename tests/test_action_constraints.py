from __future__ import annotations

import pytest
import torch

from env.action_constraints import ActionConstraints


def make_constraints() -> ActionConstraints:
    constraints = ActionConstraints(
        yellow_duration=3,
        all_red_duration=1,
        min_green_duration=5,
    )
    constraints.register_agent(
        "J1",
        num_phases=6,
        green_phase_indices=[0, 3],
        yellow_phase_map={(0, 3): 1, (3, 0): 4},
        all_red_phase_map={(0, 3): 2, (3, 0): 5},
    )
    return constraints


def test_action_phase_mapping() -> None:
    constraints = make_constraints()
    assert constraints.action_to_green_phase("J1", 0) == 0
    assert constraints.action_to_green_phase("J1", 1) == 3
    assert constraints.green_phase_to_action("J1", 0) == 0
    assert constraints.green_phase_to_action("J1", 3) == 1


def test_min_green_mask() -> None:
    constraints = make_constraints()
    mask = constraints.get_action_mask("J1", current_green_phase=0, elapsed_green=2)
    assert torch.equal(mask, torch.tensor([True, False]))


def test_begin_switch_and_progress() -> None:
    constraints = make_constraints()
    constraints.begin_switch("J1", current_green_phase=0, target_action=1)

    assert constraints.in_transition("J1")
    assert constraints.phase_to_apply("J1") == 1

    assert constraints.tick("J1", 1) is False
    assert constraints.phase_to_apply("J1") == 1

    assert constraints.tick("J1", 1) is False
    assert constraints.phase_to_apply("J1") == 1

    assert constraints.tick("J1", 1) is False
    assert constraints.phase_to_apply("J1") == 2

    assert constraints.tick("J1", 1) is True
    assert constraints.phase_to_apply("J1") == 3
    assert constraints.destination_green("J1") == 3

    constraints.complete_switch("J1")
    assert not constraints.in_transition("J1")
    assert constraints.phase_to_apply("J1") is None


def test_transition_mask_only_allows_destination() -> None:
    constraints = make_constraints()
    constraints.begin_switch("J1", current_green_phase=0, target_action=1)
    mask = constraints.get_action_mask("J1", current_green_phase=0, elapsed_green=99)
    assert torch.equal(mask, torch.tensor([False, True]))


def test_same_phase_is_noop() -> None:
    constraints = make_constraints()
    constraints.begin_switch("J1", current_green_phase=0, target_action=0)
    assert not constraints.in_transition("J1")


def test_invalid_action_raises() -> None:
    constraints = make_constraints()
    with pytest.raises(ValueError):
        constraints.begin_switch("J1", current_green_phase=0, target_action=99)
