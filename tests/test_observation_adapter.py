from __future__ import annotations

import torch

from marl_env.observation_adapter import CanonicalObservationLayout, ObservationAdapter


def test_canonical_snapshot_adapter_reproduces_wave_mplight_and_drq() -> None:
    layout = CanonicalObservationLayout(max_lanes=2)
    signal_specs = {
        "J1": {
            "directions": ["N-N", "S-S"],
            "lane_order": ["j1_l1", "j1_l2"],
            "lane_sets": {
                "N-N": ["j1_l1"],
                "S-S": ["j1_l2"],
            },
            "lane_sets_outbound": {
                "N-N": ["j2_l1"],
                "S-S": [],
            },
            "out_lane_to_signal_id": {
                "j2_l1": "J2",
            },
        },
        "J2": {
            "directions": ["N-N", "S-S"],
            "lane_order": ["j2_l1", "j2_l2"],
            "lane_sets": {
                "N-N": ["j2_l1"],
                "S-S": ["j2_l2"],
            },
            "lane_sets_outbound": {
                "N-N": [],
                "S-S": [],
            },
            "out_lane_to_signal_id": {},
        },
    }
    adapter = ObservationAdapter(
        signal_specs=signal_specs,
        tl_ids=["J1", "J2"],
        layout=layout,
        graph_metadata=None,
    )

    # phase_index, phase_length, lane_mask(2), approaching(2), queued(2), total_wait(2), total_speed(2)
    observations = torch.tensor(
        [
            [1.0, 4.0, 1.0, 1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0],
            [0.0, 2.0, 1.0, 1.0, 1.0, 4.0, 6.0, 2.0, 0.5, 1.5, 3.5, 5.5],
        ],
        dtype=torch.float32,
    )

    wave = adapter.as_state_dict(observations, feature_mode="wave")
    assert wave["J1"] == [7.0, 10.0]
    assert wave["J2"] == [7.0, 6.0]

    mplight = adapter.as_state_dict(observations, feature_mode="mplight")
    assert mplight["J1"] == [1.0, -1.0, 7.0]
    assert mplight["J2"] == [0.0, 6.0, 2.0]

    drq = adapter.as_state_dict(observations, feature_mode="drq")
    assert drq["J1"] == [0.0, 2.0, 11.0, 5.0, 17.0, 1.0, 3.0, 13.0, 7.0, 19.0]
