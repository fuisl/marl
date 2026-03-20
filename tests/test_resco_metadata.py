from __future__ import annotations

from marl_env.resco_metadata import (
    SUPPORTED_RESCO_MAPS,
    get_resco_map_metadata,
    load_resco_signal_metadata,
)


def test_supported_resco_maps_are_vendored() -> None:
    data = load_resco_signal_metadata()
    assert set(SUPPORTED_RESCO_MAPS).issubset(data)


def test_homogeneous_maps_expose_identity_pair_to_act_map() -> None:
    for map_name in ("grid4x4", "arterial4x4", "cologne1", "ingolstadt1"):
        metadata = get_resco_map_metadata(map_name=map_name)
        expected = {idx: idx for idx in range(len(metadata["phase_pairs"]))}
        signal_ids = [
            signal_id
            for signal_id, signal_meta in metadata.items()
            if signal_id not in {"phase_pairs", "pair_to_act_map"} and isinstance(signal_meta, dict)
        ]
        assert signal_ids
        for signal_id in signal_ids[:3]:
            assert metadata["pair_to_act_map"][signal_id] == expected
            assert metadata[signal_id]["pair_to_act_map"] == expected


def test_malformed_cologne8_pair_mapping_is_normalized() -> None:
    metadata = get_resco_map_metadata(map_name="cologne8")
    assert metadata["pair_to_act_map"]["32319828"] == {0: 0, 1: 1}
    assert metadata["32319828"]["pair_to_act_map"] == {0: 0, 1: 1}


def test_heterogeneous_maps_expose_contiguous_local_action_mappings() -> None:
    for map_name in ("cologne3", "cologne8", "ingolstadt7", "ingolstadt21"):
        metadata = get_resco_map_metadata(map_name=map_name)
        for signal_id, mapping in metadata["pair_to_act_map"].items():
            assert mapping
            global_indices = sorted(mapping)
            local_indices = sorted(set(mapping.values()))
            assert global_indices[0] >= 0
            assert global_indices[-1] < len(metadata["phase_pairs"])
            assert local_indices == list(range(len(local_indices)))
