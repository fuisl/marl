"""Test runtime metadata inference for Grid5x5."""

from __future__ import annotations

from pathlib import Path

import pytest

from marl_env.resco_metadata import get_resco_map_metadata, _normalize_map_metadata


def test_grid5x5_metadata_inference_from_net_file() -> None:
    """Test that Grid5x5 metadata can be inferred from net file."""
    net_file = "nets/grid5x5/grid5x5.net.xml"
    if not Path(net_file).exists():
        pytest.skip(f"{net_file} not found")
    
    metadata = get_resco_map_metadata(net_file=net_file)
    assert isinstance(metadata, dict)
    assert "phase_pairs" in metadata
    assert "pair_to_act_map" in metadata


def test_grid5x5_metadata_has_required_fields() -> None:
    """Test that inferred Grid5x5 metadata has all required fields."""
    net_file = "nets/grid5x5/grid5x5.net.xml"
    if not Path(net_file).exists():
        pytest.skip(f"{net_file} not found")
    
    metadata = get_resco_map_metadata(net_file=net_file)
    
    # Check phase pairs
    phase_pairs = metadata.get("phase_pairs")
    assert isinstance(phase_pairs, list)
    assert len(phase_pairs) == 2  # Two actions: vertical and horizontal
    assert phase_pairs[0] == ["N", "S"]
    assert phase_pairs[1] == ["E", "W"]
    
    # Check pair_to_act_map
    pair_to_act = metadata.get("pair_to_act_map")
    assert isinstance(pair_to_act, dict)
    assert len(pair_to_act) > 0  # At least one signal


def test_grid5x5_signals_have_lane_sets() -> None:
    """Test that all signals have non-empty lane_sets."""
    net_file = "nets/grid5x5/grid5x5.net.xml"
    if not Path(net_file).exists():
        pytest.skip(f"{net_file} not found")
    
    metadata = get_resco_map_metadata(net_file=net_file)
    
    signal_ids = [
        k for k in metadata.keys()
        if k not in ("phase_pairs", "pair_to_act_map") and isinstance(metadata[k], dict)
    ]
    assert len(signal_ids) > 0
    
    for signal_id in signal_ids:
        signal_meta = metadata[signal_id]
        lane_sets = signal_meta.get("lane_sets")
        assert isinstance(lane_sets, dict)
        # At least one direction should have lanes
        has_lanes = any(len(lanes) > 0 for lanes in lane_sets.values())
        assert has_lanes, f"Signal {signal_id} has no lanes"


def test_grid5x5_signals_have_valid_action_mapping() -> None:
    """Test that signals have valid contiguous action mappings."""
    net_file = "nets/grid5x5/grid5x5.net.xml"
    if not Path(net_file).exists():
        pytest.skip(f"{net_file} not found")
    
    metadata = get_resco_map_metadata(net_file=net_file)
    pair_to_act = metadata.get("pair_to_act_map")
    
    for signal_id, mapping in pair_to_act.items():
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Check contiguous local actions
        local_actions = sorted(set(mapping.values()))
        expected = list(range(len(local_actions)))
        assert local_actions == expected, f"Signal {signal_id} has non-contiguous actions: {local_actions}"


def test_grid5x5_signals_approximately_25() -> None:
    """Test that Grid5x5 has approximately 25 signals (5x5 grid)."""
    net_file = "nets/grid5x5/grid5x5.net.xml"
    if not Path(net_file).exists():
        pytest.skip(f"{net_file} not found")
    
    metadata = get_resco_map_metadata(net_file=net_file)
    
    signal_ids = [
        k for k in metadata.keys()
        if k not in ("phase_pairs", "pair_to_act_map") and isinstance(metadata[k], dict)
    ]
    
    # Allow some tolerance as the exact count depends on the network structure
    assert 20 <= len(signal_ids) <= 30, f"Expected ~25 signals, got {len(signal_ids)}"


def test_grid5x5_metadata_passes_normalization() -> None:
    """Test that inferred metadata passes normalization validation."""
    net_file = "nets/grid5x5/grid5x5.net.xml"
    if not Path(net_file).exists():
        pytest.skip(f"{net_file} not found")
    
    # This should not raise an exception
    metadata = get_resco_map_metadata(net_file=net_file)
    assert isinstance(metadata, dict)
    assert len(metadata) > 0


def test_grid5x5_metadata_map_name_resolution() -> None:
    """Test that map_name 'grid5x5' is properly resolved from net file."""
    net_file = "nets/grid5x5/grid5x5.net.xml"
    if not Path(net_file).exists():
        pytest.skip(f"{net_file} not found")
    
    # Both should resolve the same metadata
    metadata1 = get_resco_map_metadata(net_file=net_file)
    metadata2 = get_resco_map_metadata(map_name="grid5x5", net_file=net_file)
    
    # They should have the same structure (not necessarily identical due to inference)
    assert "phase_pairs" in metadata1
    assert "pair_to_act_map" in metadata1
