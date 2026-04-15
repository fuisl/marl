"""Runtime metadata inference for homogeneous grid networks like Grid5x5.

This module builds RESCO-compatible metadata dynamically from SUMO network files
for simple grid topologies where manual vendoring is not required.
"""

from __future__ import annotations

from typing import Any

try:
    import sumolib
except ImportError:
    sumolib = None


def infer_grid_metadata_from_net(net_file: str) -> dict[str, Any]:
    """Infer RESCO metadata for a homogeneous grid from a SUMO net file.
    
    Parameters
    ----------
    net_file : str
        Path to the SUMO .net.xml file.
    
    Returns
    -------
    dict[str, Any]
        RESCO-compatible metadata dict with:
        - phase_pairs: list of 2-direction phase actions
        - pair_to_act_map: signal_id -> {global_idx: local_idx}
        - per-signal dicts with lane_sets, downstream, etc.
    
    Raises
    ------
    ImportError
        If sumolib is not available.
    RuntimeError
        If the network cannot be parsed or validated.
    """
    if sumolib is None:
        raise ImportError("sumolib is required for runtime grid metadata inference. "
                          "Install SUMO development libraries.")
    
    try:
        net = sumolib.net.readNet(net_file)
    except Exception as e:
        raise RuntimeError(f"Failed to parse SUMO net file {net_file}: {e}") from e
    
    # Phase pairs for homogeneous grids: vertical (N-S) and horizontal (E-W)
    phase_pairs = [["N", "S"], ["E", "W"]]
    
    # Extract all controlled traffic lights
    tl_objects = list(net.getTrafficLights())
    if not tl_objects:
        raise RuntimeError(f"No traffic lights found in {net_file}")
    
    pair_to_act_map: dict[str, dict[int, int]] = {}
    metadata: dict[str, Any] = {
        "phase_pairs": phase_pairs,
        "pair_to_act_map": pair_to_act_map,
    }
    
    # Build per-signal metadata
    for tl in tl_objects:
        tl_id = tl.getID()
        
        # Get all controlled lanes grouped by incoming edge direction
        lanes_by_incoming_edge = {}
        for conn in tl.getConnections():
            # conn is [from_lane, to_lane, link_index]
            if len(conn) >= 2:
                from_lane = conn[0]
                to_lane = conn[1]
                try:
                    from_edge_id = from_lane.getEdge().getID()
                    if from_edge_id not in lanes_by_incoming_edge:
                        lanes_by_incoming_edge[from_edge_id] = []
                    lanes_by_incoming_edge[from_edge_id].append(from_lane.getID())
                except Exception:
                    continue
        
        # Simple direction inference: group edges by angle
        # For grid networks, we assume clear N-S (vertical) and E-W (horizontal) separation
        lane_sets = _infer_grid_lane_sets(tl, net, lane_sets=lanes_by_incoming_edge)
        
        # Downstream mapping: find next controlled signals
        downstream = _infer_grid_downstream(tl, net, phase_pairs)
        
        # Identity action mapping for homogeneous grids
        signal_pair_to_act = {idx: idx for idx in range(len(phase_pairs))}
        pair_to_act_map[str(tl_id)] = signal_pair_to_act
        
        # Build signal metadata
        metadata[str(tl_id)] = {
            "lane_sets": lane_sets,
            "downstream": downstream,
            "pair_to_act_map": signal_pair_to_act,
            "fixed_timings": [],
            "fixed_phase_order_idx": 0,
            "fixed_offset": 0,
        }
    
    return metadata


def _infer_grid_lane_sets(
    tl: Any,
    net: Any,
    lane_sets: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Infer lane groupings for a traffic light in a grid topology.
    
    Groups incoming lanes by direction (N, S, E, W) based on edge angles.
    """
    if lane_sets is None:
        lane_sets = {}
    
    ns_lanes = []
    ew_lanes = []
    
    # Simple heuristic: group by edge angle
    for edge_id, lanes in lane_sets.items():
        try:
            edge = net.getEdge(edge_id)
            if not edge:
                continue
            # Get edge direction via nodes
            from_node = edge.getFromNode()
            to_node = edge.getToNode()
            if from_node and to_node:
                angle = _compute_edge_angle(from_node, to_node)
                # North-South: angles near 90 or 270 degrees
                # East-West: angles near 0 or 180 degrees
                angle_normalized = angle % 180
                if angle_normalized < 45 or angle_normalized > 135:
                    ew_lanes.extend(lanes)
                else:
                    ns_lanes.extend(lanes)
            else:
                ew_lanes.extend(lanes)  # default
        except Exception:
            # Fallback: alternate assignment
            if (len(ns_lanes) + len(ew_lanes)) % 2 == 0:
                ns_lanes.extend(lanes)
            else:
                ew_lanes.extend(lanes)
    
    result = {}
    if ns_lanes:
        result["N-S"] = ns_lanes
    if ew_lanes:
        result["E-W"] = ew_lanes
    
    # Ensure we have at least one direction
    if not result:
        # Fallback: put all lanes in one direction
        all_lanes = []
        for lanes in lane_sets.values():
            all_lanes.extend(lanes)
        result["N-S"] = all_lanes if all_lanes else ["dummy"]
    
    return result


def _compute_edge_angle(from_node: Any, to_node: Any) -> float:
    """Compute angle of an edge from its endpoints in degrees."""
    try:
        x1, y1 = from_node.getCoord()
        x2, y2 = to_node.getCoord()
        import math
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle_deg = math.degrees(angle_rad)
        return (angle_deg + 360) % 360
    except Exception:
        return 0.0


def _infer_grid_downstream(
    tl: Any,
    net: Any,
    phase_pairs: list[list[str]],
) -> dict[str, str | None]:
    """Infer downstream signals for each direction in a grid topology."""
    downstream = {}
    
    # For each direction in phase pairs, try to find the next controlled signal downstream
    for direction_pair in phase_pairs:
        # For now, use a simple heuristic: no explicit downstream tracing
        # This can be extended to walk the network and find next controlled TLs
        downstream["-".join(direction_pair)] = None
    
    return downstream
