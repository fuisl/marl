"""Export graph topology for visualization and analysis.

Usage::

    python scripts/export_graph_topology.py \
        --net-file nets/grid5x5/grid5x5.net.xml \
        --out runs/grid5x5_topology.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from marl_env.graph_builder import GraphBuilder


def export_topology(
    net_file: str,
    out_path: str,
    mode: str = "all_intersections",
    validate: bool = True,
) -> None:
    """Export graph topology from a SUMO network file.
    
    Parameters
    ----------
    net_file : str
        Path to SUMO .net.xml file.
    out_path : str
        Output JSON file path.
    mode : str
        Graph builder mode: "all_intersections", "walk_to_light", or "original".
    validate : bool
        Whether to validate expected topology for bundled scenarios.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build graph
    builder = GraphBuilder(net_file=net_file, tl_ids=None, mode=mode)
    edge_index = builder.edge_index
    edge_attr = builder.edge_attr
    num_nodes = builder.num_nodes
    
    # Convert to Python dicts
    edges_list = edge_index.t().tolist()  # List of [src, dst]
    edge_attrs_list = edge_attr.tolist() if edge_attr is not None else None
    
    # Collect node IDs from network
    try:
        import sumolib
        net = sumolib.net.readNet(net_file)
        node_ids = [node.getID() for node in net.getNodes()]
    except Exception:
        node_ids = [str(i) for i in range(num_nodes)]
    
    # Build export dict
    topology = {
        "num_nodes": num_nodes,
        "num_edges": len(edges_list),
        "node_ids": node_ids[:num_nodes],  # Ensure we don't exceed actual nodes
        "edges": edges_list,  # Each edge is [src_idx, dst_idx]
        "edge_attributes": edge_attrs_list,  # Each attr is [distance, n_lanes] if available
        "mode": mode,
        "net_file": str(net_file),
    }
    
    # Validation for known scenarios
    if validate:
        net_name = Path(net_file).stem
        if net_name == "grid5x5":
            expected_nodes = 25
            expected_edges = 80
            if topology["num_nodes"] != expected_nodes:
                print(
                    f"WARNING: grid5x5 expected {expected_nodes} nodes but got {topology['num_nodes']}"
                )
            if topology["num_edges"] < expected_edges * 0.9:
                print(
                    f"WARNING: grid5x5 expected ~{expected_edges} edges but got {topology['num_edges']}"
                )
            if edge_attrs_list is not None and len(edge_attrs_list) > 0:
                first_attr = edge_attrs_list[0]
                if len(first_attr) != 2:
                    print(
                        f"WARNING: grid5x5 edge attributes should have 2 columns but got {len(first_attr)}"
                    )
            print(f"✓ grid5x5 topology: {topology['num_nodes']} nodes, {topology['num_edges']} edges")
    
    # Write to JSON
    with open(out_path, "w") as f:
        json.dump(topology, f, indent=2)
    print(f"Wrote topology to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export graph topology from a SUMO network."
    )
    parser.add_argument(
        "--net-file",
        type=str,
        required=True,
        help="Path to SUMO .net.xml file",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all_intersections", "walk_to_light", "original"],
        default="all_intersections",
        help="Graph builder mode",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip built-in validation checks",
    )
    
    args = parser.parse_args()
    export_topology(
        net_file=args.net_file,
        out_path=args.out,
        mode=args.mode,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
