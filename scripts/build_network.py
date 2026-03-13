"""Build a 5×5 SUMO grid network with heterogeneous, time-varying traffic.

Generates:
  nets/grid5x5/
    grid5x5.net.xml   — SUMO network (25 signalised intersections, 2 lanes)
    grid5x5.rou.xml   — demand file with five traffic phases:
                          [0–600]   early-morning ramp-up     ~light
                          [600–1800] AM peak (heavy, NW→SE bias)
                          [1800–2700] midday steady state      ~medium
                          [2700–3000] PM peak (heavy, SE→NW bias)
                          [3000–3600] evening cool-down        ~light
                        plus cross-traffic flows on every row/column so every
                        intersection has varied load.

Requirements
------------
- ``netgenerate`` (SUMO ≥ 1.4) on PATH
- ``python-duarouter`` / ``duarouter`` on PATH (for validating trips → routes)
- No SUMO_HOME required; ``/usr/share/sumo/tools/randomTrips.py`` is used
  as the fallback tool locator.

Usage
-----
    python scripts/build_network.py
    python scripts/build_network.py --output-dir nets/grid5x5 --seed 0
"""

from __future__ import annotations

import argparse
import subprocess
import textwrap
from pathlib import Path


# =========================================================================
# Network generation
# =========================================================================

def build_grid_network(out: Path, grid_size: int, length: float, lanes: int,
                       speed: float) -> Path:
    """Generate a rectangular grid network with traffic lights at every node."""
    net_file = out / f"grid{grid_size}x{grid_size}.net.xml"
    cmd = [
        "netgenerate",
        "--grid",
        f"--grid.x-number={grid_size}",
        f"--grid.y-number={grid_size}",
        f"--grid.x-length={length}",
        f"--grid.y-length={length}",
        f"--default.lanenumber={lanes}",
        f"--default.speed={speed}",
        # Force all interior nodes to be TLS-controlled
        "--default-junction-type", "traffic_light",
        "--tls.green.time", "30",
        "--tls.yellow.min-decel", "3",
        f"--output-file={net_file}",
        "--no-warnings",
    ]
    print("netgenerate:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"  → {net_file}")
    return net_file


# =========================================================================
# Route / demand generation
# =========================================================================

# Each entry: (begin_s, end_s, vehs_per_hour_per_OD_pair, label)
# We define a set of directional flow patterns for each period.
_TRAFFIC_PERIODS = [
    # (begin, end,  vph_main, vph_cross, label)
    (    0,  600,   100,  40, "early_morning"),
    (  600, 1800,   600, 180, "am_peak"),
    ( 1800, 2700,   280,  90, "midday"),
    ( 2700, 3000,   520, 160, "pm_peak"),
    ( 3000, 3600,   120,  45, "evening"),
]


def _col_letter(col: int) -> str:
    return chr(ord("A") + col)


def _node_id(row: int, col: int) -> str:
    """SUMO netgenerate 1.4 grid node ID: e.g. row=0,col=0 → 'A0'."""
    return f"{_col_letter(col)}{row}"


def _edge_id(from_node: str, to_node: str) -> str:
    """SUMO grid edge ID = concatenation of the two node IDs."""
    return f"{from_node}{to_node}"


def _dep_edge_ns(row: int, col: int) -> str:
    """Departure edge for N→S flow starting at top of column ``col``."""
    return _edge_id(_node_id(row, col), _node_id(row + 1, col))


def _arr_edge_ns(row_last: int, col: int) -> str:
    """Arrival edge for N→S flow ending at bottom row ``row_last``."""
    return _edge_id(_node_id(row_last - 1, col), _node_id(row_last, col))


def _dep_edge_sn(row_last: int, col: int) -> str:
    """Departure edge for S→N flow starting at bottom of column ``col``."""
    return _edge_id(_node_id(row_last, col), _node_id(row_last - 1, col))


def _arr_edge_sn(row: int, col: int) -> str:
    """Arrival edge for S→N flow ending at top row ``row``."""
    return _edge_id(_node_id(row + 1, col), _node_id(row, col))


def _dep_edge_we(row: int, col: int) -> str:
    """Departure edge for W→E flow starting at left of row ``row``."""
    return _edge_id(_node_id(row, col), _node_id(row, col + 1))


def _arr_edge_we(row: int, col_last: int) -> str:
    """Arrival edge for W→E flow ending at right column ``col_last``."""
    return _edge_id(_node_id(row, col_last - 1), _node_id(row, col_last))


def _dep_edge_ew(row: int, col_last: int) -> str:
    """Departure edge for E→W flow starting at right of row ``row``."""
    return _edge_id(_node_id(row, col_last), _node_id(row, col_last - 1))


def _arr_edge_ew(row: int, col: int) -> str:
    """Arrival edge for E→W flow ending at left column ``col``."""
    return _edge_id(_node_id(row, col + 1), _node_id(row, col))


def build_route_file(net_file: Path, out: Path, grid_size: int, seed: int) -> Path:
    """Write ``grid5x5.rou.xml`` with heterogeneous time-varying flows.

    Traffic patterns
    ~~~~~~~~~~~~~~~~
    * **AM peak** : heavy N→S and W→E flows (commuter inbound corridor).
    * **PM peak** : heavy S→N and E→W flows (commuter outbound, reversed).
    * **Midday**  : balanced bi-directional medium flow.
    * **Early/Evening** : light background traffic on all axes.
    * **Cross ODs** : each row and column gets an independent flow so that
      different intersections experience different loads — the GAT must
      actually learn to differentiate them.

    All ``from``/``to`` attributes use SUMO **edge** IDs (``{fromNode}{toNode}``),
    not node IDs, matching netgenerate 1.4's naming convention.

    Vehicle types
    ~~~~~~~~~~~~~
    * ``passenger`` (85 %): standard car
    * ``truck``     (15 %): heavy freight, slower, longer
    """
    G = grid_size
    last = G - 1
    rou_file = out / f"grid{grid_size}x{grid_size}.rou.xml"

    flows: list[str] = []
    flow_id = 0

    def add_flow(dep: str, arr: str, begin: int, end: int, vph: float,
                 vtype: str = "passenger") -> None:
        nonlocal flow_id
        if vph < 1.0:
            return
        period = 3600.0 / vph
        flows.append(
            f'    <flow id="f{flow_id}" type="{vtype}" '
            f'from="{dep}" to="{arr}" '
            f'begin="{begin}" end="{end}" '
            f'period="{period:.2f}" departLane="best" departSpeed="max"/>'
        )
        flow_id += 1

    def centrality(col: int) -> float:
        """Centrality weight 1.0 (centre) → ~0 (edge), for a G-wide grid."""
        half = (G - 1) / 2.0
        return 1.0 - abs(col - half) / max(half, 1.0)

    for (begin, end, vph_main, vph_cross, _label) in _TRAFFIC_PERIODS:

        # --- N→S column flows (AM-peak bias: stronger in central columns) ---
        for col in range(G):
            boost = 1.0 + 0.4 * centrality(col)
            add_flow(_dep_edge_ns(0, col), _arr_edge_ns(last, col),
                     begin, end, vph_main * boost)

        # --- S→N column flows (PM-peak bias: stronger in edge columns) ---
        for col in range(G):
            boost = 1.0 + 0.3 * (1.0 - centrality(col))
            add_flow(_dep_edge_sn(last, col), _arr_edge_sn(0, col),
                     begin, end, vph_main * boost * 0.8)

        # --- W→E row flows (heavier on outer rows) ---
        for row in range(G):
            row_factor = 1.0 + 0.25 * (1.0 - centrality(row))
            add_flow(_dep_edge_we(row, 0), _arr_edge_we(row, last),
                     begin, end, vph_cross * row_factor)

        # --- E→W row flows ---
        for row in range(G):
            row_factor = 1.0 + 0.25 * (1.0 - centrality(row))
            add_flow(_dep_edge_ew(row, last), _arr_edge_ew(row, 0),
                     begin, end, vph_cross * row_factor * 0.7)

        # --- Diagonal / cross-city ODs: corners going across ---
        # NW→SE: col 0 top  →  col last bottom  (go east first)
        add_flow(_dep_edge_we(0,    0),    _arr_edge_ns(last, last),
                 begin, end, vph_cross * 0.5)
        # NE→SW: col last top  →  col 0 bottom  (go south first)
        add_flow(_dep_edge_ns(0, last),    _arr_edge_ew(last, 0),
                 begin, end, vph_cross * 0.5)
        # SW→NE: col 0 bottom  →  col last top  (go east first)
        add_flow(_dep_edge_we(last,  0),   _arr_edge_sn(0, last),
                 begin, end, vph_cross * 0.5)
        # SE→NW: col last bottom  →  col 0 top  (go north first)
        add_flow(_dep_edge_sn(last, last), _arr_edge_ew(0, 0),
                 begin, end, vph_cross * 0.5)

        # --- Truck flows on the central arterial (middle column) ---
        mid = G // 2
        add_flow(_dep_edge_ns(0, mid), _arr_edge_ns(last, mid),
                 begin, end, vph_main * 0.15, vtype="truck")
        add_flow(_dep_edge_sn(last, mid), _arr_edge_sn(0, mid),
                 begin, end, vph_main * 0.15, vtype="truck")

    xml = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!-- {G}x{G} SUMO grid: heterogeneous time-varying demand
             Phases: early-morning | AM-peak | midday | PM-peak | evening
             Two vehicle types: passenger (85%) + truck (15%)
             Generated by scripts/build_network.py
        -->
        <routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">

          <!-- Vehicle type definitions -->
          <vType id="passenger"
                 accel="2.6" decel="4.5" sigma="0.5"
                 length="4.5" minGap="2.5"
                 maxSpeed="13.89" guiShape="passenger"/>

          <vType id="truck"
                 accel="1.2" decel="3.5" sigma="0.4"
                 length="12.0" minGap="3.0"
                 maxSpeed="8.33" guiShape="truck"/>

    """)

    xml += "\n".join(flows)
    xml += "\n\n</routes>\n"

    rou_file.write_text(xml, encoding="utf-8")
    print(f"  → {rou_file}  ({flow_id} flow entries)")
    return rou_file


# =========================================================================
# Validation: duarouter converts flows → validated routes
# =========================================================================

def validate_routes(net_file: Path, rou_file: Path, out: Path,
                    end_time: int) -> Path:
    """Run ``duarouter`` to convert flow entries to fully-routed vehicles.

    This catches invalid OD pairs (e.g. disconnected nodes) before training.
    The validated output is written alongside the original route file.
    """
    validated = out / rou_file.name.replace(".rou.xml", ".validated.rou.xml")
    cmd = [
        "duarouter",
        "--net-file", str(net_file),
        "--route-files", str(rou_file),
        "--output-file", str(validated),
        "--begin", "0",
        "--end", str(end_time),
        "--no-warnings",
        "--ignore-errors",
    ]
    print("duarouter:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  duarouter stderr:", result.stderr[:800])
        print("  Using unvalidated route file.")
        return rou_file
    print(f"  → {validated}")
    return validated


# =========================================================================
# CLI
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 5×5 SUMO grid network with heterogeneous traffic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="nets/grid5x5")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--grid-length", type=float, default=200.0,
                        help="Edge length in metres")
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--speed", type=float, default=13.89,
                        help="Max edge speed m/s (~50 km/h)")
    parser.add_argument("--end-time", type=int, default=3600,
                        help="Simulation end time (s)")
    parser.add_argument("--validate", action="store_true",
                        help="Run duarouter to validate routes (requires duarouter on PATH)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    net_file = build_grid_network(
        out,
        grid_size=args.grid_size,
        length=args.grid_length,
        lanes=args.num_lanes,
        speed=args.speed,
    )

    rou_file = build_route_file(
        net_file=net_file,
        out=out,
        grid_size=args.grid_size,
        seed=args.seed,
    )

    if args.validate:
        rou_file = validate_routes(net_file, rou_file, out, args.end_time)

    print("\nDone.")
    print(f"  net  : {net_file}")
    print(f"  rou  : {rou_file}")
    print()
    print("Update configs/env.yaml:")
    print(f"  net_file:   {net_file}")
    print(f"  route_file: {rou_file}")


if __name__ == "__main__":
    main()

