"""Build a SUMO network and generate a route file for testing.

This is a utility script that creates a simple grid network using
SUMO's netgenerate tool and randomTrips.py for route generation.

Usage::

    python scripts/build_network.py --output-dir nets/grid4x4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_grid_network(
    output_dir: str,
    grid_size: int = 4,
    grid_length: float = 200.0,
    num_lanes: int = 2,
    speed_limit: float = 13.89,  # ~50 km/h
) -> str:
    """Generate a grid network using SUMO's ``netgenerate``.

    Returns path to the generated ``.net.xml``.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    net_file = os.path.join(output_dir, f"grid{grid_size}x{grid_size}.net.xml")

    cmd = [
        "netgenerate",
        "--grid",
        f"--grid.x-number={grid_size}",
        f"--grid.y-number={grid_size}",
        f"--grid.x-length={grid_length}",
        f"--grid.y-length={grid_length}",
        f"--default.lanenumber={num_lanes}",
        f"--default.speed={speed_limit}",
        "--tls.guess", "true",
        f"--output-file={net_file}",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Network written to {net_file}")
    return net_file


def generate_routes(
    net_file: str,
    output_dir: str,
    n_vehicles: int = 1000,
    begin_time: int = 0,
    end_time: int = 3600,
    seed: int = 42,
) -> str:
    """Generate random trips using SUMO's ``randomTrips.py``.

    Returns path to the generated ``.rou.xml``.
    """
    route_file = os.path.join(output_dir, "routes.rou.xml")
    trip_file = os.path.join(output_dir, "trips.trips.xml")

    # Locate randomTrips.py from SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME", "")
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.isfile(random_trips):
        print(
            "Warning: SUMO_HOME not set or randomTrips.py not found. "
            "Skipping route generation."
        )
        return ""

    period = (end_time - begin_time) / n_vehicles

    cmd = [
        sys.executable, random_trips,
        "-n", net_file,
        "-r", route_file,
        "-o", trip_file,
        "-b", str(begin_time),
        "-e", str(end_time),
        "-p", f"{period:.4f}",
        "--seed", str(seed),
        "--validate",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Routes written to {route_file}")
    return route_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SUMO grid network.")
    parser.add_argument("--output-dir", type=str, default="nets/grid4x4")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--grid-length", type=float, default=200.0)
    parser.add_argument("--num-lanes", type=int, default=2)
    parser.add_argument("--n-vehicles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    net_file = build_grid_network(
        args.output_dir,
        grid_size=args.grid_size,
        grid_length=args.grid_length,
        num_lanes=args.num_lanes,
    )
    generate_routes(
        net_file,
        args.output_dir,
        n_vehicles=args.n_vehicles,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
