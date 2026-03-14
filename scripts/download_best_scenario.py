"""Download and extract the BeST SUMO scenario into the nets directory.

Usage:
    python scripts/download_best_scenario.py
    python scripts/download_best_scenario.py --output-dir nets
    python scripts/download_best_scenario.py --no-record
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import urllib.request
import zipfile


DOWNLOAD_URL = (
    "https://www.dcaiti.tu-berlin.de/research/simulation/downloads/get/"
    "best-scenario-v2.zip"
)
ZIP_NAME = "best-scenario.zip"
MARKER_FILE = "berlin.sumocfg"


def report_progress(block_n: int, block_size: int, complete_size: int) -> None:
    if complete_size <= 0:
        print("Downloading BeST scenario ...", end="\r")
        return
    progress = 100.0 * min(1.0, (block_n * block_size) / complete_size)
    print(f"Downloading BeST scenario ... {int(progress)}%", end="\r")


def download_best(output_dir: Path, record: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_path = output_dir / MARKER_FILE
    if marker_path.exists():
        print(f"BeST scenario already exists at {marker_path}, skipping")
        return

    zip_path = output_dir / ZIP_NAME
    url = f"{DOWNLOAD_URL}?record=true" if record else DOWNLOAD_URL

    print("Downloading BeST scenario ...", end="\r")
    urllib.request.urlretrieve(url, filename=str(zip_path), reporthook=report_progress)
    print("\nUnzipping ...")

    with zipfile.ZipFile(zip_path, "r") as best_zip:
        best_zip.extractall(path=output_dir)

    os.remove(zip_path)
    print(f"Finished. Scenario extracted to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download BeST scenario and extract it under nets/"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nets"),
        help="Directory where the scenario archive is extracted (default: nets)",
    )
    parser.add_argument(
        "--record",
        dest="record",
        action="store_true",
        help="Record download statistics on dcaiti.tu-berlin.de (default)",
    )
    parser.add_argument(
        "--no-record",
        dest="record",
        action="store_false",
        help="Do not record download statistics",
    )
    parser.set_defaults(record=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_best(output_dir=args.output_dir, record=args.record)


if __name__ == "__main__":
    main()
