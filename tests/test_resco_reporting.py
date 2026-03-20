from __future__ import annotations

import csv
from pathlib import Path

from marl_env.resco_metadata import get_resco_map_metadata
from marl_env.resco_reporting import (
    load_episode_raw_metrics,
    parse_metrics_csv,
    parse_tripinfo_metrics,
    to_public_metrics,
)


def test_resco_metadata_loader_returns_supported_map() -> None:
    metadata = get_resco_map_metadata(map_name="grid4x4")
    assert "phase_pairs" in metadata
    assert "A0" in metadata
    assert metadata["A0"]["fixed_timings"]


def test_resco_tripinfo_and_csv_parsers_match_expected_averages(tmp_path: Path) -> None:
    tripinfo_path = tmp_path / "tripinfo_1.xml"
    tripinfo_path.write_text(
        """<tripinfos>
<tripinfo id="veh0" duration="10" waitingTime="2" timeLoss="3" departDelay="1" />
<tripinfo id="ghost_0" duration="999" waitingTime="999" timeLoss="999" departDelay="999" />
<tripinfo id="veh1" duration="20" waitingTime="4" timeLoss="5" departDelay="2" />
</tripinfos>
""",
        encoding="utf-8",
    )

    metrics_path = tmp_path / "metrics_1.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rewards", "max_queues", "queue_lengths", "vehicles", "phase_length"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rewards": "{'A0': 1.0, 'B0': 3.0}",
                "max_queues": "{'A0': 2.0, 'B0': 4.0}",
                "queue_lengths": "{'A0': 2.0, 'B0': 0.0}",
                "vehicles": "{'A0': 10.0, 'B0': 10.0}",
                "phase_length": "{'A0': 5.0, 'B0': 7.0}",
            }
        )
        writer.writerow(
            {
                "rewards": "{'A0': 2.0, 'B0': 4.0}",
                "max_queues": "{'A0': 6.0, 'B0': 2.0}",
                "queue_lengths": "{'A0': 4.0, 'B0': 2.0}",
                "vehicles": "{'A0': 10.0, 'B0': 10.0}",
                "phase_length": "{'A0': 3.0, 'B0': 5.0}",
            }
        )

    tripinfo_metrics = parse_tripinfo_metrics(tripinfo_path)
    assert tripinfo_metrics == {
        "duration": 15.0,
        "waitingTime": 3.0,
        "timeLoss": 5.5,
    }

    csv_metrics = parse_metrics_csv(metrics_path)
    assert csv_metrics == {
        "rewards": 2.5,
        "max_queues": 3.5,
        "queue_lengths": 2.0,
        "vehicles": 10.0,
        "phase_length": 5.0,
    }

    raw_metrics = load_episode_raw_metrics(
        tripinfo_path=tripinfo_path,
        metrics_path=metrics_path,
        global_reward=42.0,
    )
    assert raw_metrics["global_reward"] == 42.0

    public_metrics = to_public_metrics(raw_metrics)
    assert public_metrics == {
        "Avg Duration": 15.0,
        "Avg Waiting Time": 3.0,
        "Avg Time Loss": 5.5,
        "Avg Queue Length": 2.0,
        "Avg Reward": 2.5,
        "Global Reward": 42.0,
    }
