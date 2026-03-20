from __future__ import annotations

from pathlib import Path

import pytest

from marl_env.sumo_env import TrafficSignalEnv


REPO_ROOT = Path(__file__).resolve().parents[1]

RESCO_SCENARIOS = (
    ("grid4x4", "nets/grid4x4/grid4x4.net.xml", "nets/grid4x4/grid4x4_1.rou.xml", 0),
    ("arterial4x4", "nets/arterial4x4/arterial4x4.net.xml", "nets/arterial4x4/arterial4x4_1.rou.xml", 0),
    ("cologne1", "nets/cologne1/cologne1.net.xml", "nets/cologne1/cologne1.rou.xml", 25200),
    ("cologne3", "nets/cologne3/cologne3.net.xml", "nets/cologne3/cologne3.rou.xml", 25200),
    ("cologne8", "nets/cologne8/cologne8.net.xml", "nets/cologne8/cologne8.rou.xml", 25200),
    ("ingolstadt1", "nets/ingolstadt1/ingolstadt1.net.xml", "nets/ingolstadt1/ingolstadt1.rou.xml", 57600),
    ("ingolstadt7", "nets/ingolstadt7/ingolstadt7.net.xml", "nets/ingolstadt7/ingolstadt7.rou.xml", 57600),
    ("ingolstadt21", "nets/ingolstadt21/ingolstadt21.net.xml", "nets/ingolstadt21/ingolstadt21.rou.xml", 57600),
)


@pytest.mark.sumo
@pytest.mark.parametrize(
    ("scenario_name", "net_rel", "route_rel", "begin_time"),
    RESCO_SCENARIOS,
)
def test_resco_scenarios_reset_successfully(
    tmp_path: Path,
    scenario_name: str,
    net_rel: str,
    route_rel: str,
    begin_time: int,
) -> None:
    env = TrafficSignalEnv(
        net_file=str(REPO_ROOT / net_rel),
        route_file=str(REPO_ROOT / route_rel),
        begin_time=begin_time,
        end_time=begin_time + 60,
        step_length=5,
        output_dir=str(tmp_path / scenario_name),
    )
    try:
        td = env.reset()
        assert env.n_agents > 0
        assert td["agents", "observation"].shape[0] == env.n_agents
        specs = env.get_signal_specs()
        assert set(specs) == set(env.tl_ids)
        for signal_id, spec in specs.items():
            assert int(spec["local_num_actions"]) > 0
            assert isinstance(spec["pair_to_act_map"], dict)
            assert spec["pair_to_act_map"]
            assert max(spec["pair_to_act_map"].values()) + 1 == int(spec["local_num_actions"])
            assert len(env._green_phases[signal_id]) == int(spec["local_num_actions"])
    finally:
        env.close()
