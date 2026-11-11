# 계획 대기 산식과 RAPID 실제 가동률 계산을 검증하는 전용 회귀 테스트.
from __future__ import annotations

import importlib
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.S3_path_planning import S41_execution_metrics as execution_metrics


REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = (
    REPO
    / "docs"
    / "experiments"
    / "2026-08-30_hrl-v4-paper-pointwise"
    / "results"
)


def test_paper_schedule_metrics_decomposes_wait_and_collision() -> None:
    metrics = execution_metrics.paper_schedule_metrics(
        [4.0, 7.0, 8.0],
        [6.0, 9.0, 10.0],
        10.0,
        collision_waits_sec=[1.0, 0.5, 0.0],
        collision_wait_status="measured",
    )

    assert metrics["robot_process_wait_sec"] == pytest.approx([2.0, 2.0, 2.0])
    assert metrics["robot_completion_wait_sec"] == pytest.approx([4.0, 1.0, 0.0])
    assert metrics["robot_total_wait_sec"] == pytest.approx([6.0, 3.0, 2.0])
    assert metrics["average_work_sec"] == pytest.approx(19.0 / 3.0)
    assert metrics["average_total_wait_sec"] == pytest.approx(11.0 / 3.0)
    assert metrics["paper_wait_ratio_pct"] == pytest.approx(100.0 * 11.0 / 30.0)
    assert metrics["paper_process_wait_share_pct"] == pytest.approx(
        100.0 * 6.0 / 11.0
    )
    assert metrics["collision_wait_share_of_total_wait_pct"] == pytest.approx(
        100.0 * 1.5 / 11.0
    )
    assert (
        metrics["paper_wait_ratio_pct"]
        + metrics["schedule_motion_utilization_pct"]
    ) == pytest.approx(100.0)


def test_paper_schedule_metrics_rejects_impossible_waits() -> None:
    with pytest.raises(ValueError, match="process wait"):
        execution_metrics.paper_schedule_metrics([6.0], [5.0], 10.0)
    with pytest.raises(ValueError, match="collision wait exceeds"):
        execution_metrics.paper_schedule_metrics(
            [4.0], [5.0], 10.0, collision_waits_sec=[2.0]
        )


def _waypoint(robot: int, t: float, x: float, kind: str) -> dict:
    return {
        "robot": robot,
        "t": t,
        "x": x,
        "y": 0.0,
        "z": 0.0,
        "kind": kind,
    }


def test_rapid_execution_metrics_excludes_wait_and_sweeps_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod_dir = tmp_path / "rapid"
    mod_dir.mkdir()
    waypoints = [
        _waypoint(0, 0.0, 0.0, "park"),
        _waypoint(0, 2.0, 2.0, "transit"),
        _waypoint(0, 2.0, 2.0, "wait_start"),
        _waypoint(0, 5.0, 2.0, "wait_end"),
        _waypoint(0, 7.0, 4.0, "weld"),
        _waypoint(0, 10.0, 4.0, "wait_end"),
        _waypoint(1, 0.0, 0.0, "park"),
        _waypoint(1, 1.0, 0.0, "wait_end"),
        _waypoint(1, 2.0, 1.0, "transit"),
        _waypoint(1, 4.0, 3.0, "weld"),
        _waypoint(1, 6.0, 5.0, "transit"),
        _waypoint(1, 10.0, 5.0, "wait_end"),
        _waypoint(2, 0.0, 0.0, "park"),
        _waypoint(2, 3.0, 0.0, "wait_end"),
        _waypoint(2, 6.0, 3.0, "transit"),
        _waypoint(2, 8.0, 5.0, "weld"),
        _waypoint(2, 10.0, 5.0, "wait_end"),
    ]
    tasks = [
        SimpleNamespace(
            robot=0,
            beads=[
                SimpleNamespace(t_start=5.0, t_end=6.5),
                SimpleNamespace(t_start=6.0, t_end=7.0),
            ],
        ),
        SimpleNamespace(
            robot=1,
            beads=[SimpleNamespace(t_start=2.0, t_end=4.0)],
        ),
        SimpleNamespace(
            robot=2,
            beads=[SimpleNamespace(t_start=6.0, t_end=8.0)],
        ),
    ]

    def fake_loader(_mod_dir: str):
        return [], tasks, {"waypoints": waypoints}

    monkeypatch.setattr(execution_metrics, "_events_from_mod_dir", fake_loader)
    metrics = execution_metrics.rapid_execution_metrics(mod_dir)

    assert metrics["rapid_total_end_sec"] == pytest.approx(10.0)
    assert metrics["robot_active_sec"] == pytest.approx([4.0, 5.0, 5.0])
    assert metrics["robot_weld_sec"] == pytest.approx([2.0, 2.0, 2.0])
    assert metrics["robot_first_active_start_sec"] == pytest.approx(
        [0.0, 1.0, 3.0]
    )
    assert metrics["robot_last_active_end_sec"] == pytest.approx([7.0, 6.0, 8.0])
    assert metrics["robot_initial_wait_sec"] == pytest.approx([0.0, 1.0, 3.0])
    assert metrics["robot_process_wait_sec"] == pytest.approx([3.0, 0.0, 0.0])
    assert metrics["robot_completion_wait_sec"] == pytest.approx([3.0, 4.0, 2.0])
    assert metrics["robot_total_wait_sec"] == pytest.approx([6.0, 5.0, 5.0])
    assert metrics["robot_wait_ratio_pct"] == pytest.approx([60.0, 50.0, 50.0])
    assert metrics["rapid_wait_ratio_pct"] == pytest.approx(100.0 * 16.0 / 30.0)
    assert metrics["rapid_process_wait_share_pct"] == pytest.approx(
        100.0 * 3.0 / 16.0
    )
    assert metrics["rapid_active_utilization_pct"] == pytest.approx(100.0 * 14.0 / 30.0)
    assert metrics["rapid_weld_utilization_pct"] == pytest.approx(20.0)
    assert metrics["rapid_concurrent_2plus_sec"] == pytest.approx(5.0)
    assert metrics["rapid_concurrent_3_sec"] == pytest.approx(1.0)
    assert metrics["rapid_concurrent_2plus_pct"] == pytest.approx(50.0)
    assert metrics["rapid_concurrent_3_pct"] == pytest.approx(10.0)


def test_current_l1_schedule_regression() -> None:
    cp_path = EXPERIMENT / "cpsat" / "L1" / "schedule.json"
    hrl_path = EXPERIMENT / "hrl_smoke" / "layers" / "layer_01" / "result.json"
    if not cp_path.is_file() or not hrl_path.is_file():
        pytest.skip("current CP-SAT and HRL L1 artifacts are not available")

    cp = json.loads(cp_path.read_text(encoding="utf-8"))
    units_per_sec = float(cp["total_makespan"]) / float(cp["total_makespan_sec"])
    cp_work = []
    cp_completion = []
    for robot in range(3):
        entries = [entry for entry in cp["entries_summary"] if entry["robot"] == robot]
        cp_work.append(sum(float(entry["duration_sec"]) for entry in entries))
        cp_completion.append(max(float(entry["end"]) for entry in entries) / units_per_sec)
    cp_metrics = execution_metrics.paper_schedule_metrics(
        cp_work,
        cp_completion,
        float(cp["total_makespan_sec"]),
        collision_waits_sec=[0.0, 0.0, 0.0],
        collision_wait_status="deduced_exactly_from_zero_process_idle",
    )
    assert cp_metrics["robot_work_sec"] == pytest.approx([1096.4, 1538.0, 1589.7])
    assert cp_metrics["robot_process_wait_sec"] == pytest.approx([83.9, 0.0, 0.0])
    assert cp_metrics["robot_completion_wait_sec"] == pytest.approx([409.4, 51.7, 0.0])
    assert cp_metrics["paper_wait_ratio_pct"] == pytest.approx(11.4277326958965)
    assert cp_metrics["paper_process_wait_share_pct"] == pytest.approx(
        15.39449541284401
    )

    hrl = json.loads(hrl_path.read_text(encoding="utf-8"))
    hrl_work = [0.0, 0.0, 0.0]
    hrl_completion = [0.0, 0.0, 0.0]
    for record in hrl["records"]:
        robot = int(record["robot"])
        hrl_work[robot] += float(record["duration_sec"])
        hrl_completion[robot] = max(
            hrl_completion[robot], float(record["end_sec"])
        )
    hrl_collision = [0.0, 0.0, 0.0]
    for step in hrl["step_log"]:
        for robot, wait in enumerate(step["collision_wait_sec"]):
            hrl_collision[robot] += float(wait)
    hrl_metrics = execution_metrics.paper_schedule_metrics(
        hrl_work,
        hrl_completion,
        float(hrl["makespan_sec"]),
        collision_waits_sec=hrl_collision,
        collision_wait_status="measured_hrl_xy_block",
    )
    assert hrl_metrics["robot_work_sec"] == pytest.approx(
        [1031.0441189108512, 1675.0796996266738, 1504.5854866247735]
    )
    assert hrl_metrics["robot_process_wait_sec"] == pytest.approx(
        [1853.7314931442422, 2187.6205882312415, 2391.798701066278]
    )
    assert hrl_metrics["robot_completion_wait_sec"] == pytest.approx(
        [1011.6085756359585, 33.68389983313682, 0.0]
    )
    assert hrl_metrics["paper_wait_ratio_pct"] == pytest.approx(63.97763411469005)
    assert hrl_metrics["paper_process_wait_share_pct"] == pytest.approx(
        86.02259267845133
    )
    assert hrl_metrics["collision_wait_share_of_total_wait_pct"] == pytest.approx(
        12.754928646470859
    )


@dataclass
class _LightweightBeadEvent:
    layer_idx: int
    layer_z: float
    robot: int
    cluster_id: int
    btype: str
    start_pt: object
    end_pt: object
    t_start_sec: float
    t_end_sec: float
    shell_id: str | None = None

    @property
    def task_key(self) -> tuple[int, str | None]:
        return self.cluster_id, self.shell_id


def _ensure_lightweight_rapid_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    if "verify.rapid_timeline" in sys.modules:
        return
    fake_visualizer = types.ModuleType("pipeline.S3_path_planning.S34_vis_3d")
    fake_visualizer.BeadEvent = _LightweightBeadEvent
    monkeypatch.setitem(
        sys.modules, "pipeline.S3_path_planning.S34_vis_3d", fake_visualizer
    )
    importlib.import_module("verify.rapid_timeline")


def test_current_l1_rapid_execution_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cp_mod = EXPERIMENT / "cpsat" / "L1" / "rapid"
    hrl_mod = EXPERIMENT / "hrl_smoke" / "scenarios" / "L1" / "rapid"
    if not cp_mod.is_dir() or not hrl_mod.is_dir():
        pytest.skip("current CP-SAT and HRL L1 RAPID artifacts are not available")

    _ensure_lightweight_rapid_timeline(monkeypatch)
    cp = execution_metrics.rapid_execution_metrics(cp_mod)
    hrl = execution_metrics.rapid_execution_metrics(hrl_mod)

    assert cp["rapid_total_end_sec"] == pytest.approx(1593.502300600802)
    assert cp["robot_active_sec"] == pytest.approx(
        [1097.5876509953223, 1542.1745776707062, 1593.502300600802]
    )
    assert cp["rapid_active_utilization_pct"] == pytest.approx(88.5526287530461)
    assert cp["rapid_weld_utilization_pct"] == pytest.approx(70.84569718588568)
    assert cp["rapid_concurrent_2plus_pct"] == pytest.approx(96.77893637739062)
    assert cp["rapid_concurrent_3_pct"] == pytest.approx(68.87894988174766)

    assert hrl["rapid_total_end_sec"] == pytest.approx(3896.38420062477)
    assert hrl["robot_active_sec"] == pytest.approx(
        [1031.0441189108515, 1675.0796996266715, 1504.5854866247705]
    )
    assert hrl["rapid_active_utilization_pct"] == pytest.approx(36.02236576573671)
    assert hrl["rapid_weld_utilization_pct"] == pytest.approx(28.973729396416953)
    assert hrl["rapid_concurrent_2plus_pct"] == pytest.approx(13.277556108386369)
    assert hrl["rapid_concurrent_3_pct"] == pytest.approx(1.5393951321157364)
