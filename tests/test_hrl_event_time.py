# 독립 시간 HRL의 비동기 진행·대기·이어하기·곡선 길이·출력을 검증하는 테스트
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import numpy as np
import pytest

from pipeline.S3_path_planning.S38_hrl_pointwise import NaturalPath, PathSegment, PointEdge, RobotPointQueue
from pipeline.S3_path_planning.S39_hrl_paper_training import PointLayerTrainingResult
from pipeline.S3_path_planning.S40_hrl_pointwise_rapid import generate_pointwise_rapid_per_robot
from pipeline.S3_path_planning.S42_hrl_event_time import (
    EventPointSchedulingEnv, MetaEventPointSchedulingEnv, build_event_queues,
    csv_natural_paths, curve_flags, sample_adaptive_path,
)
from tests.s36_hrl.test_hrl_paper_training_rapid import _three_sector_paths, _beads


def _edge(x, distance, duration):
    start, end = (x, 0.0, 100.0), (x + distance, 0.0, 100.0)
    return PointEdge(start, end, (start, end), "TRANSIT", distance, duration, 0)


def _queues():
    return tuple(RobotPointQueue(robot, (0,), (0.0, 0.0, 100.0),
                                (_edge(0, 10, duration), _edge(10, 10, duration)))
                 for robot, duration in enumerate((2.0, 5.0, 10.0)))


def test_short_robot_restarts_while_long_robot_is_still_moving():
    environment = EventPointSchedulingEnv(_queues())
    observation, _ = environment.reset(seed=42)
    assert environment.observation_space.contains(observation)
    _, _, done, _, _ = environment.step(7)
    assert not done
    assert environment.now_sec == 2.0
    assert environment.indices == {0: 1, 1: 0, 2: 0}
    assert environment._position(1) == pytest.approx((4.0, 0.0, 100.0))
    _, _, _, _, info = environment.step(1)
    assert environment.now_sec == 4.0
    assert info["started"] == [0]
    assert environment.indices == {0: 2, 1: 0, 2: 0}
    assert environment.records[-1]["start_sec"] == 2.0
    assert environment._position(2)[0] == pytest.approx(4.0)


def test_wait_does_not_interrupt_busy_robot_and_reconsiders_at_completion():
    environment = EventPointSchedulingEnv(_queues())
    environment.step(7)
    _, _, _, _, info = environment.step(0)
    assert info["waiting"] == [True, False, False]
    assert environment.now_sec == 5.0
    assert environment.indices == {0: 1, 1: 1, 2: 0}
    assert environment.wait_steps.tolist() == [1, 0, 0]


def test_all_idle_wait_advances_one_second_then_deadlocks_after_limit():
    environment = EventPointSchedulingEnv(_queues())
    for _ in range(50):
        _, _, done, truncated, info = environment.step(0)
    assert not done and truncated
    assert environment.now_sec == 50.0
    assert info["termination_reason"] == "deadlock"
    assert not environment.records


def test_simultaneous_completions_and_progress_reward_are_counted_once():
    queues = tuple(RobotPointQueue(robot, (0,), (0.0, 0.0, 100.0), (_edge(0, 10, 2),))
                   for robot in range(3))
    environment = EventPointSchedulingEnv(queues)
    _, reward, done, truncated, info = environment.step(7)
    assert done and not truncated
    assert info["completed"] == [0, 1, 2]
    assert reward == pytest.approx(-0.01 + 3 + 30 + 100)
    assert len(environment.records) == 3


def test_horizon_continuation_preserves_busy_robot_remaining_time():
    environment = MetaEventPointSchedulingEnv(_queues(), meta_reset_probability=0.0)
    environment.horizon = 1
    _, _, _, truncated, _ = environment.step(7)
    assert truncated
    environment.reset(seed=1)
    assert environment.now_sec == 2.0
    assert environment.active[1]["end_sec"] == 5.0
    assert environment._position(1)[0] == pytest.approx(4.0)
    environment.step(0)
    assert environment.now_sec == 5.0


def test_collision_rejected_move_gets_penalty_and_is_not_logged(monkeypatch):
    environment = EventPointSchedulingEnv(_queues())
    monkeypatch.setattr(environment, "_conflicts", lambda motions: bool(motions))
    _, reward, _, _, info = environment.step(7)
    assert info["blocked_by_collision"] == [True, True, True]
    assert reward == pytest.approx(-0.01 - 6.0 - 0.15)
    assert not environment.records


def test_smooth_curve_spacing_preserves_length_vertices_and_assignment_atom():
    points = [(0.0, 0.0, 3.9)]
    for degrees in (0, 5, 10):
        angle = np.radians(degrees)
        points.append(tuple(np.asarray(points[-1]) + (40 * np.cos(angle), 40 * np.sin(angle), 0)))
    segments = tuple(PathSegment(a, b, index, "wall")
                     for index, (a, b) in enumerate(zip(points, points[1:])))
    path = NaturalPath(0, 0, (0, 1, 2), "wall", segments)
    assert curve_flags(path) == [True, True, True]
    edges = sample_adaptive_path(path)
    assert max(edge.distance_mm for edge in edges) <= 10.0 + 1e-8
    assert sum(edge.distance_mm for edge in edges) == pytest.approx(120.0)
    geometry_points = [point for edge in edges for point in edge.geometry]
    for point in points:
        assert any(np.linalg.norm(np.asarray(value) - point) < 1e-7 for value in geometry_points)


def test_csv_atoms_do_not_merge_independent_infill_rows():
    paths = _three_sector_paths()
    beads = _beads(paths)
    result = csv_natural_paths(beads, layer_z=3.9)
    assert len(result) == 3
    assert [path.bead_indices for path in result] == [(1,), (2,), (3,)]


def test_event_rapid_roundtrip_preserves_length_and_independent_robot_finish(tmp_path):
    paths = _three_sector_paths()
    beads = _beads(paths)
    queues = build_event_queues(paths, [0, 1, 2], beads, straight_mm=10.0)
    environment = EventPointSchedulingEnv(queues)
    for _ in range(1000):
        _, _, done, truncated, _ = environment.step(7)
        if done or truncated:
            break
    assert done and not truncated
    result = PointLayerTrainingResult(
        0, 3.9, paths, [0, 1, 2], queues, environment.records,
        environment.step_log, environment.now_sec, True, 0.0, 1,
        evaluation_stats={"time_model": "independent_event_time"},
    )
    summary = generate_pointwise_rapid_per_robot([result], tmp_path)
    assert sum(robot["weld_length_mm"] for robot in summary["robots"].values()) == pytest.approx(
        sum(path.length_mm for path in paths))
    assert max(robot["derived_end_sec"] for robot in summary["robots"].values()) == pytest.approx(environment.now_sec)
    assert all(robot["arc_restarts"] == 0 for robot in summary["robots"].values())
    from pipeline.S3_path_planning.S41_execution_metrics import rapid_execution_metrics
    metrics = rapid_execution_metrics(tmp_path)
    assert metrics["rapid_total_end_sec"] == pytest.approx(environment.now_sec, abs=0.02)


def test_wait_inside_path_emits_arc_end_and_restart_without_extra_weld(tmp_path):
    paths = _three_sector_paths()
    beads = _beads(paths)
    queues = build_event_queues(paths, [0, 1, 2], beads, straight_mm=10.0)
    environment = EventPointSchedulingEnv(queues)
    stopped = False
    for _ in range(1000):
        first = environment.indices[0]
        edge = queues[0].edges[first] if first < len(queues[0].edges) else None
        if not stopped and 0 not in environment.active and edge and edge.motion_kind == "WELD" and edge.cumulative_start_mm > 0:
            action, stopped = 0, True
        else:
            action = 7
        _, _, done, truncated, _ = environment.step(action)
        if done or truncated:
            break
    assert done and stopped
    result = PointLayerTrainingResult(0, 3.9, paths, [0, 1, 2], queues, environment.records,
                                     environment.step_log, environment.now_sec, True, 0.0, 1,
                                     evaluation_stats={"time_model": "independent_event_time"})
    summary = generate_pointwise_rapid_per_robot([result], tmp_path)
    assert summary["robots"]["0"]["arc_restarts"] == 1
    assert summary["robots"]["0"]["weld_length_mm"] == pytest.approx(paths[0].length_mm)


def test_xy_check_uses_interpolated_busy_position_at_completion(monkeypatch):
    from dataclasses import replace
    import pipeline.S3_path_planning.S42_hrl_event_time as module
    queues = tuple(replace(queue, work_plane_z=100.0) for queue in _queues())
    checked = []
    monkeypatch.setattr(module, "legacy_xy_pose_conflict",
                        lambda b1, p1, b2, p2: checked.append((tuple(p1), tuple(p2))) or False)
    environment = EventPointSchedulingEnv(queues)
    environment.step(7)
    assert any(first == (10.0, 0.0) and second == (4.0, 0.0) for first, second in checked)


def test_frozen_v4_input_and_serial_output_preserve_all_deposition(tmp_path):
    from exp_tools.hrl_event_run import prepare_input
    root = Path(__file__).resolve().parents[3]
    beads, z, paths, manifest = prepare_input(Namespace(
        input=root / "docs/experiments/2026-08-08_poster-contracts/beads/V4", straight_mm=50.0,
    ), tmp_path)
    assert len(beads) == 280 and len(paths) == 138
    assert manifest["source_csv_sha256"]["beads_Z3.9.csv"] == "2b1787c06434fa05b36f049a162b8fe34c87c4525c94808d76ebbda06843916e"
    assert len(manifest["frame_scope"]) == 11
    assert sum(len(path["curve_bead_indices"]) for path in manifest["paths"]) == 144
    queues = build_event_queues(paths, [0] * 138, beads)
    environment = EventPointSchedulingEnv(queues)
    for _ in range(5000):
        _, _, done, truncated, _ = environment.step(7)
        if done or truncated:
            break
    assert done and not truncated
    result = PointLayerTrainingResult(0, z, paths, [0] * 138, queues, environment.records,
                                     environment.step_log, environment.now_sec, True, 0.0, 1,
                                     evaluation_stats={"time_model": "independent_event_time"})
    summary = generate_pointwise_rapid_per_robot([result], tmp_path / "rapid")
    assert summary["robots"]["0"]["weld_length_mm"] == pytest.approx(manifest["total_length_mm"], abs=1e-5)
    from pipeline.S3_path_planning.S41_execution_metrics import rapid_execution_metrics
    metrics = rapid_execution_metrics(tmp_path / "rapid")
    assert metrics["rapid_total_end_sec"] == pytest.approx(environment.now_sec, abs=0.05)
