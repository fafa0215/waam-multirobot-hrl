# 논문식 포인트 HRL의 자연 경로·50mm 큐·대기·XY 충돌 동작을 검증하는 테스트
from __future__ import annotations

import numpy as np
import pytest

from config import DEFAULT_CONFIG
from pipeline.S3_path_planning.S31_types import (
    PARK_REACH_MM,
    ROBOT_BASES,
)
from pipeline.S3_path_planning.S36_legacy_xy_collision import (
    legacy_xy_pose_conflict,
)
from pipeline.S3_path_planning.S37_task_granularity import TaskGroup
from pipeline.S3_path_planning.S38_hrl_pointwise import (
    MAX_NO_PROGRESS_STEPS,
    PointEdge,
    PointSchedulingEnv,
    RobotPointQueue,
    build_natural_paths,
    build_robot_point_queues,
    sample_natural_path,
)


def _bead(
    index: int,
    bead_type: str,
    x: float,
    y: float,
    dx: float,
    dy: float,
    length: float,
) -> dict:
    return {
        "index": index,
        "type": bead_type,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "kappa": 0.0,
        "length": length,
        "width": 8.09,
    }


def _edge(
    start,
    end,
    *,
    duration: float = 1.0,
    motion_kind: str = "WELD",
) -> PointEdge:
    start3 = tuple(float(value) for value in start)
    end3 = tuple(float(value) for value in end)
    return PointEdge(
        start=start3,
        end=end3,
        geometry=(start3, end3),
        motion_kind=motion_kind,
        distance_mm=float(np.linalg.norm(np.asarray(end3) - np.asarray(start3))),
        duration_sec=duration,
        path_id=0,
        bead_indices=(1,),
        bead_type="i_f",
    )


def _queue(
    robot: int,
    edges=(),
    *,
    park=None,
    work_plane_z=None,
) -> RobotPointQueue:
    base = ROBOT_BASES[robot]
    if park is None:
        park = (float(base[0]), float(base[1]), 1400.0)
    return RobotPointQueue(
        robot=robot,
        path_ids=(0,) if edges else (),
        park_xyz=park,
        edges=tuple(edges),
        work_plane_z=work_plane_z,
    )


def test_natural_path_preserves_direction_and_covers_each_bead_once():
    beads = [
        _bead(1, "wall", 20.0, 0.0, 1.0, 0.0, 40.0),
        _bead(2, "wall", 60.0, 0.0, 1.0, 0.0, 40.0),
        _bead(3, "i_f", 15.0, 20.0, 0.6, 0.8, 50.0),
    ]
    paths = build_natural_paths(
        [TaskGroup(0, (1, 2, 3))], beads, layer_z=30.0)

    assert [path.bead_indices for path in paths] == [(1, 2), (3,)]
    assert [index for path in paths for index in path.bead_indices] == [1, 2, 3]
    assert paths[0].start == pytest.approx((0.0, 0.0, 30.0))
    assert paths[0].end == pytest.approx((80.0, 0.0, 30.0))
    assert paths[1].start == pytest.approx((0.0, 0.0, 30.0))
    assert paths[1].end == pytest.approx((30.0, 40.0, 30.0))


def test_50mm_edges_keep_internal_csv_chord_vertex():
    beads = [
        _bead(1, "wall", 20.0, 0.0, 1.0, 0.0, 40.0),
        _bead(2, "wall", 60.0, 0.0, 1.0, 0.0, 40.0),
    ]
    path = build_natural_paths(
        [TaskGroup(0, (1, 2))], beads, layer_z=30.0)[0]
    edges = sample_natural_path(path)

    assert [edge.distance_mm for edge in edges] == pytest.approx([50.0, 30.0])
    assert [point[0] for point in edges[0].geometry] == pytest.approx(
        [0.0, 40.0, 50.0])
    assert [point[0] for point in edges[1].geometry] == pytest.approx([50.0, 80.0])
    assert edges[0].bead_indices == (1, 2)
    assert edges[1].bead_indices == (2,)
    assert max(edge.distance_mm for edge in edges) <= 50.0
    assert sum(edge.duration_sec for edge in edges) == pytest.approx(80.0 / 2.0)


def test_exact_multiple_of_50_has_no_duplicate_terminal_edge():
    bead = _bead(1, "i_f", 50.0, 0.0, 1.0, 0.0, 100.0)
    path = build_natural_paths(
        [TaskGroup(0, (1,))], [bead], layer_z=3.9)[0]
    edges = sample_natural_path(path)

    assert [edge.distance_mm for edge in edges] == pytest.approx([50.0, 50.0])
    assert edges[0].end == edges[1].start
    assert edges[-1].end == pytest.approx((100.0, 0.0, 3.9))


def test_robot_queue_uses_one_park_round_trip_and_connects_two_paths_overhead():
    beads = [
        _bead(1, "wall", 20.0, 0.0, 1.0, 0.0, 40.0),
        _bead(2, "wall", 60.0, 0.0, 1.0, 0.0, 40.0),
        _bead(3, "i_f", 200.0, 100.0, 1.0, 0.0, 60.0),
    ]
    paths = build_natural_paths(
        [TaskGroup(0, (1, 2, 3))], beads, layer_z=30.0)
    queues = build_robot_point_queues(paths, [0, 0], beads)
    queue = queues[0]

    assert len(queues) == 3
    assert queue.path_ids == tuple(path.path_id for path in paths)
    assert queues[1].edges == queues[2].edges == ()
    assert [item.work_plane_z for item in queues] == pytest.approx([30.0] * 3)
    assert np.linalg.norm(
        np.asarray(queue.park_xyz[:2]) - ROBOT_BASES[0]
    ) == pytest.approx(PARK_REACH_MM)
    assert queue.edges[0].start == queue.park_xyz
    assert queue.edges[-1].end == queue.park_xyz
    assert sum(edge.start == queue.park_xyz for edge in queue.edges) == 1
    assert sum(edge.end == queue.park_xyz for edge in queue.edges) == 1
    assert all(edge.distance_mm <= 50.0 + 1e-9 for edge in queue.edges)
    assert all(
        first.end == second.start
        for first, second in zip(queue.edges, queue.edges[1:])
    )

    weld_edges = [edge for edge in queue.edges if edge.motion_kind == "WELD"]
    transit_edges = [edge for edge in queue.edges if edge.motion_kind == "TRANSIT"]
    assert sum(edge.duration_sec for edge in weld_edges) == pytest.approx(
        80.0 / 2.0 + 60.0 / 6.67)
    assert transit_edges
    transition_vertices = {
        tuple(point)
        for edge in transit_edges
        for point in edge.geometry
    }
    assert (paths[0].end[0], paths[0].end[1], 40.0) in transition_vertices
    assert (paths[1].start[0], paths[1].start[1], 40.0) in transition_vertices
    assert all(
        edge.duration_sec == pytest.approx(
            edge.distance_mm / DEFAULT_CONFIG.process.flyby_speed)
        for edge in transit_edges
    )


def test_wait_then_advance_keeps_original_18d_and_8_action_contract():
    base0 = ROBOT_BASES[0]
    edge0 = _edge(
        (float(base0[0]), float(base0[1]), 1400.0),
        (float(base0[0] - 10.0), float(base0[1]), 1400.0),
        duration=2.0,
    )
    env = PointSchedulingEnv((_queue(0, (edge0,)), _queue(1), _queue(2)))
    observation, _ = env.reset(seed=42)

    assert observation.shape == (18,)
    assert env.observation_space.shape == (18,)
    assert env.action_space.n == 8

    _obs, _reward, terminated, truncated, info = env.step(0)
    assert not terminated and not truncated
    assert info["indices"] == [0, 0, 0]
    assert env.wait_steps.tolist() == [1, 0, 0]
    assert info["makespan_sec"] == pytest.approx(1.0)

    _obs, _reward, terminated, truncated, info = env.step(0b001)
    assert terminated and not truncated
    assert info["indices"] == [1, 0, 0]
    assert info["effective_move"] == [True, False, False]
    assert info["makespan_sec"] == pytest.approx(3.0)
    assert info["records"][0]["start_sec"] == pytest.approx(1.0)
    assert info["records"][0]["end_sec"] == pytest.approx(3.0)
    assert env.step_log[0]["commanded_wait_sec"] == pytest.approx([1.0, 0.0, 0.0])
    assert env.step_log[1]["edge_duration_sec"] == pytest.approx([2.0, 0.0, 0.0])


def test_simultaneous_edges_advance_by_longest_physical_duration():
    edges = []
    for robot, duration in ((0, 2.0), (1, 5.0)):
        base = ROBOT_BASES[robot]
        edges.append(_edge(
            (float(base[0]), float(base[1]), 1400.0),
            (float(base[0] * 0.99), float(base[1] * 0.99), 1400.0),
            duration=duration,
        ))
    env = PointSchedulingEnv(
        (_queue(0, (edges[0],)), _queue(1, (edges[1],)), _queue(2)))
    env.reset(seed=42)

    _obs, _reward, terminated, _truncated, info = env.step(0b011)

    assert terminated
    assert info["makespan_sec"] == pytest.approx(5.0)
    by_robot = {record["robot"]: record for record in info["records"]}
    assert by_robot[0]["end_sec"] == pytest.approx(2.0)
    assert by_robot[1]["end_sec"] == pytest.approx(5.0)
    assert by_robot[0]["step_end_sec"] == pytest.approx(5.0)
    assert by_robot[1]["step_end_sec"] == pytest.approx(5.0)
    assert env.step_log[-1]["barrier_wait_sec"] == pytest.approx([3.0, 0.0, 0.0])
    assert env.step_log[-1]["collision_wait_sec"] == pytest.approx([0.0, 0.0, 0.0])


def test_legacy_xy_pose_conflict_blocks_later_robot_at_same_candidate():
    center = (0.0, 0.0, 30.0)
    queues = []
    for robot in (0, 1):
        base = ROBOT_BASES[robot]
        queues.append(_queue(robot, (_edge(
            (float(base[0]), float(base[1]), 1400.0),
            center,
        ),)))
    env = PointSchedulingEnv((queues[0], queues[1], _queue(2)))
    env.reset(seed=42)

    _obs, _reward, terminated, truncated, info = env.step(0b011)

    assert not terminated and not truncated
    assert info["indices"] == [1, 0, 0]
    assert info["effective_move"] == [True, False, False]
    assert info["blocked_by_collision"] == [False, True, False]
    assert info["collision_count"] == 1
    assert env.collision_steps.tolist() == [0, 1, 0]
    assert env.wait_steps.tolist() == [0, 1, 0]
    assert env.step_log[-1]["collision_wait_sec"] == pytest.approx([0.0, 1.0, 0.0])


def test_equal_size_subset_prefers_robot_with_longer_existing_wait():
    center = (0.0, 0.0, 30.0)
    queues = []
    for robot in (0, 1):
        base = ROBOT_BASES[robot]
        queues.append(_queue(robot, (_edge(
            (float(base[0]), float(base[1]), 1400.0), center,
        ),)))
    env = PointSchedulingEnv((queues[0], queues[1], _queue(2)))
    env.reset(seed=42)
    env.wait_steps[1] = 3

    _obs, _reward, _terminated, _truncated, info = env.step(0b011)

    assert info["selected_subset"] == [1]
    assert info["effective_move"] == [False, True, False]
    assert info["blocked_by_collision"] == [True, False, False]


def test_reconstructed_smoke_deadlock_ignores_high_z_transit_false_collision():
    # hrl_smoke L1 교착 직전의 실제 index [29, 50, 16] 자세를 재구성한 값이다.
    current = [
        (-209.6382568691862, -155.5208289011058, 281.6413564377856),
        (-451.89313039138415, -57.90191701097234, 3.9),
        (-390.1806911829286, -209.08835897156513, 708.6446917417176),
    ]
    candidate = [
        (-241.0050979961377, -160.8996903909259, 243.07726528046783),
        (-442.7975677277574, -107.01201315605226, 3.9),
        (-392.9993799299891, -184.0888342774602, 665.4349849755748),
    ]
    assert legacy_xy_pose_conflict(
        ROBOT_BASES[0], candidate[0][:2], ROBOT_BASES[1], candidate[1][:2])

    queues = tuple(_queue(
        robot,
        (_edge(
            current[robot],
            candidate[robot],
            motion_kind="WELD" if robot == 1 else "TRANSIT",
        ),),
        park=current[robot],
        work_plane_z=3.9,
    ) for robot in range(3))
    env = PointSchedulingEnv(queues)
    env.reset(seed=42)

    _obs, _reward, terminated, truncated, info = env.step(0b111)

    assert terminated and not truncated
    assert info["effective_move"] == [True, True, True]
    assert info["blocked_by_collision"] == [False, False, False]
    assert info["collision_count"] == 0
    assert info["termination_reason"] == "complete"


def test_maximum_feasible_subset_can_escape_sequential_greedy_deadlock():
    current_xy = [
        (-433.9928745243723, -450.53083061802556),
        (727.8384943408246, 359.8286946627827),
        (-745.0610427861632, 770.5872547066119),
    ]
    candidate_xy = [
        (-786.4061339329782, -376.06098935793443),
        (667.718642455364, -671.3796537763152),
        (567.5451267525248, -568.5161642058188),
    ]
    queues = tuple(_queue(
        robot,
        (_edge((*current_xy[robot], 3.9), (*candidate_xy[robot], 3.9)),),
        park=(*current_xy[robot], 3.9),
        work_plane_z=3.9,
    ) for robot in range(3))
    env = PointSchedulingEnv(queues)
    env.reset(seed=42)

    _obs, _reward, terminated, truncated, info = env.step(0b111)

    assert not terminated and not truncated
    assert info["selected_subset"] == [0, 2]
    assert info["indices"] == [1, 0, 1]
    assert info["effective_move"] == [True, False, True]
    assert info["blocked_by_collision"] == [False, True, False]


def test_deadlock_reason_forces_noncontinuable_state_without_done_corruption():
    work_z = 30.0
    base0 = (float(ROBOT_BASES[0, 0]), float(ROBOT_BASES[0, 1]), work_z)
    center = (0.0, 0.0, work_z)
    env = PointSchedulingEnv((
        _queue(0, (_edge(base0, center),), park=base0, work_plane_z=work_z),
        _queue(1, park=center, work_plane_z=work_z),
        _queue(2),
    ))
    env.reset(seed=42)

    for _ in range(MAX_NO_PROGRESS_STEPS):
        _obs, _reward, terminated, truncated, info = env.step(0b001)

    assert not terminated and truncated
    assert info["termination_reason"] == "deadlock"
    assert not info["continuation_feasible"]
    assert env.step_log[-1]["termination_reason"] == "deadlock"
    assert not env.step_log[-1]["continuation_feasible"]
    assert not env.done_robot[0]


def test_feasible_horizon_state_is_explicitly_continuable():
    base0 = ROBOT_BASES[0]
    edge = _edge(
        (float(base0[0]), float(base0[1]), 30.0),
        (float(base0[0] - 10.0), float(base0[1]), 30.0),
    )
    env = PointSchedulingEnv((
        _queue(0, (edge,), park=edge.start, work_plane_z=30.0),
        _queue(1),
        _queue(2),
    ))
    env.horizon = 1
    env.reset(seed=42)

    _obs, _reward, terminated, truncated, info = env.step(0)

    assert not terminated and truncated
    assert info["termination_reason"] == "horizon"
    assert info["continuation_feasible"]


def test_default_horizon_can_finish_every_edge_even_without_parallel_motion():
    queues = tuple(_queue(
        robot,
        tuple(_edge(
            (float(ROBOT_BASES[robot, 0] - index), float(ROBOT_BASES[robot, 1]), 30.0),
            (float(ROBOT_BASES[robot, 0] - index - 1), float(ROBOT_BASES[robot, 1]), 30.0),
        ) for index in range(robot + 1)),
        work_plane_z=30.0,
    ) for robot in range(3))
    environment = PointSchedulingEnv(queues)

    assert environment.horizon == sum(len(queue.edges) for queue in queues) + 200


def test_horizon_state_without_any_safe_advance_is_not_continuable():
    work_z = 30.0
    base0 = (float(ROBOT_BASES[0, 0]), float(ROBOT_BASES[0, 1]), work_z)
    center = (0.0, 0.0, work_z)
    env = PointSchedulingEnv((
        _queue(0, (_edge(base0, center),), park=base0, work_plane_z=work_z),
        _queue(1, park=center, work_plane_z=work_z),
        _queue(2),
    ))
    env.horizon = 1
    env.reset(seed=42)

    _obs, _reward, terminated, truncated, info = env.step(0b001)

    assert not terminated and truncated
    assert info["termination_reason"] == "horizon"
    assert not info["continuation_feasible"]
