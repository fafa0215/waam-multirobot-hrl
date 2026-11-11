# 원본식 XY 충돌 백엔드와 TASK 스케줄 연결을 검증하는 테스트
import numpy as np

from pipeline.S3_path_planning.S36_hrl_layerwise import (
    HrlTask,
    TaskSchedulingEnv,
)
from pipeline.S3_path_planning.S36_legacy_xy_collision import (
    aabb_overlap,
    compute_legacy_xy_task_conflicts,
    legacy_xy_pose_conflict,
    line_box_intersect,
    line_segments_intersect,
)


def _task(task_id: int, bead_index: int) -> HrlTask:
    return HrlTask(
        layer_idx=0,
        task_type="infill",
        task_id=task_id,
        cluster_id=task_id,
        centroid=np.asarray([float(task_id), 0.0]),
        bbox_min=np.asarray([float(task_id), -1.0]),
        bbox_max=np.asarray([float(task_id), 1.0]),
        bead_indices=(bead_index,),
        duration_by_robot=(10.0, 10.0, 10.0),
        eligible_robots=(True, True, True),
    )


def test_legacy_xy_primitives_match_original_cases():
    assert aabb_overlap(0, 0, 10, 10, 4, 0, 10, 10)
    assert not aabb_overlap(0, 0, 10, 10, 20, 0, 10, 10)
    assert line_segments_intersect((-10, 0), (10, 0), (0, -10), (0, 10))
    assert not line_segments_intersect(
        (-10, 0), (10, 0), (-10, 200), (10, 200), margin=100)
    assert line_box_intersect((-20, 0), (20, 0), 0, 0, 10, 10, 0, margin=0)
    assert not line_box_intersect(
        (-20, 30), (20, 30), 0, 0, 10, 10, 0, margin=0)


def test_legacy_xy_pose_detects_crossing_arms():
    assert legacy_xy_pose_conflict(
        np.asarray([-500.0, 0.0]),
        np.asarray([500.0, 0.0]),
        np.asarray([0.0, -500.0]),
        np.asarray([0.0, 500.0]),
    )


def test_task_conflict_table_uses_current_direction_columns():
    tasks = [_task(0, 1), _task(1, 2)]
    beads = [{
        "index": 1, "type": "i_f", "x": 0.0, "y": 0.0,
        "dx": 1.0, "dy": 0.0, "length": 100.0, "width": 8.0,
    }, {
        "index": 2, "type": "i_f", "x": 0.0, "y": 0.0,
        "dx": 0.0, "dy": 1.0, "length": 100.0, "width": 8.0,
    }]
    conflicts = compute_legacy_xy_task_conflicts(tasks, beads)
    assert conflicts
    assert all(len(key) == 6 for key in conflicts)


def test_external_conflict_table_blocks_simultaneous_task_start():
    tasks = [_task(0, 1), _task(1, 2)]
    conflict = {("infill", 0, 0, "infill", 1, 1): True}
    env = TaskSchedulingEnv(tasks, [0, 1], conflict)
    env.reset(seed=42)
    _obs, _reward, _terminated, _truncated, info = env.step(0b011)
    assert info["started"] == 1
    assert info["blocked"] == 1
