# 원본 HRL의 XY 직사각형·선분 충돌식을 현재 TASK 경로에 적용하는 백엔드
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from pipeline.S3_path_planning.S31_types import (
    N_ROBOTS,
    PARK_REACH_MM,
    ROBOT_BASES,
)


ROBOT_BODY_LEN_MM = 500.0
ROBOT_BODY_WID_MM = 300.0
ARM_COLLISION_MARGIN_MM = 100.0
DEFAULT_SAMPLE_STEP_MM = 50.0


def aabb_overlap(cx1, cy1, w1, h1, cx2, cy2, w2, h2) -> bool:
    """원본 HRL의 axis-aligned body box 중첩 판정."""

    return ((abs(cx1 - cx2) * 2 < (w1 + w2))
            and (abs(cy1 - cy2) * 2 < (h1 + h2)))


def _point_to_segment_distance(point, start, end) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom < 1e-10:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def line_segments_intersect(
    p1,
    p2,
    p3,
    p4,
    margin: float = ARM_COLLISION_MARGIN_MM,
) -> bool:
    """원본 HRL의 arm-arm 선분 교차 및 근접 판정."""

    def cross(origin, first, second):
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    def on_segment(start, point, end):
        return (min(start[0], end[0]) <= point[0] <= max(start[0], end[0])
                and min(start[1], end[1]) <= point[1] <= max(start[1], end[1]))

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if (((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
            and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))):
        return True
    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True
    distance = min(
        _point_to_segment_distance(p1, p3, p4),
        _point_to_segment_distance(p2, p3, p4),
        _point_to_segment_distance(p3, p1, p2),
        _point_to_segment_distance(p4, p1, p2),
    )
    return distance < margin


def line_box_intersect(
    p1,
    p2,
    cx,
    cy,
    width,
    height,
    theta,
    margin: float = ARM_COLLISION_MARGIN_MM,
) -> bool:
    """원본 HRL의 arm-body 선분 대 회전 직사각형 판정."""

    width_ext = width + 2 * margin
    height_ext = height + 2 * margin
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)

    def to_local(point):
        dx = point[0] - cx
        dy = point[1] - cy
        return dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t

    local_p1 = to_local(p1)
    local_p2 = to_local(p2)
    half_width = width_ext / 2
    half_height = height_ext / 2
    if ((abs(local_p1[0]) <= half_width and abs(local_p1[1]) <= half_height)
            or (abs(local_p2[0]) <= half_width
                and abs(local_p2[1]) <= half_height)):
        return True

    dx = local_p2[0] - local_p1[0]
    dy = local_p2[1] - local_p1[1]
    t_min, t_max = 0.0, 1.0
    edges = [
        (-dx, local_p1[0] + half_width),
        (dx, half_width - local_p1[0]),
        (-dy, local_p1[1] + half_height),
        (dy, half_height - local_p1[1]),
    ]
    for p_value, q_value in edges:
        if abs(p_value) < 1e-10:
            if q_value < 0:
                return False
        else:
            ratio = q_value / p_value
            if p_value < 0:
                t_min = max(t_min, ratio)
            else:
                t_max = min(t_max, ratio)
            if t_min > t_max:
                return False
    return True


def _collider_aabb(base, tcp):
    dx, dy = float(tcp[0] - base[0]), float(tcp[1] - base[1])
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        theta, ux, uy = 0.0, 1.0, 0.0
    else:
        ux, uy = dx / distance, dy / distance
        theta = math.atan2(uy, ux)
    cx = float(tcp[0]) - ux * ROBOT_BODY_LEN_MM / 2
    cy = float(tcp[1]) - uy * ROBOT_BODY_LEN_MM / 2
    half_length = ROBOT_BODY_LEN_MM / 2
    half_width = ROBOT_BODY_WID_MM / 2
    hx = abs(half_length * math.cos(theta)) + abs(half_width * math.sin(theta))
    hy = abs(half_length * math.sin(theta)) + abs(half_width * math.cos(theta))
    return cx, cy, 2 * hx, 2 * hy


def _collider_pose(base, tcp):
    dx, dy = float(tcp[0] - base[0]), float(tcp[1] - base[1])
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        theta, ux, uy = 0.0, 1.0, 0.0
    else:
        ux, uy = dx / distance, dy / distance
        theta = math.atan2(uy, ux)
    return (
        float(tcp[0]) - ux * ROBOT_BODY_LEN_MM / 2,
        float(tcp[1]) - uy * ROBOT_BODY_LEN_MM / 2,
        ROBOT_BODY_LEN_MM,
        ROBOT_BODY_WID_MM,
        theta,
    )


def legacy_xy_pose_conflict(base1, tcp1, base2, tcp2) -> bool:
    """두 로봇 TCP 자세를 원본의 box-box·line-line·line-box로 검사한다."""

    aabb1 = _collider_aabb(base1, tcp1)
    aabb2 = _collider_aabb(base2, tcp2)
    if aabb_overlap(*aabb1, *aabb2):
        return True
    if line_segments_intersect(base1, tcp1, base2, tcp2):
        return True
    pose1 = _collider_pose(base1, tcp1)
    pose2 = _collider_pose(base2, tcp2)
    if line_box_intersect(base1, tcp1, *pose2):
        return True
    return line_box_intersect(base2, tcp2, *pose1)


def _bead_points(bead: dict, step_mm: float) -> list[np.ndarray]:
    direction = np.asarray([bead["dx"], bead["dy"]], dtype=float)
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-9 else np.asarray([1.0, 0.0])
    center = np.asarray([bead["x"], bead["y"]], dtype=float)
    length = abs(float(bead["length"]))
    start = center - direction * length / 2
    end = center + direction * length / 2
    n_steps = max(2, int(length / step_mm) + 1)
    return [start + (end - start) * (index / (n_steps - 1))
            for index in range(n_steps)]


def _task_work_points(task, bead_by_index: dict[int, dict], step_mm: float):
    points = []
    for bead_index in task.bead_indices:
        points.extend(_bead_points(bead_by_index[int(bead_index)], step_mm))
    return points


def _park_xy(robot: int, work_centroid) -> np.ndarray:
    base = np.asarray(ROBOT_BASES[robot], dtype=float)
    direction = np.asarray(work_centroid, dtype=float)[:2] - base
    distance = float(np.linalg.norm(direction))
    return base if distance < 1e-9 else base + PARK_REACH_MM * direction / distance


def _task_pose_points(work_points, robot: int, work_centroid):
    """현재 Park 왕복을 포함해 원본 XY collider가 검사할 TCP 표본을 만든다."""

    park = _park_xy(robot, work_centroid)
    if not work_points:
        return [park]
    points = []
    for target in (work_points[0], work_points[-1]):
        points.extend(park * (1 - ratio) + target * ratio
                      for ratio in (0.2, 0.4, 0.6, 0.8, 1.0))
    return points[:5] + list(work_points) + points[5:] + [park]


def compute_legacy_xy_task_conflicts(
    tasks: Sequence,
    beads: Sequence[dict],
    *,
    step_mm: float = DEFAULT_SAMPLE_STEP_MM,
) -> dict:
    """TASK 쌍과 로봇 쌍의 원본식 XY swept conflict 테이블을 만든다."""

    if step_mm <= 0:
        raise ValueError("step_mm must be positive")
    bead_by_index = {int(bead["index"]): bead for bead in beads}
    work_centroid = np.mean(
        np.asarray([[bead["x"], bead["y"]] for bead in beads], dtype=float), axis=0)
    work_points = [_task_work_points(task, bead_by_index, step_mm) for task in tasks]
    pose_points = {
        (task_index, robot): _task_pose_points(
            work_points[task_index], robot, work_centroid)
        for task_index in range(len(tasks))
        for robot in range(N_ROBOTS)
    }
    conflicts = {}
    for first_index, first in enumerate(tasks):
        for second_index in range(first_index + 1, len(tasks)):
            second = tasks[second_index]
            for first_robot in range(N_ROBOTS):
                first_base = ROBOT_BASES[first_robot]
                for second_robot in range(N_ROBOTS):
                    if first_robot == second_robot:
                        continue
                    second_base = ROBOT_BASES[second_robot]
                    hit = any(
                        legacy_xy_pose_conflict(
                            first_base, first_tcp, second_base, second_tcp)
                        for first_tcp in pose_points[(first_index, first_robot)]
                        for second_tcp in pose_points[(second_index, second_robot)]
                    )
                    if hit:
                        conflicts[(
                            first.task_type,
                            first_robot,
                            first.task_id,
                            second.task_type,
                            second_robot,
                            second.task_id,
                        )] = True
    return conflicts
