# V4 자연 경로를 50mm 판단점 큐와 논문식 포인트 스케줄링 환경으로 변환하는 모듈
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import DEFAULT_CONFIG, KitechConfig
from pipeline.S3_path_planning.S31_types import (
    N_ROBOTS,
    PARK_REACH_MM,
    PARK_Z_MM,
    ROBOT_BASES,
    classify_bead,
)
from pipeline.S3_path_planning.S33_rapid_writer import (
    DEFAULT_RAPID_CONFIG,
    RapidConfig,
)
from pipeline.S3_path_planning.S36_legacy_xy_collision import (
    DEFAULT_SAMPLE_STEP_MM,
    legacy_xy_pose_conflict,
)
from pipeline.S3_path_planning.S37_task_granularity import (
    TaskGroup,
    natural_path_atoms,
)


Point3 = Tuple[float, float, float]

TIME_COST = -0.01
PROGRESS_GAIN = 1.0
COLLISION_PENALTY = -2.0
WAIT_PENALTY = -0.05
DONE_BONUS = 10.0
ALL_DONE_BONUS = 100.0
TOTAL_WAIT_PENALTY = -0.01
MAX_NO_PROGRESS_STEPS = 50
DEADLOCK_PENALTY = -20.0
WORK_PLANE_TOL_MM = 1e-3


@dataclass(frozen=True)
class PathSegment:
    """원래 CSV 비드 한 행이 만든 방향성 3차원 선분."""

    start: Point3
    end: Point3
    bead_index: Optional[int]
    bead_type: Optional[str]

    @property
    def length_mm(self) -> float:
        return _distance(self.start, self.end)


@dataclass(frozen=True)
class NaturalPath:
    """S37 연결 규칙으로 묶은, 한 로봇이 끝까지 맡는 자연 경로."""

    path_id: int
    root_id: int
    bead_indices: Tuple[int, ...]
    bead_type: str
    segments: Tuple[PathSegment, ...]

    @property
    def start(self) -> Point3:
        return self.segments[0].start

    @property
    def end(self) -> Point3:
        return self.segments[-1].end

    @property
    def length_mm(self) -> float:
        return float(sum(segment.length_mm for segment in self.segments))


@dataclass(frozen=True)
class PointEdge:
    """한 HRL 행동으로 이동하는 최대 50mm 구간과 보존된 내부 꼭짓점."""

    start: Point3
    end: Point3
    geometry: Tuple[Point3, ...]
    motion_kind: str
    distance_mm: float
    duration_sec: float
    path_id: Optional[int]
    bead_indices: Tuple[int, ...] = ()
    bead_type: Optional[str] = None
    cumulative_start_mm: float = 0.0
    cumulative_end_mm: float = 0.0


@dataclass(frozen=True)
class RobotPointQueue:
    """한 로봇의 층당 한 번인 Park 출발·경로 연결·Park 복귀 판단 큐."""

    robot: int
    path_ids: Tuple[int, ...]
    park_xyz: Point3
    edges: Tuple[PointEdge, ...]
    work_plane_z: Optional[float] = None


def _point3(value: Sequence[float]) -> Point3:
    return float(value[0]), float(value[1]), float(value[2])


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(second, float) - np.asarray(first, float)))


def _bead_endpoints(bead: dict, layer_z: float) -> Tuple[Point3, Point3]:
    """S33 RAPID writer와 같은 center+direction+length 끝점 공식을 쓴다."""

    half = 0.5 * float(bead["length"])
    start = (
        float(bead["x"]) - float(bead["dx"]) * half,
        float(bead["y"]) - float(bead["dy"]) * half,
        float(layer_z),
    )
    end = (
        float(bead["x"]) + float(bead["dx"]) * half,
        float(bead["y"]) + float(bead["dy"]) * half,
        float(layer_z),
    )
    return start, end


def build_natural_paths(
    base_groups: Sequence[TaskGroup],
    beads: Sequence[dict],
    *,
    layer_z: float,
) -> List[NaturalPath]:
    """S37 자연 경로 원자를 방향과 CSV 순서를 바꾸지 않고 3D로 만든다."""

    bead_by_index = {int(bead["index"]): bead for bead in beads}
    paths: List[NaturalPath] = []
    for path_id, atom in enumerate(natural_path_atoms(base_groups, beads)):
        members = [bead_by_index[index] for index in atom.bead_indices]
        bead_types = {str(bead.get("type", "")) for bead in members}
        if len(bead_types) != 1:
            raise ValueError(f"natural path {path_id} mixes bead types")

        segments: List[PathSegment] = []
        previous_end: Optional[Point3] = None
        for bead in members:
            start, end = _bead_endpoints(bead, layer_z)
            if previous_end is not None:
                if _distance(previous_end, start) > 1e-3:
                    raise ValueError(f"natural path {path_id} is not continuous")
                # S33은 허용오차 안의 다음 시작점을 생략하므로 같은 폴리라인을 만든다.
                start = previous_end
            segment = PathSegment(
                start=start,
                end=end,
                bead_index=int(bead["index"]),
                bead_type=str(bead["type"]),
            )
            if segment.length_mm <= 1e-9:
                raise ValueError(f"bead {bead['index']} has a zero-length path")
            segments.append(segment)
            previous_end = end

        paths.append(NaturalPath(
            path_id=path_id,
            root_id=int(atom.root_id),
            bead_indices=tuple(int(index) for index in atom.bead_indices),
            bead_type=next(iter(bead_types)),
            segments=tuple(segments),
        ))
    return paths


def _point_at_distance(
    segments: Sequence[PathSegment],
    cumulative_ends: Sequence[float],
    distance_mm: float,
) -> Point3:
    if distance_mm >= cumulative_ends[-1] - 1e-9:
        return segments[-1].end
    previous = 0.0
    for segment, end_distance in zip(segments, cumulative_ends):
        if distance_mm <= end_distance + 1e-9:
            span = end_distance - previous
            ratio = 0.0 if span <= 1e-12 else (distance_mm - previous) / span
            start = np.asarray(segment.start, float)
            end = np.asarray(segment.end, float)
            return _point3(start + np.clip(ratio, 0.0, 1.0) * (end - start))
        previous = end_distance
    return segments[-1].end


def _append_distinct(points: List[Point3], point: Point3) -> None:
    if not points or _distance(points[-1], point) > 1e-9:
        points.append(point)


def _sample_segments(
    segments: Sequence[PathSegment],
    *,
    motion_kind: str,
    speed_mm_s: float,
    max_edge_mm: float,
    path_id: Optional[int],
    bead_type: Optional[str],
) -> List[PointEdge]:
    if max_edge_mm <= 0:
        raise ValueError("max_edge_mm must be positive")
    if speed_mm_s <= 0:
        raise ValueError("motion speed must be positive")
    if not segments:
        return []

    cumulative_ends: List[float] = []
    total = 0.0
    for segment in segments:
        total += segment.length_mm
        cumulative_ends.append(total)
    if total <= 1e-9:
        return []

    edges: List[PointEdge] = []
    interval_start = 0.0
    while interval_start < total - 1e-9:
        interval_end = min(interval_start + max_edge_mm, total)
        geometry: List[Point3] = []
        _append_distinct(
            geometry,
            _point_at_distance(segments, cumulative_ends, interval_start),
        )
        for segment, vertex_distance in zip(segments[:-1], cumulative_ends[:-1]):
            if interval_start + 1e-9 < vertex_distance < interval_end - 1e-9:
                _append_distinct(geometry, segment.end)
        _append_distinct(
            geometry,
            _point_at_distance(segments, cumulative_ends, interval_end),
        )

        source_indices: List[int] = []
        segment_start = 0.0
        for segment, segment_end in zip(segments, cumulative_ends):
            overlaps = (
                segment_end > interval_start + 1e-9
                and segment_start < interval_end - 1e-9
            )
            if overlaps and segment.bead_index is not None:
                if segment.bead_index not in source_indices:
                    source_indices.append(segment.bead_index)
            segment_start = segment_end

        distance_mm = interval_end - interval_start
        edges.append(PointEdge(
            start=geometry[0],
            end=geometry[-1],
            geometry=tuple(geometry),
            motion_kind=motion_kind,
            distance_mm=distance_mm,
            duration_sec=distance_mm / speed_mm_s,
            path_id=path_id,
            bead_indices=tuple(source_indices),
            bead_type=bead_type,
            cumulative_start_mm=interval_start,
            cumulative_end_mm=interval_end,
        ))
        interval_start = interval_end
    return edges


def sample_natural_path(
    path: NaturalPath,
    *,
    config: KitechConfig = DEFAULT_CONFIG,
    max_edge_mm: float = DEFAULT_SAMPLE_STEP_MM,
) -> List[PointEdge]:
    """자연 경로를 실제 적층 속도의 최대 50mm WELD 판단 구간으로 나눈다."""

    bead_class = classify_bead(path.bead_type)
    speed = (
        float(config.weaving.weld_speed_weave)
        if bead_class == "WALL"
        else float(config.weaving.weld_speed_normal)
    )
    return _sample_segments(
        path.segments,
        motion_kind="WELD",
        speed_mm_s=speed,
        max_edge_mm=max_edge_mm,
        path_id=path.path_id,
        bead_type=path.bead_type,
    )


def _polyline_segments(points: Sequence[Point3]) -> List[PathSegment]:
    segments = []
    for start, end in zip(points, points[1:]):
        if _distance(start, end) <= 1e-9:
            continue
        segments.append(PathSegment(start, end, None, None))
    return segments


def _sample_transit(
    points: Sequence[Point3],
    *,
    speed_mm_s: float,
    max_edge_mm: float,
    path_id: int,
) -> List[PointEdge]:
    return _sample_segments(
        _polyline_segments(points),
        motion_kind="TRANSIT",
        speed_mm_s=speed_mm_s,
        max_edge_mm=max_edge_mm,
        path_id=path_id,
        bead_type=None,
    )


def _work_centroid_xy(paths: Sequence[NaturalPath], beads: Sequence[dict]) -> np.ndarray:
    bead_by_index = {int(bead["index"]): bead for bead in beads}
    points = []
    for path in paths:
        for bead_index in path.bead_indices:
            start, end = _bead_endpoints(bead_by_index[bead_index], 0.0)
            points.extend((start[:2], end[:2]))
    if not points:
        raise ValueError("cannot place Park poses without natural-path beads")
    return np.mean(np.asarray(points, float), axis=0)


def _park_point(robot: int, work_centroid_xy: np.ndarray) -> Point3:
    base = np.asarray(ROBOT_BASES[robot], float)
    facing = float(np.arctan2(
        work_centroid_xy[1] - base[1],
        work_centroid_xy[0] - base[0],
    ))
    return (
        float(base[0] + PARK_REACH_MM * np.cos(facing)),
        float(base[1] + PARK_REACH_MM * np.sin(facing)),
        float(PARK_Z_MM),
    )


def build_robot_point_queues(
    paths: Sequence[NaturalPath],
    assignment: Sequence[int],
    beads: Sequence[dict],
    *,
    config: KitechConfig = DEFAULT_CONFIG,
    rapid_config: RapidConfig = DEFAULT_RAPID_CONFIG,
    max_edge_mm: float = DEFAULT_SAMPLE_STEP_MM,
) -> Tuple[RobotPointQueue, ...]:
    """배정 경로를 층당 한 번의 Park 왕복과 상공 이송으로 연결한다."""

    if N_ROBOTS != 3:
        raise ValueError("paper pointwise environment requires exactly three robots")
    if len(paths) != len(assignment):
        raise ValueError("paths and assignment length differ")
    robots = [int(robot) for robot in assignment]
    if any(robot < 0 or robot >= N_ROBOTS for robot in robots):
        raise ValueError("assignment contains an invalid robot")

    path_z_values = [
        point[2]
        for path in paths
        for segment in path.segments
        for point in (segment.start, segment.end)
    ]
    if not path_z_values:
        raise ValueError("cannot build point queues without natural paths")
    if max(path_z_values) - min(path_z_values) > WORK_PLANE_TOL_MM:
        raise ValueError("point queues must contain exactly one physical layer")
    work_plane_z = float(np.mean(path_z_values))

    work_centroid = _work_centroid_xy(paths, beads)
    parks = tuple(_park_point(robot, work_centroid) for robot in range(N_ROBOTS))
    edge_lists: List[List[PointEdge]] = [[] for _ in range(N_ROBOTS)]
    path_lists: List[List[int]] = [[] for _ in range(N_ROBOTS)]
    flyby_speed = float(config.process.flyby_speed)

    paths_by_robot: List[List[NaturalPath]] = [[] for _ in range(N_ROBOTS)]
    for path, robot in zip(paths, robots):
        paths_by_robot[robot].append(path)
        path_lists[robot].append(path.path_id)

    for robot, robot_paths in enumerate(paths_by_robot):
        if not robot_paths:
            continue
        park = parks[robot]
        first_path = robot_paths[0]
        first_above = (
            first_path.start[0],
            first_path.start[1],
            first_path.start[2] + float(rapid_config.flyby_z_offset),
        )
        edge_lists[robot].extend(_sample_transit(
            (park, first_above, first_path.start),
            speed_mm_s=flyby_speed,
            max_edge_mm=max_edge_mm,
            path_id=first_path.path_id,
        ))

        previous_path: Optional[NaturalPath] = None
        for path in robot_paths:
            if previous_path is not None:
                above_previous_end = (
                    previous_path.end[0],
                    previous_path.end[1],
                    previous_path.end[2] + float(rapid_config.flyby_z_offset),
                )
                above_start = (
                    path.start[0],
                    path.start[1],
                    path.start[2] + float(rapid_config.flyby_z_offset),
                )
                edge_lists[robot].extend(_sample_transit(
                    (previous_path.end, above_previous_end, above_start, path.start),
                    speed_mm_s=flyby_speed,
                    max_edge_mm=max_edge_mm,
                    path_id=path.path_id,
                ))
            edge_lists[robot].extend(sample_natural_path(
                path,
                config=config,
                max_edge_mm=max_edge_mm,
            ))
            previous_path = path

        last_path = robot_paths[-1]
        last_above = (
            last_path.end[0],
            last_path.end[1],
            last_path.end[2] + float(rapid_config.flyby_z_offset),
        )
        edge_lists[robot].extend(_sample_transit(
            (last_path.end, last_above, park),
            speed_mm_s=flyby_speed,
            max_edge_mm=max_edge_mm,
            path_id=last_path.path_id,
        ))

    return tuple(RobotPointQueue(
        robot=robot,
        path_ids=tuple(path_lists[robot]),
        park_xyz=parks[robot],
        edges=tuple(edge_lists[robot]),
        work_plane_z=work_plane_z,
    ) for robot in range(N_ROBOTS))


class PointSchedulingEnv(gym.Env):
    """원본 18D 관측·8행동을 현재 점 큐와 원본식 XY 자세 충돌에 적용한다."""

    metadata = {"render_modes": []}

    def __init__(self, queues: Sequence[RobotPointQueue]):
        super().__init__()
        ordered = sorted(queues, key=lambda queue: queue.robot)
        if N_ROBOTS != 3 or len(ordered) != 3:
            raise ValueError("paper pointwise environment requires three robot queues")
        if [queue.robot for queue in ordered] != list(range(3)):
            raise ValueError("robot queues must contain robot ids 0, 1, and 2")
        self.queues = tuple(ordered)
        self.work_plane_z = tuple(
            self._queue_work_plane_z(queue) for queue in self.queues
        )
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(18,),
            dtype=np.float32,
        )
        points = []
        for queue in self.queues:
            points.append(queue.park_xyz[:2])
            points.extend(edge.end[:2] for edge in queue.edges)
        all_xy = np.asarray(points, float)
        self.xy_min = all_xy.min(axis=0)
        self.xy_max = all_xy.max(axis=0)
        # V4의 다방향 경로는 충돌 회피 대기로 병렬성이 크게 낮아질 수 있다.
        # 모든 구간을 한 대씩만 진행해도 끝낼 수 있는 보수적 상한을 사용한다.
        self.horizon = sum(len(queue.edges) for queue in self.queues) + 200
        self.reset()

    @staticmethod
    def decode_action(action: int) -> np.ndarray:
        return np.asarray([(int(action) >> robot) & 1 for robot in range(3)], int)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.indices = {robot: 0 for robot in range(3)}
        self.done_robot = {
            robot: len(self.queues[robot].edges) == 0 for robot in range(3)
        }
        self.gave_bonus = {robot: False for robot in range(3)}
        self.t = 0
        self.now_sec = 0.0
        self.wait_steps = np.zeros(3, dtype=np.int64)
        self.collision_steps = np.zeros(3, dtype=np.int64)
        self.collision_count = 0
        self.no_progress_counter = 0
        self.records: List[dict] = []
        self.step_log: List[dict] = []
        self.traj_indices: List[List[int]] = [[0, 0, 0]]
        return self._build_obs(), {}

    def _point(self, robot: int, index: int) -> Point3:
        queue = self.queues[robot]
        if index <= 0 or not queue.edges:
            return queue.park_xyz
        return queue.edges[min(index, len(queue.edges)) - 1].end

    @staticmethod
    def _queue_work_plane_z(queue: RobotPointQueue) -> Optional[float]:
        if queue.work_plane_z is not None:
            return float(queue.work_plane_z)
        weld_z = [
            float(edge.end[2])
            for edge in queue.edges
            if edge.motion_kind == "WELD"
        ]
        if not weld_z:
            return None
        if max(weld_z) - min(weld_z) > WORK_PLANE_TOL_MM:
            raise ValueError(f"robot {queue.robot} WELD edges span multiple layers")
        return float(np.mean(weld_z))

    def _is_work_plane_pose(self, robot: int, index: int) -> bool:
        work_z = self.work_plane_z[robot]
        return (
            work_z is not None
            and abs(self._point(robot, index)[2] - work_z) <= WORK_PLANE_TOL_MM
        )

    def _total_remaining(self) -> int:
        return sum(
            len(self.queues[robot].edges) - self.indices[robot]
            for robot in range(3)
        )

    def _configuration_conflicts(self, indices: dict[int, int]) -> bool:
        """논문식 XY 판정은 두 TCP가 모두 실제 작업면에 있을 때만 적용한다."""

        for first in range(3):
            if not self._is_work_plane_pose(first, indices[first]):
                continue
            first_point = self._point(first, indices[first])
            for second in range(first + 1, 3):
                if not self._is_work_plane_pose(second, indices[second]):
                    continue
                second_point = self._point(second, indices[second])
                if legacy_xy_pose_conflict(
                    ROBOT_BASES[first],
                    first_point[:2],
                    ROBOT_BASES[second],
                    second_point[:2],
                ):
                    return True
        return False

    def _maximum_feasible_subset(
        self,
        requested: Sequence[int],
        priority: Sequence[int],
    ) -> Tuple[Tuple[int, ...], dict[int, int]]:
        """최종 동시 자세가 안전한 최대 이동 부분집합을 최대 8개에서 고른다."""

        requested_set = set(int(robot) for robot in requested)
        best_subset: Tuple[int, ...] = ()
        best_indices = dict(self.indices)
        best_key = (-1, ())
        requested_list = sorted(requested_set)
        for subset_mask in range(1 << len(requested_list)):
            subset = tuple(
                robot
                for position, robot in enumerate(requested_list)
                if (subset_mask >> position) & 1
            )
            proposed = dict(self.indices)
            for robot in subset:
                proposed[robot] += 1
            if self._configuration_conflicts(proposed):
                continue
            priority_key = tuple(
                1 if robot in subset else 0 for robot in priority
            )
            key = (len(subset), priority_key)
            if key > best_key:
                best_subset = subset
                best_indices = proposed
                best_key = key
        return best_subset, best_indices

    def _has_feasible_advance(self, priority: Sequence[int]) -> bool:
        unfinished = [
            robot for robot in range(3) if not self.done_robot[robot]
        ]
        subset, _indices = self._maximum_feasible_subset(unfinished, priority)
        return bool(subset)

    def step(self, action: int):
        self.t += 1
        actions = self.decode_action(action)
        done_before = [self.done_robot[robot] for robot in range(3)]
        previous_remaining = self._total_remaining()
        order = sorted(range(3), key=lambda robot: (-self.wait_steps[robot], robot))
        requested = [
            robot
            for robot in range(3)
            if not self.done_robot[robot] and actions[robot] == 1
        ]
        selected, new_indices = self._maximum_feasible_subset(requested, order)
        selected_set = set(selected)
        effective_move = [False, False, False]
        blocked_by_collision = [
            robot in requested and robot not in selected_set for robot in range(3)
        ]
        accepted: List[Tuple[int, int, PointEdge]] = []

        for robot in order:
            if robot not in selected_set:
                continue
            candidate_index = new_indices[robot]
            effective_move[robot] = True
            edge = self.queues[robot].edges[candidate_index - 1]
            accepted.append((robot, candidate_index - 1, edge))
            if candidate_index >= len(self.queues[robot].edges):
                self.done_robot[robot] = True

        self.indices = new_indices
        step_start_sec = self.now_sec
        step_duration_sec = max(
            (edge.duration_sec for _robot, _index, edge in accepted),
            default=1.0,
        )
        edge_duration_sec = [0.0, 0.0, 0.0]
        for robot, _edge_index, edge in accepted:
            edge_duration_sec[robot] = edge.duration_sec
        barrier_wait_sec = [
            step_duration_sec - edge_duration_sec[robot]
            if effective_move[robot] else 0.0
            for robot in range(3)
        ]
        collision_wait_sec = [
            step_duration_sec if blocked_by_collision[robot] else 0.0
            for robot in range(3)
        ]
        commanded_wait_sec = [
            step_duration_sec
            if not done_before[robot] and actions[robot] == 0 else 0.0
            for robot in range(3)
        ]
        self.now_sec += step_duration_sec
        for robot, edge_index, edge in accepted:
            self.records.append({
                "t": self.t,
                "robot": robot,
                "edge_index": edge_index,
                "path_id": edge.path_id,
                "motion_kind": edge.motion_kind,
                "bead_indices": list(edge.bead_indices),
                "geometry": [list(point) for point in edge.geometry],
                "start_sec": step_start_sec,
                "end_sec": step_start_sec + edge.duration_sec,
                "step_end_sec": self.now_sec,
                "duration_sec": edge.duration_sec,
            })

        wait_count = 0
        collision_count = 0
        for robot in range(3):
            if blocked_by_collision[robot]:
                collision_count += 1
                self.collision_steps[robot] += 1
                self.collision_count += 1
            if not self.done_robot[robot] and not effective_move[robot]:
                self.wait_steps[robot] += 1
                wait_count += 1

        self.traj_indices.append([self.indices[robot] for robot in range(3)])
        remaining = self._total_remaining()
        progress = previous_remaining - remaining
        if progress <= 0:
            self.no_progress_counter += 1
        else:
            self.no_progress_counter = 0

        reward = TIME_COST
        reward += PROGRESS_GAIN * float(progress)
        reward += COLLISION_PENALTY * float(collision_count)
        reward += WAIT_PENALTY * float(wait_count)
        for robot in range(3):
            if self.done_robot[robot] and not self.gave_bonus[robot]:
                reward += DONE_BONUS
                self.gave_bonus[robot] = True

        terminated = all(self.done_robot.values())
        deadlock = self.no_progress_counter >= MAX_NO_PROGRESS_STEPS
        horizon_reached = self.t >= self.horizon
        if terminated:
            termination_reason = "complete"
        elif deadlock:
            termination_reason = "deadlock"
        elif horizon_reached:
            termination_reason = "horizon"
        else:
            termination_reason = None
        truncated = termination_reason in ("deadlock", "horizon")
        continuation_feasible = (
            termination_reason == "horizon"
            and remaining > 0
            and self._has_feasible_advance(order)
        )
        if terminated:
            reward += ALL_DONE_BONUS
            reward += TOTAL_WAIT_PENALTY * float(self.wait_steps.sum())
        elif truncated:
            reward -= 0.5 * remaining
            if deadlock:
                reward += DEADLOCK_PENALTY

        self.step_log.append({
            "t": self.t,
            "effective_move": list(effective_move),
            "blocked_by_collision": list(blocked_by_collision),
            "collision": collision_count,
            "remaining": remaining,
            "wait_count": wait_count,
            "no_progress_count": self.no_progress_counter,
            "step_start_sec": step_start_sec,
            "step_end_sec": self.now_sec,
            "step_duration_sec": step_duration_sec,
            "edge_duration_sec": list(edge_duration_sec),
            "barrier_wait_sec": list(barrier_wait_sec),
            "collision_wait_sec": list(collision_wait_sec),
            "commanded_wait_sec": list(commanded_wait_sec),
            "requested_move": [bool(value) for value in actions],
            "selected_subset": list(selected),
            "termination_reason": termination_reason,
            "continuation_feasible": continuation_feasible,
        })
        total_edges = sum(len(queue.edges) for queue in self.queues)
        info = {
            "collision_count": self.collision_count,
            "total_wait": int(self.wait_steps.sum()),
            "progress": total_edges - remaining,
            "total_path_len": total_edges,
            "indices": [self.indices[robot] for robot in range(3)],
            "effective_move": list(effective_move),
            "blocked_by_collision": list(blocked_by_collision),
            "makespan_sec": self.now_sec,
            "step_duration_sec": step_duration_sec,
            "edge_duration_sec": list(edge_duration_sec),
            "barrier_wait_sec": list(barrier_wait_sec),
            "collision_wait_sec": list(collision_wait_sec),
            "commanded_wait_sec": list(commanded_wait_sec),
            "requested_move": [bool(value) for value in actions],
            "selected_subset": list(selected),
            "termination_reason": termination_reason,
            "continuation_feasible": continuation_feasible,
            "records": list(self.records) if terminated or truncated else None,
        }
        return self._build_obs(), float(reward), terminated, truncated, info

    def _normal_xy(self, point: Point3) -> np.ndarray:
        xy = np.asarray(point[:2], float)
        return 2.0 * (xy - self.xy_min) / np.maximum(
            self.xy_max - self.xy_min,
            1e-6,
        ) - 1.0

    def _build_obs(self) -> np.ndarray:
        observation = []
        for robot in range(3):
            index = self.indices[robot]
            current = self._normal_xy(self._point(robot, index))
            next_index = min(index + 1, len(self.queues[robot].edges))
            next_point = self._normal_xy(self._point(robot, next_index))
            progress = index / max(len(self.queues[robot].edges), 1)
            done = 1.0 if self.done_robot[robot] else 0.0
            observation.extend((
                current[0], current[1], next_point[0], next_point[1], progress, done,
            ))
        return np.asarray(observation, dtype=np.float32)
