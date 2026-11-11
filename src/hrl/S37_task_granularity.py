# V4 비드를 공통 TASK 계층으로 나누고 TopK 기준 스케줄을 생성하는 모듈
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from config import DEFAULT_CONFIG, KitechConfig
from pipeline.S3_path_planning.S31_analysis import build_clusters_from_groups
from pipeline.S3_path_planning.S31_types import (
    N_ROBOTS,
    ROBOT_BASES,
    LayerAnalysis,
    bead_duration_sec,
)
from pipeline.S3_path_planning.S35_collision_audit import (
    compute_all_task_conflict_3d,
)
from pipeline.S3_path_planning.S36_hrl_layerwise import (
    HrlTask,
    LayerHrlResult,
    analysis_to_hrl_tasks,
)


@dataclass(frozen=True)
class TaskGroup:
    """한 TASK에 속하는 비드 인덱스와 최초 TASK-8 그룹 계보."""

    root_id: int
    bead_indices: Tuple[int, ...]


@dataclass(frozen=True)
class _PartitionGroup:
    root_id: int
    atoms: Tuple[Tuple[int, ...], ...]

    @property
    def bead_indices(self) -> Tuple[int, ...]:
        return tuple(index for atom in self.atoms for index in atom)


def phase0_base_groups(analysis: LayerAnalysis) -> List[TaskGroup]:
    """현재 Phase0의 cluster와 wall task를 원래 순서의 TASK-8로 변환한다."""

    groups: List[TaskGroup] = []
    members = [*analysis.clusters, *analysis.shell_tasks]
    for root_id, task in enumerate(members):
        indices = tuple(int(value) for value in task.bead_indices)
        if not indices:
            raise ValueError(f"Phase0 task {root_id} has no beads")
        groups.append(TaskGroup(root_id=root_id, bead_indices=indices))
    if not groups:
        raise ValueError("Phase0 analysis has no tasks")
    return groups


def natural_path_atoms(
    base_groups: Sequence[TaskGroup],
    beads: Sequence[dict],
) -> List[TaskGroup]:
    """infill 선분, port shell loop, wall 연속 arc를 분할 금지 원자로 만든다."""

    bead_by_index = {int(bead["index"]): bead for bead in beads}
    atoms: List[TaskGroup] = []
    for group in base_groups:
        members = [bead_by_index[index] for index in group.bead_indices]
        position = 0
        while position < len(members):
            bead_type = str(members[position].get("type", ""))
            if bead_type.startswith("i_"):
                atoms.append(TaskGroup(
                    group.root_id, (int(members[position]["index"]),)))
                position += 1
                continue
            end = position + 1
            while end < len(members) and str(members[end].get("type", "")) == bead_type:
                end += 1
            same_type = members[position:end]
            chain: List[int] = []
            previous_end = None
            for bead in same_type:
                center = np.asarray([bead["x"], bead["y"]], dtype=float)
                direction = np.asarray([bead["dx"], bead["dy"]], dtype=float)
                half = 0.5 * float(bead["length"]) * direction
                start, finish = center - half, center + half
                if (chain and previous_end is not None
                        and float(np.linalg.norm(start - previous_end)) > 1e-3):
                    atoms.append(TaskGroup(group.root_id, tuple(chain)))
                    chain = []
                chain.append(int(bead["index"]))
                previous_end = finish
            if chain:
                atoms.append(TaskGroup(group.root_id, tuple(chain)))
            position = end
    return atoms


def _group_deposition_sec(
    group: TaskGroup,
    bead_by_index: Dict[int, dict],
    config: KitechConfig,
) -> float:
    return float(sum(
        bead_duration_sec(bead_by_index[index], config)
        for index in group.bead_indices
    ))


def _split_balanced(
    group: _PartitionGroup,
    bead_by_index: Dict[int, dict],
    config: KitechConfig,
) -> Tuple[_PartitionGroup, _PartitionGroup]:
    """자연 경로 경계를 우선 보존하며 누적 적층시간 균형점에서 둘로 나눈다."""

    if len(group.bead_indices) < 2:
        raise ValueError("A one-bead task cannot be split")
    if len(group.atoms) > 1:
        units = list(group.atoms)
    else:
        units = [(index,) for index in group.atoms[0]]
    durations = [
        sum(bead_duration_sec(bead_by_index[index], config) for index in unit)
        for unit in units
    ]
    total = float(sum(durations))
    running = 0.0
    best_cut = 1
    best_error = float("inf")
    for cut in range(1, len(units)):
        running += float(durations[cut - 1])
        error = abs(total - 2.0 * running)
        if error < best_error:
            best_cut = cut
            best_error = error
    return (
        _PartitionGroup(group.root_id, tuple(units[:best_cut])),
        _PartitionGroup(group.root_id, tuple(units[best_cut:])),
    )


def split_task_groups(
    base_groups: Sequence[TaskGroup],
    beads: Sequence[dict],
    target_count: int,
    config: KitechConfig = DEFAULT_CONFIG,
    *,
    path_atoms: Sequence[TaskGroup] | None = None,
) -> List[TaskGroup]:
    """TASK-8 계보를 보존하면서 정확히 target_count개의 중첩 TASK를 만든다."""

    bead_by_index = {int(bead["index"]): bead for bead in beads}
    atom_source = list(path_atoms or [
        TaskGroup(group.root_id, (index,))
        for group in base_groups
        for index in group.bead_indices
    ])
    atoms_by_root: Dict[int, List[Tuple[int, ...]]] = {}
    for atom in atom_source:
        atoms_by_root.setdefault(atom.root_id, []).append(tuple(atom.bead_indices))
    groups = [
        _PartitionGroup(
            group.root_id,
            tuple(atoms_by_root.get(group.root_id, ())),
        )
        for group in base_groups
    ]
    for base, partition in zip(base_groups, groups):
        if partition.bead_indices != tuple(base.bead_indices):
            raise ValueError(
                f"Path atoms do not preserve base group {base.root_id} order")
    if target_count < len(groups):
        raise ValueError(
            f"target_count={target_count} is smaller than base count={len(groups)}")
    total_beads = sum(len(group.bead_indices) for group in groups)
    if target_count > total_beads:
        raise ValueError(
            f"target_count={target_count} exceeds bead count={total_beads}")

    expected = [int(bead["index"]) for bead in beads]
    flattened = [index for group in groups for index in group.bead_indices]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(expected):
        raise ValueError("Base groups do not cover every bead exactly once")

    while len(groups) < target_count:
        atom_boundary_candidates = [
            (index, group) for index, group in enumerate(groups)
            if len(group.atoms) > 1
        ]
        candidates = atom_boundary_candidates or [
            (index, group) for index, group in enumerate(groups)
            if len(group.bead_indices) > 1
        ]
        if not candidates:
            raise RuntimeError("No splittable group remains")
        split_index, selected = max(
            candidates,
            key=lambda item: (
                _group_deposition_sec(item[1], bead_by_index, config),
                len(item[1].bead_indices),
                -item[0],
            ),
        )
        left, right = _split_balanced(selected, bead_by_index, config)
        groups[split_index:split_index + 1] = [left, right]
    return [TaskGroup(group.root_id, group.bead_indices) for group in groups]


def _layer_z(layer_name: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(layer_name))
    return float(match.group(0)) if match else 0.0


def build_common_analysis(
    beads: Sequence[dict],
    groups: Sequence[TaskGroup],
    *,
    layer_idx: int = 0,
    layer_name: str = "",
    config: KitechConfig = DEFAULT_CONFIG,
    with_conflict: bool = True,
) -> LayerAnalysis:
    """그룹을 선행조건 없는 공통 task로 만들고 현재 3D 충돌표를 연결한다."""

    bead_by_index = {int(bead["index"]): bead for bead in beads}
    cluster_groups = [
        [bead_by_index[index] for index in group.bead_indices]
        for group in groups
    ]
    clusters = build_clusters_from_groups(cluster_groups, layer_idx, config)
    for cluster in clusters:
        cluster.kind = "infill"

    all_conflict: Dict = {}
    if with_conflict:
        task_samples = [
            ("infill", cluster.id, cluster.sample_xy)
            for cluster in clusters
        ]
        xy = np.asarray([[bead["x"], bead["y"]] for bead in beads], dtype=float)
        work_centroid = xy.mean(axis=0) if len(xy) else np.zeros(2)
        all_conflict = compute_all_task_conflict_3d(
            task_samples,
            _layer_z(layer_name),
            work_centroid,
            flatten_z=False,
        )
    return LayerAnalysis(
        clusters=clusters,
        shell_tasks=[],
        conflict={},
        all_conflict=all_conflict,
        eps=0.0,
        layer_idx=layer_idx,
        layer_name=layer_name,
    )


def validate_task_contract(
    analyses: Iterable[LayerAnalysis],
    beads: Sequence[dict],
) -> dict:
    """모든 단계의 비드 보존과 TASK 수를 검사하고 비교 가능한 요약을 돌려준다."""

    expected = [int(bead["index"]) for bead in beads]
    expected_set = set(expected)
    summaries = []
    for analysis in analyses:
        indices = [
            int(index)
            for cluster in analysis.clusters
            for index in cluster.bead_indices
        ]
        if len(indices) != len(set(indices)):
            raise ValueError(f"TASK-{len(analysis.clusters)} contains duplicate beads")
        if set(indices) != expected_set:
            missing = sorted(expected_set - set(indices))
            extra = sorted(set(indices) - expected_set)
            raise ValueError(
                f"TASK-{len(analysis.clusters)} coverage differs, "
                f"missing={missing}, extra={extra}")
        summaries.append({
            "task_count": len(analysis.clusters),
            "bead_count": len(indices),
            "min_beads_per_task": min(len(c.bead_indices) for c in analysis.clusters),
            "max_beads_per_task": max(len(c.bead_indices) for c in analysis.clusters),
            "deposition_sec": float(sum(c.duration_sec for c in analysis.clusters)),
            "internal_hop_sec": float(sum(c.hop_sec for c in analysis.clusters)),
            "conflict_entries": len(analysis.all_conflict),
        })
    return {"bead_count": len(expected), "levels": summaries}


def topk_common_assignment(
    tasks: Sequence[HrlTask],
    *,
    k: float = 0.35,
) -> Tuple[List[int], Dict[int, List[int]], List[float]]:
    """TopK 거리 랭킹을 공통 로봇별 실제시간으로 적용한다."""

    if not 0.0 < k <= 1.0:
        raise ValueError("k must be in (0, 1]")
    remaining = set(range(len(tasks)))
    queues: Dict[int, List[int]] = {robot: [] for robot in range(N_ROBOTS)}
    loads = [0.0] * N_ROBOTS
    assignment = [-1] * len(tasks)
    while remaining:
        robot = min(range(N_ROBOTS), key=lambda value: (loads[value], value))
        reachable = [index for index in remaining if tasks[index].eligible_robots[robot]]
        if not reachable:
            fallback = [
                (index, other)
                for index in remaining
                for other in range(N_ROBOTS)
                if tasks[index].eligible_robots[other]
            ]
            if not fallback:
                raise RuntimeError("No eligible robot remains for one or more tasks")
            _index, robot = min(fallback, key=lambda value: (loads[value[1]], value))
            reachable = [index for index in remaining if tasks[index].eligible_robots[robot]]
        ranked = sorted(
            reachable,
            key=lambda index: (
                float(np.linalg.norm(tasks[index].centroid - ROBOT_BASES[robot][:2])),
                index,
            ),
        )
        threshold = k * sum(tasks[index].duration_by_robot[robot] for index in ranked)
        accumulated = 0.0
        for index in ranked:
            queues[robot].append(index)
            assignment[index] = robot
            duration = float(tasks[index].duration_by_robot[robot])
            loads[robot] += duration
            accumulated += duration
            remaining.remove(index)
            if accumulated > threshold:
                break
    return assignment, queues, loads


def _tasks_conflict(
    conflicts: Dict,
    first: HrlTask,
    first_robot: int,
    second: HrlTask,
    second_robot: int,
) -> bool:
    direct = (
        first.task_type, first_robot, first.task_id,
        second.task_type, second_robot, second.task_id,
    )
    reverse = (
        second.task_type, second_robot, second.task_id,
        first.task_type, first_robot, first.task_id,
    )
    return bool(conflicts.get(direct) or conflicts.get(reverse))


def topk_common_schedule(
    tasks: Sequence[HrlTask],
    queues: Dict[int, List[int]],
    conflicts: Dict,
) -> Tuple[List[dict], float, dict]:
    """TopK의 고정 큐를 현재 3D 충돌표에 맞춰 가능한 즉시 시작한다."""

    positions = {robot: 0 for robot in range(N_ROBOTS)}
    active: Dict[int, Tuple[int, float, float]] = {}
    records: List[dict] = []
    now_sec = 0.0
    blocked_attempts = 0
    decision_steps = 0
    while True:
        exhausted = all(positions[r] >= len(queues[r]) for r in range(N_ROBOTS))
        if exhausted and not active:
            break
        starting: List[Tuple[int, int]] = []
        for robot in range(N_ROBOTS):
            if robot in active or positions[robot] >= len(queues[robot]):
                continue
            task_index = queues[robot][positions[robot]]
            task = tasks[task_index]
            blocked = any(
                _tasks_conflict(
                    conflicts, task, robot, tasks[other_index], other_robot)
                for other_robot, (other_index, _start, _end) in active.items()
            ) or any(
                _tasks_conflict(
                    conflicts, task, robot, tasks[other_index], other_robot)
                for other_robot, other_index in starting
            )
            if blocked:
                blocked_attempts += 1
                continue
            starting.append((robot, task_index))
        for robot, task_index in starting:
            duration = float(tasks[task_index].duration_by_robot[robot])
            end_sec = now_sec + duration
            active[robot] = (task_index, now_sec, end_sec)
            positions[robot] += 1
            records.append({
                "task_index": task_index,
                "robot": robot,
                "start_sec": now_sec,
                "end_sec": end_sec,
            })
        decision_steps += 1
        if active:
            now_sec = min(end for _index, _start, end in active.values())
            finished = [
                robot for robot, (_index, _start, end) in active.items()
                if end <= now_sec + 1e-9
            ]
            for robot in finished:
                active.pop(robot)
        elif not starting:
            raise RuntimeError("TopK scheduler deadlocked")
    return records, now_sec, {
        "blocked_attempts": blocked_attempts,
        "decision_steps": decision_steps,
        "truncated": False,
    }


def run_topk_common(
    analysis: LayerAnalysis,
    *,
    config: KitechConfig = DEFAULT_CONFIG,
    k: float = 0.35,
) -> LayerHrlResult:
    """공통 TASK·시간·충돌 계약으로 TopK를 실행한다."""

    tasks = analysis_to_hrl_tasks(analysis, config)
    assignment, queues, loads = topk_common_assignment(tasks, k=k)
    records, makespan, stats = topk_common_schedule(
        tasks, queues, analysis.all_conflict)
    stats["robot_loads_sec"] = loads
    stats["queues"] = {str(robot): list(values) for robot, values in queues.items()}
    return LayerHrlResult(
        layer_idx=analysis.layer_idx,
        tasks=list(tasks),
        assignment=assignment,
        records=records,
        makespan_sec=float(makespan),
        complete=len(records) == len(tasks),
        meta_metrics=[],
        training_wall_sec=0.0,
        evaluation_stats=stats,
    )
