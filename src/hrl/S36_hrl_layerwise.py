# 현재 V4 작업을 계층적 강화학습 배정·스케줄 환경에 연결하는 모듈
"""Layer-wise HRL adapter for the current KITECH scheduling pipeline.

The legacy research code assumed every bead was a vertical Y-axis line.  This
module keeps the two-stage PPO idea, but operates on the current Phase-0 task
objects.  Geometry and RAPID generation remain owned by the current pipeline.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import DEFAULT_CONFIG, KitechConfig
from pipeline.S3_path_planning.S31_analysis import phase0_analysis
from pipeline.S3_path_planning.S31_cpsat_scheduler import apply_cell_frame
from pipeline.S3_path_planning.S31_types import (
    CLUSTER_MODE_DEFAULT,
    N_ROBOTS,
    ROBOT_BASES,
    LayerAnalysis,
    ScheduleEntry,
    load_beads,
    sec_to_units,
    task_duration_sec_with_flyby,
)


TaskKey = Tuple[str, object]


@dataclass(frozen=True)
class HrlTask:
    """Current Phase-0 task reduced to the features required by HRL."""

    layer_idx: int
    task_type: str
    task_id: object
    cluster_id: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bead_indices: Tuple[int, ...]
    duration_by_robot: Tuple[float, ...]
    eligible_robots: Tuple[bool, ...]
    predecessor: Optional[TaskKey] = None

    @property
    def key(self) -> TaskKey:
        return self.task_type, self.task_id

    @property
    def bbox_diagonal(self) -> float:
        return float(np.linalg.norm(self.bbox_max - self.bbox_min))


@dataclass
class LayerHrlResult:
    """Deterministic evaluation result for one trained layer."""

    layer_idx: int
    tasks: List[HrlTask]
    assignment: List[int]
    records: List[dict]
    makespan_sec: float
    complete: bool
    meta_metrics: List[dict] = field(default_factory=list)
    training_wall_sec: float = 0.0
    evaluation_stats: dict = field(default_factory=dict)


def _sector_eligibility(centroid: np.ndarray) -> Tuple[bool, ...]:
    """Mirror the current CP-SAT KITECH_SECTOR_ASSIGN robot domain rule."""

    mode = os.environ.get("KITECH_SECTOR_ASSIGN", "off").lower()
    eligible = [True] * N_ROBOTS
    if mode not in ("hard", "soft") or N_ROBOTS == 1:
        return tuple(eligible)
    dists = [float(np.linalg.norm(ROBOT_BASES[r] - centroid[:2]))
             for r in range(N_ROBOTS)]
    if mode == "hard":
        eligible = [False] * N_ROBOTS
        eligible[int(np.argmin(dists))] = True
    else:
        eligible[int(np.argmax(dists))] = False
    return tuple(eligible)


def analysis_to_hrl_tasks(
    analysis: LayerAnalysis,
    config: KitechConfig = DEFAULT_CONFIG,
) -> List[HrlTask]:
    """Convert current infill/shell task objects without changing bead geometry."""

    tasks: List[HrlTask] = []
    for task in analysis.clusters:
        tasks.append(HrlTask(
            layer_idx=analysis.layer_idx,
            task_type="infill",
            task_id=task.id,
            cluster_id=int(task.id),
            centroid=np.asarray(task.centroid, dtype=float),
            bbox_min=np.asarray(task.bbox_min, dtype=float),
            bbox_max=np.asarray(task.bbox_max, dtype=float),
            bead_indices=tuple(int(i) for i in task.bead_indices),
            duration_by_robot=tuple(
                float(task_duration_sec_with_flyby(task, r, config))
                for r in range(N_ROBOTS)),
            eligible_robots=_sector_eligibility(np.asarray(task.centroid, float)),
        ))
    for task in analysis.shell_tasks:
        tasks.append(HrlTask(
            layer_idx=analysis.layer_idx,
            task_type="shell",
            task_id=task.id,
            cluster_id=int(task.nearest_infill_cluster),
            centroid=np.asarray(task.centroid, dtype=float),
            bbox_min=np.asarray(task.bbox_min, dtype=float),
            bbox_max=np.asarray(task.bbox_max, dtype=float),
            bead_indices=tuple(int(i) for i in task.bead_indices),
            duration_by_robot=tuple(
                float(task_duration_sec_with_flyby(task, r, config))
                for r in range(N_ROBOTS)),
            eligible_robots=_sector_eligibility(np.asarray(task.centroid, float)),
            predecessor=("infill", int(task.nearest_infill_cluster)),
        ))
    if not tasks:
        raise ValueError(f"Layer {analysis.layer_idx} has no schedulable tasks")
    return tasks


def _positive_v4_csvs(csv_dir: Path) -> List[Tuple[float, str, Path]]:
    found = []
    pattern = re.compile(r"beads_Z(-?\d+(?:\.\d+)?)\.csv$")
    for path in csv_dir.glob("beads_Z*.csv"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        z_text = match.group(1)
        z_value = float(z_text)
        if z_value > 0:
            found.append((z_value, z_text, path))
    found.sort(key=lambda item: item[0])
    return found


def prepare_v4_layers(
    csv_dir: str | os.PathLike,
    *,
    limit: int = 11,
    config: KitechConfig = DEFAULT_CONFIG,
    with_conflict: bool = True,
    conflict_3d: bool = False,
) -> Tuple[List[str], List[List[dict]], List[LayerAnalysis], tuple, float]:
    """Load V4, apply one common cell frame, and build current Phase-0 tasks."""

    csv_paths = _positive_v4_csvs(Path(csv_dir))
    if len(csv_paths) < limit:
        raise ValueError(
            f"Requested {limit} positive-Z layers, found {len(csv_paths)} in {csv_dir}")
    names = [f"z{z_text}" for _, z_text, _ in csv_paths]
    raw_layers = [load_beads(str(path)) for _, _, path in csv_paths]
    transformed, offset, rotation = apply_cell_frame(
        raw_layers, getattr(config, "cell_frame", None))
    analyses = [
        phase0_analysis(
            beads,
            layer_idx=idx,
            layer_name=names[idx],
            config=config,
            conflict_3d=conflict_3d,
            cluster_mode=CLUSTER_MODE_DEFAULT,
            with_conflict=with_conflict,
        )
        for idx, beads in enumerate(transformed[:limit])
    ]
    return names[:limit], transformed[:limit], analyses, offset, rotation


class TaskAllocationEnv(gym.Env):
    """MaskablePPO environment assigning current task units to three robots."""

    metadata = {"render_modes": []}

    def __init__(self, tasks: Sequence[HrlTask], *, seed: int = 42,
                 shuffle_tasks: bool = True):
        super().__init__()
        if not tasks:
            raise ValueError("TaskAllocationEnv requires at least one task")
        self.tasks = list(tasks)
        self.shuffle_tasks = bool(shuffle_tasks)
        self.rng = np.random.RandomState(seed)
        self.action_space = spaces.Discrete(N_ROBOTS)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(13,), dtype=np.float32)
        points = np.asarray([task.centroid for task in self.tasks], float)
        self.xy_min = points.min(axis=0)
        self.xy_max = points.max(axis=0)
        self.duration_ref = max(
            float(np.mean([min(t.duration_by_robot) for t in self.tasks])), 1.0)
        self.total_duration = max(
            float(sum(min(t.duration_by_robot) for t in self.tasks)), 1.0)
        pair_dists = [
            float(np.linalg.norm(points[i] - points[j]))
            for i in range(len(points)) for j in range(i + 1, len(points))
        ]
        self.distance_ref = max(float(np.median(pair_dists)) if pair_dists else 1.0,
                                1.0)
        self.external_bonus = 0.0
        self.reset(seed=seed)

    @staticmethod
    def _norm(value: float, lower: float, upper: float) -> float:
        if upper - lower <= 1e-9:
            return 0.0
        return float((value - lower) / (upper - lower))

    def _ordered_indices(self) -> np.ndarray:
        infill = [i for i, task in enumerate(self.tasks)
                  if task.task_type == "infill"]
        shell = [i for i, task in enumerate(self.tasks)
                 if task.task_type == "shell"]
        if self.shuffle_tasks:
            self.rng.shuffle(infill)
            self.rng.shuffle(shell)
        return np.asarray(infill + shell, dtype=int)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.order = self._ordered_indices()
        self.position = 0
        self.assignment = np.full(len(self.tasks), -1, dtype=int)
        self.robot_tasks: List[List[int]] = [[] for _ in range(N_ROBOTS)]
        self.robot_loads = np.zeros(N_ROBOTS, dtype=float)
        return self._observation(), {}

    def action_masks(self) -> np.ndarray:
        if self.position >= len(self.tasks):
            return np.ones(N_ROBOTS, dtype=bool)
        task = self.tasks[int(self.order[self.position])]
        return np.asarray(task.eligible_robots, dtype=bool)

    def _cv(self) -> float:
        mean = float(np.mean(self.robot_loads))
        return float(np.std(self.robot_loads) / mean) if mean > 1e-9 else 0.0

    def _compactness(self) -> float:
        values = []
        for indices in self.robot_tasks:
            if len(indices) < 2:
                values.append(0.0)
                continue
            points = np.asarray([self.tasks[i].centroid for i in indices], float)
            center = points.mean(axis=0)
            values.append(float(np.mean(np.linalg.norm(points - center, axis=1)))
                          / self.distance_ref)
        return float(np.mean(values))

    def _proximity(self) -> float:
        weighted, weights = 0.0, 0.0
        for task_idx, robot in enumerate(self.assignment):
            if robot < 0:
                continue
            task = self.tasks[task_idx]
            weight = min(task.duration_by_robot)
            weighted += (float(np.linalg.norm(task.centroid - ROBOT_BASES[robot]))
                         / self.distance_ref) * weight
            weights += weight
        return weighted / weights if weights > 0 else 0.0

    def _score(self) -> float:
        return self._cv() + 0.15 * self._compactness() + 0.10 * self._proximity()

    def step(self, action: int):
        task_idx = int(self.order[self.position])
        robot = int(action)
        if not self.tasks[task_idx].eligible_robots[robot]:
            return self._observation(), -10.0, False, False, {"invalid_action": True}
        before = self._score()
        self.assignment[task_idx] = robot
        self.robot_tasks[robot].append(task_idx)
        self.robot_loads[robot] += self.tasks[task_idx].duration_by_robot[robot]
        after = self._score()
        self.position += 1
        terminated = self.position >= len(self.tasks)
        reward = before - after
        if terminated:
            reward += -self._cv() + float(self.external_bonus)
            obs = np.zeros(13, dtype=np.float32)
        else:
            obs = self._observation()
        info = {
            "robot_loads_sec": self.robot_loads.tolist(),
            "assignment": self.assignment.tolist(),
        }
        return obs, float(reward), terminated, False, info

    def _observation(self) -> np.ndarray:
        if self.position >= len(self.tasks):
            return np.zeros(13, dtype=np.float32)
        task_idx = int(self.order[self.position])
        task = self.tasks[task_idx]
        x = self._norm(task.centroid[0], self.xy_min[0], self.xy_max[0])
        y = self._norm(task.centroid[1], self.xy_min[1], self.xy_max[1])
        duration = min(task.duration_by_robot) / self.duration_ref
        footprint = task.bbox_diagonal / self.distance_ref
        progress = self.position / max(len(self.tasks) - 1, 1)
        loads = self.robot_loads / max(float(np.mean(self.robot_loads)), 1.0)
        dists = [
            float(np.linalg.norm(task.centroid - ROBOT_BASES[r])) / self.distance_ref
            for r in range(N_ROBOTS)
        ]
        remaining = self.order[self.position:]
        remaining_duration = sum(
            min(self.tasks[int(i)].duration_by_robot) for i in remaining)
        obs = [x, y, duration, footprint, progress]
        obs.extend(float(v) for v in loads)
        obs.extend(dists)
        obs.extend([
            len(remaining) / len(self.tasks),
            remaining_duration / self.total_duration,
        ])
        return np.asarray(obs, dtype=np.float32)


class TaskSchedulingEnv(gym.Env):
    """Event-driven PPO environment choosing three robots' start/wait bits."""

    metadata = {"render_modes": []}

    def __init__(self, tasks: Sequence[HrlTask], assignment: Sequence[int],
                 conflicts: Optional[Dict] = None, *, max_steps: Optional[int] = None):
        super().__init__()
        self.tasks = list(tasks)
        self.assignment = [int(r) for r in assignment]
        if len(self.tasks) != len(self.assignment):
            raise ValueError("tasks and assignment length differ")
        if any(r < 0 or r >= N_ROBOTS for r in self.assignment):
            raise ValueError("assignment contains an invalid robot")
        self.queues = {r: [] for r in range(N_ROBOTS)}
        for task_idx, robot in enumerate(self.assignment):
            self.queues[robot].append(task_idx)
        self.conflicts = conflicts or {}
        self.action_space = spaces.Discrete(2 ** N_ROBOTS)
        self.obs_len_per_robot = 7
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0,
            shape=(self.obs_len_per_robot * N_ROBOTS,), dtype=np.float32)
        points = np.asarray([task.centroid for task in self.tasks], float)
        self.xy_min = points.min(axis=0)
        self.xy_max = points.max(axis=0)
        self.duration_ref = max(
            max(max(t.duration_by_robot) for t in self.tasks), 1.0)
        self.serial_duration = max(
            sum(max(t.duration_by_robot) for t in self.tasks), 1.0)
        self.max_steps = max_steps or (40 * len(self.tasks) + 200)
        self.reset()

    @staticmethod
    def decode_action(action: int) -> np.ndarray:
        return np.asarray([(int(action) >> r) & 1 for r in range(N_ROBOTS)], int)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.now_sec = 0.0
        self.steps = 0
        self.next_position = {r: 0 for r in range(N_ROBOTS)}
        self.active: Dict[int, Tuple[int, float, float]] = {}
        self.completed: set[TaskKey] = set()
        self.records: List[dict] = []
        return self._observation(), {}

    def _conflict(self, first_idx: int, first_robot: int,
                  second_idx: int, second_robot: int) -> bool:
        first = self.tasks[first_idx]
        second = self.tasks[second_idx]
        direct = (first.task_type, first_robot, first.task_id,
                  second.task_type, second_robot, second.task_id)
        reverse = (second.task_type, second_robot, second.task_id,
                   first.task_type, first_robot, first.task_id)
        return bool(self.conflicts.get(direct) or self.conflicts.get(reverse))

    def _next_task(self, robot: int) -> Optional[int]:
        pos = self.next_position[robot]
        queue = self.queues[robot]
        return queue[pos] if pos < len(queue) else None

    def _can_start(self, task_idx: int, robot: int,
                   starting: Sequence[Tuple[int, int]]) -> bool:
        task = self.tasks[task_idx]
        if task.predecessor is not None and task.predecessor not in self.completed:
            return False
        for other_robot, (other_idx, _start, _end) in self.active.items():
            if self._conflict(task_idx, robot, other_idx, other_robot):
                return False
        for other_robot, other_idx in starting:
            if self._conflict(task_idx, robot, other_idx, other_robot):
                return False
        return True

    def step(self, action: int):
        self.steps += 1
        bits = self.decode_action(action)
        starting: List[Tuple[int, int]] = []
        blocked = 0
        for robot in range(N_ROBOTS):
            if robot in self.active or bits[robot] == 0:
                continue
            task_idx = self._next_task(robot)
            if task_idx is None:
                continue
            if not self._can_start(task_idx, robot, starting):
                blocked += 1
                continue
            starting.append((robot, task_idx))

        for robot, task_idx in starting:
            duration = self.tasks[task_idx].duration_by_robot[robot]
            end_sec = self.now_sec + duration
            self.active[robot] = (task_idx, self.now_sec, end_sec)
            self.next_position[robot] += 1
            self.records.append({
                "task_index": task_idx,
                "robot": robot,
                "start_sec": self.now_sec,
                "end_sec": end_sec,
            })

        previous_time = self.now_sec
        completed_now = 0
        if self.active:
            self.now_sec = min(end for _idx, _start, end in self.active.values())
            finished = [robot for robot, (_idx, _start, end) in self.active.items()
                        if end <= self.now_sec + 1e-9]
            for robot in finished:
                task_idx, _start, _end = self.active.pop(robot)
                self.completed.add(self.tasks[task_idx].key)
                completed_now += 1
        else:
            self.now_sec += 1.0

        exhausted = all(
            self.next_position[r] >= len(self.queues[r]) for r in range(N_ROBOTS))
        terminated = exhausted and not self.active
        truncated = self.steps >= self.max_steps or self.now_sec > 2.0 * self.serial_duration
        elapsed = self.now_sec - previous_time
        reward = (0.35 * len(starting) + 1.5 * completed_now
                  - 0.002 * elapsed - 0.5 * blocked)
        if not starting and not self.active and not terminated:
            reward -= 1.0
        if terminated:
            reward += 10.0 - 5.0 * self.now_sec / self.serial_duration
        info = {
            "complete": terminated,
            "makespan_sec": self.now_sec,
            "started": len(starting),
            "blocked": blocked,
            "records": list(self.records) if terminated or truncated else None,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self) -> np.ndarray:
        obs = []
        for robot in range(N_ROBOTS):
            task_idx = self._next_task(robot)
            if task_idx is None:
                x = y = 0.0
            else:
                task = self.tasks[task_idx]
                x = 2.0 * (task.centroid[0] - self.xy_min[0]) / max(
                    self.xy_max[0] - self.xy_min[0], 1e-6) - 1.0
                y = 2.0 * (task.centroid[1] - self.xy_min[1]) / max(
                    self.xy_max[1] - self.xy_min[1], 1e-6) - 1.0
            queue = self.queues[robot]
            progress = self.next_position[robot] / max(len(queue), 1)
            remaining_busy = max(
                self.active.get(robot, (0, 0.0, self.now_sec))[2] - self.now_sec,
                0.0) / self.duration_ref
            remaining_queue = (len(queue) - self.next_position[robot]) / max(len(queue), 1)
            active = 1.0 if robot in self.active else 0.0
            done = 1.0 if self.next_position[robot] >= len(queue) and not active else 0.0
            obs.extend([x, y, progress, remaining_busy, remaining_queue, active, done])
        return np.asarray(obs, dtype=np.float32)


def _allocation_rollout(model, tasks: Sequence[HrlTask], seed: int) -> List[int]:
    env = TaskAllocationEnv(tasks, seed=seed, shuffle_tasks=False)
    obs, _ = env.reset(seed=seed)
    terminated = False
    while not terminated:
        action, _ = model.predict(
            obs, deterministic=True, action_masks=env.action_masks())
        obs, _reward, terminated, _truncated, _info = env.step(
            int(np.asarray(action).item()))
    return [int(value) for value in env.assignment]


def _schedule_rollout(model, tasks: Sequence[HrlTask], assignment: Sequence[int],
                      conflicts: Dict, seed: int) -> Tuple[List[dict], float, bool, dict]:
    env = TaskSchedulingEnv(tasks, assignment, conflicts)
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    blocked_attempts = 0
    decision_steps = 0
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(
            int(np.asarray(action).item()))
        blocked_attempts += int(info["blocked"])
        decision_steps += 1
    stats = {
        "blocked_attempts": blocked_attempts,
        "decision_steps": decision_steps,
        "truncated": bool(truncated),
    }
    return list(env.records), float(env.now_sec), bool(terminated), stats


def train_layer_hrl(
    analysis: LayerAnalysis,
    *,
    config: KitechConfig = DEFAULT_CONFIG,
    seed: int = 42,
    meta_episodes: int = 5,
    allocation_episodes: int = 20,
    scheduling_episodes: int = 40,
    model_dir: Optional[str | os.PathLike] = None,
    device: str = "cpu",
    conflicts: Optional[Dict] = None,
) -> LayerHrlResult:
    """Train legacy-style MaskablePPO allocation and PPO start/wait policies."""

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise RuntimeError(
            "HRL dependencies are missing. Install requirements-hrl.txt first.") from exc

    set_random_seed(seed)
    tasks = analysis_to_hrl_tasks(analysis, config)
    effective_conflicts = analysis.all_conflict if conflicts is None else conflicts
    alloc_env = TaskAllocationEnv(tasks, seed=seed, shuffle_tasks=True)
    alloc_steps = max(8, min(64, 2 * len(tasks)))
    allocation_model = MaskablePPO(
        "MlpPolicy", alloc_env, seed=seed, device=device, verbose=0,
        n_steps=alloc_steps, batch_size=alloc_steps,
        policy_kwargs={"net_arch": [64, 64]},
    )
    scheduler_model = None
    best: Optional[Tuple[List[int], List[dict], float, dict]] = None
    metrics = []
    started_at = time.perf_counter()
    for meta_idx in range(meta_episodes):
        allocation_model.learn(
            total_timesteps=max(alloc_steps, allocation_episodes * len(tasks)),
            reset_num_timesteps=False,
        )
        assignment = _allocation_rollout(allocation_model, tasks, seed)
        sched_env = TaskSchedulingEnv(tasks, assignment, effective_conflicts)
        sched_steps = max(16, min(128, 4 * len(tasks)))
        if scheduler_model is None:
            scheduler_model = PPO(
                "MlpPolicy", sched_env, seed=seed, device=device, verbose=0,
                n_steps=sched_steps, batch_size=sched_steps,
                policy_kwargs={"net_arch": [64, 64]},
            )
        else:
            scheduler_model.set_env(sched_env)
        scheduler_model.learn(
            total_timesteps=max(sched_steps, scheduling_episodes * len(tasks)),
            reset_num_timesteps=False,
        )
        records, makespan, complete, evaluation_stats = _schedule_rollout(
            scheduler_model, tasks, assignment, effective_conflicts, seed)
        loads = [0.0] * N_ROBOTS
        for task_idx, robot in enumerate(assignment):
            loads[robot] += tasks[task_idx].duration_by_robot[robot]
        metrics.append({
            "meta_episode": meta_idx + 1,
            "complete": complete,
            "makespan_sec": makespan,
            "robot_loads_sec": loads,
            "evaluation_stats": evaluation_stats,
        })
        alloc_env.external_bonus = (-makespan / max(sum(loads), 1.0)
                                    if complete else -2.0)
        if complete and (best is None or makespan < best[2]):
            best = (assignment, records, makespan, evaluation_stats)

    if scheduler_model is None or best is None:
        raise RuntimeError(
            f"Layer {analysis.layer_idx} HRL policy did not complete within the pilot budget")
    if model_dir is not None:
        target = Path(model_dir)
        target.mkdir(parents=True, exist_ok=True)
        allocation_model.save(str(target / f"layer_{analysis.layer_idx:02d}_allocation"))
        scheduler_model.save(str(target / f"layer_{analysis.layer_idx:02d}_scheduling"))
    return LayerHrlResult(
        layer_idx=analysis.layer_idx,
        tasks=tasks,
        assignment=list(best[0]),
        records=list(best[1]),
        makespan_sec=float(best[2]),
        complete=True,
        meta_metrics=metrics,
        training_wall_sec=time.perf_counter() - started_at,
        evaluation_stats=dict(best[3]),
    )


def build_layered_schedule_result(
    layer_names: Sequence[str],
    all_beads: Sequence[List[dict]],
    analyses: Sequence[LayerAnalysis],
    layer_results: Sequence[LayerHrlResult],
    *,
    cell_offset=(0.0, 0.0),
    cell_rotation: float = 0.0,
    seed: int = 42,
) -> dict:
    """Convert local HRL schedules to current ScheduleEntry with layer barriers."""

    if not (len(layer_names) == len(all_beads) == len(analyses) == len(layer_results)):
        raise ValueError("Layer inputs have different lengths")
    entries: List[ScheduleEntry] = []
    summaries = []
    barrier_sec = 0.0
    workload = [0.0] * N_ROBOTS
    for layer_idx, layer_result in enumerate(layer_results):
        if not layer_result.complete:
            raise ValueError(f"Layer {layer_idx} is incomplete")
        layer_end = barrier_sec
        seen = set()
        for record in layer_result.records:
            task_idx = int(record["task_index"])
            if task_idx in seen:
                raise ValueError(f"Layer {layer_idx} task {task_idx} is scheduled twice")
            seen.add(task_idx)
            task = layer_result.tasks[task_idx]
            robot = int(record["robot"])
            duration_sec = float(task.duration_by_robot[robot])
            start_sec = barrier_sec + float(record["start_sec"])
            start_units = int(round(start_sec * 10.0))
            duration_units = sec_to_units(duration_sec)
            end_units = start_units + duration_units
            entry = ScheduleEntry(
                cluster_id=task.cluster_id,
                robot=robot,
                start=start_units,
                end=end_units,
                duration=duration_units,
                layer_idx=layer_idx,
                task_type=task.task_type,
                shell_id=str(task.task_id) if task.task_type == "shell" else None,
                duration_sec=duration_sec,
            )
            entries.append(entry)
            summaries.append({
                "layer": layer_idx,
                "cluster": task.cluster_id,
                "robot": robot,
                "start": start_units,
                "end": end_units,
                "duration_units": duration_units,
                "duration_sec": duration_sec,
                "type": task.task_type,
                "shell_id": entry.shell_id,
                "bead_indices": list(task.bead_indices),
            })
            workload[robot] += duration_sec
            layer_end = max(layer_end, end_units / 10.0)
        if len(seen) != len(layer_result.tasks):
            missing = sorted(set(range(len(layer_result.tasks))) - seen)
            raise ValueError(f"Layer {layer_idx} has unscheduled tasks: {missing}")
        barrier_sec = layer_end
    mean_load = float(np.mean(workload))
    workload_cv = float(np.std(workload) / mean_load) if mean_load > 0 else 0.0
    return {
        "layer_names": list(layer_names),
        "all_beads": list(all_beads),
        "all_analyses": list(analyses),
        "all_entries": entries,
        "entries_summary": summaries,
        "total_makespan_sec": barrier_sec,
        "config": {
            "scheduler": "hrl_layerwise",
            "seed": int(seed),
            "recenter_offset_mm": list(cell_offset),
            "cell_rotation_deg": float(cell_rotation),
            "layer_barrier": True,
            "robot_bases_mm": np.asarray(ROBOT_BASES, float).tolist(),
        },
        "metrics": {
            "n_tasks": len(entries),
            "robot_workload_sec": workload,
            "workload_cv": workload_cv,
            "utilization": (sum(workload) / (barrier_sec * N_ROBOTS)
                            if barrier_sec > 0 else 0.0),
            "training_wall_sec": sum(r.training_wall_sec for r in layer_results),
        },
    }
