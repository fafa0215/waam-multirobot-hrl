# 계획 스케줄 대기와 RAPID 실제 가동률을 같은 산식으로 계산하는 공통 모듈.
"""Solver-independent schedule metrics and command-derived RAPID metrics.

The schedule metrics deliberately use the planner time axis.  The RAPID metrics
use accumulated command timestamps and never use collision-audit state labels.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence


_EPS = 1e-9


def _nonnegative_vector(name: str, values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{name} must contain finite non-negative values")
    return result


def _nonnegative_difference(lhs: float, rhs: float, label: str) -> float:
    difference = lhs - rhs
    tolerance = 1e-7 * max(1.0, abs(lhs), abs(rhs))
    if difference < -tolerance:
        raise ValueError(f"{label} would be negative: {lhs} - {rhs}")
    return max(0.0, difference)


def paper_schedule_metrics(
    active_durations_sec: Sequence[float],
    completion_times_sec: Sequence[float],
    makespan_sec: float,
    *,
    collision_waits_sec: Sequence[float] | None = None,
    collision_wait_status: str | None = None,
) -> dict[str, object]:
    """Calculate the paper-style work and wait decomposition.

    ``active_durations_sec`` is planned work time, including every motion that
    the solver models.  For this project that means deposition, internal moves,
    and Park round trips.  ``completion_times_sec`` is each robot's last planned
    completion time on the same time axis as ``makespan_sec``.
    """

    work = _nonnegative_vector("active_durations_sec", active_durations_sec)
    completion = _nonnegative_vector(
        "completion_times_sec", completion_times_sec
    )
    if len(work) != len(completion):
        raise ValueError("active durations and completion times must have equal length")

    makespan = float(makespan_sec)
    if not math.isfinite(makespan) or makespan <= 0.0:
        raise ValueError("makespan_sec must be finite and positive")

    process_wait = [
        _nonnegative_difference(done, active, f"robot {robot} process wait")
        for robot, (active, done) in enumerate(zip(work, completion))
    ]
    completion_wait = [
        _nonnegative_difference(makespan, done, f"robot {robot} completion wait")
        for robot, done in enumerate(completion)
    ]
    total_wait = [
        _nonnegative_difference(makespan, active, f"robot {robot} total wait")
        for robot, active in enumerate(work)
    ]
    for robot, (during, after, total) in enumerate(
        zip(process_wait, completion_wait, total_wait)
    ):
        if not math.isclose(during + after, total, rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError(f"robot {robot} wait decomposition is inconsistent")

    collision_wait: list[float] | None
    if collision_waits_sec is None:
        collision_wait = None
        status = collision_wait_status or "not_recorded"
    else:
        collision_wait = _nonnegative_vector(
            "collision_waits_sec", collision_waits_sec
        )
        if len(collision_wait) != len(work):
            raise ValueError("collision waits and active durations must have equal length")
        for robot, (collision, during) in enumerate(
            zip(collision_wait, process_wait)
        ):
            tolerance = 1e-7 * max(1.0, abs(collision), abs(during))
            if collision > during + tolerance:
                raise ValueError(
                    f"robot {robot} collision wait exceeds its process wait"
                )
        status = collision_wait_status or "measured"
    if not isinstance(status, str) or not status.strip():
        raise ValueError("collision_wait_status must be a non-empty string")

    robot_count = len(work)
    total_work = sum(work)
    total_process_wait = sum(process_wait)
    total_completion_wait = sum(completion_wait)
    total_wait_sum = sum(total_wait)
    total_collision_wait = None if collision_wait is None else sum(collision_wait)

    return {
        "n_robots": robot_count,
        "schedule_makespan_sec": makespan,
        "robot_work_sec": work,
        "robot_completion_sec": completion,
        "robot_process_wait_sec": process_wait,
        "robot_completion_wait_sec": completion_wait,
        "robot_total_wait_sec": total_wait,
        "total_work_sec": total_work,
        "total_process_wait_sec": total_process_wait,
        "total_completion_wait_sec": total_completion_wait,
        "total_wait_sec": total_wait_sum,
        "average_work_sec": total_work / robot_count,
        "average_process_wait_sec": total_process_wait / robot_count,
        "average_completion_wait_sec": total_completion_wait / robot_count,
        "average_total_wait_sec": total_wait_sum / robot_count,
        "paper_wait_ratio_pct": 100.0 * total_wait_sum / (robot_count * makespan),
        "paper_process_wait_share_pct": (
            100.0 * total_process_wait / total_wait_sum
            if total_wait_sum > 0.0
            else 0.0
        ),
        "schedule_motion_utilization_pct": (
            100.0 * total_work / (robot_count * makespan)
        ),
        "robot_collision_wait_sec": collision_wait,
        "total_collision_wait_sec": total_collision_wait,
        "average_collision_wait_sec": (
            None
            if total_collision_wait is None
            else total_collision_wait / robot_count
        ),
        "collision_wait_share_of_total_wait_pct": (
            None
            if total_collision_wait is None
            else (
                100.0 * total_collision_wait / total_wait_sum
                if total_wait_sum > 0.0
                else 0.0
            )
        ),
        "collision_wait_status": status,
    }


def _events_from_mod_dir(mod_dir: str):
    from verify.rapid_timeline import events_from_mod_dir

    return events_from_mod_dir(mod_dir, timing="accumulated")


def _merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(
        (float(start), float(end))
        for start, end in intervals
        if float(end) - float(start) > _EPS
    )
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0:
            raise ValueError("interval endpoints must be finite and non-negative")
        if merged and start <= merged[-1][1] + _EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def _covered_by(interval: tuple[float, float], covering: Sequence[tuple[float, float]]) -> bool:
    start, end = interval
    return any(
        cover_start <= start + _EPS and cover_end >= end - _EPS
        for cover_start, cover_end in covering
    )


def _concurrent_durations(
    intervals_by_robot: dict[int, Sequence[tuple[float, float]]],
) -> tuple[float, float]:
    changes: dict[float, int] = {}
    for intervals in intervals_by_robot.values():
        for start, end in intervals:
            changes[start] = changes.get(start, 0) + 1
            changes[end] = changes.get(end, 0) - 1

    concurrent_two_plus = 0.0
    concurrent_three = 0.0
    active_count = 0
    previous = 0.0
    for timestamp in sorted(changes):
        duration = timestamp - previous
        if active_count >= 2:
            concurrent_two_plus += duration
        if active_count >= 3:
            concurrent_three += duration
        active_count += changes[timestamp]
        previous = timestamp
    return concurrent_two_plus, concurrent_three


def rapid_execution_metrics(mod_dir: str | Path) -> dict[str, object]:
    """Calculate physical motion and weld utilization from accumulated RAPID time.

    Consecutive waypoint intervals count as active only when XYZ actually changes.
    Therefore stationary ``WaitTime`` waypoint pairs are excluded.  Weld time is
    the union of the parser's raw bead intervals.  Audit ``robot_state`` labels
    are intentionally not read here.
    """

    directory = Path(mod_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"RAPID directory does not exist: {directory}")

    _events, tasks, info = _events_from_mod_dir(str(directory))
    waypoints = list(info.get("waypoints") or [])
    if not waypoints:
        raise ValueError(f"no RAPID waypoints found in {directory}")

    waypoints_by_robot: dict[int, list[dict]] = {}
    for waypoint in waypoints:
        robot = int(waypoint["robot"])
        waypoints_by_robot.setdefault(robot, []).append(waypoint)
    for task in tasks:
        waypoints_by_robot.setdefault(int(task.robot), [])
    robot_ids = sorted(waypoints_by_robot)

    active_intervals: dict[int, list[tuple[float, float]]] = {}
    robot_program_end: list[float] = []
    robot_first_active_start: list[float | None] = []
    robot_last_active_end: list[float | None] = []
    for robot in robot_ids:
        robot_waypoints = waypoints_by_robot[robot]
        previous_time = -math.inf
        for waypoint in robot_waypoints:
            timestamp = float(waypoint["t"])
            if not math.isfinite(timestamp) or timestamp < previous_time - _EPS:
                raise ValueError(f"robot {robot} waypoint timestamps are not monotonic")
            previous_time = timestamp

        intervals: list[tuple[float, float]] = []
        for previous_waypoint, waypoint in zip(
            robot_waypoints, robot_waypoints[1:]
        ):
            start = float(previous_waypoint["t"])
            end = float(waypoint["t"])
            displacement = math.dist(
                tuple(float(previous_waypoint[axis]) for axis in ("x", "y", "z")),
                tuple(float(waypoint[axis]) for axis in ("x", "y", "z")),
            )
            if end - start > _EPS and displacement > _EPS:
                intervals.append((start, end))
        active_intervals[robot] = _merge_intervals(intervals)
        if active_intervals[robot]:
            robot_first_active_start.append(active_intervals[robot][0][0])
            robot_last_active_end.append(active_intervals[robot][-1][1])
        else:
            robot_first_active_start.append(None)
            robot_last_active_end.append(None)
        robot_program_end.append(float(robot_waypoints[-1]["t"]))

    weld_raw: dict[int, list[tuple[float, float]]] = {
        robot: [] for robot in robot_ids
    }
    for task in tasks:
        robot = int(task.robot)
        for bead in task.beads:
            weld_raw[robot].append((float(bead.t_start), float(bead.t_end)))
    weld_intervals = {
        robot: _merge_intervals(weld_raw[robot]) for robot in robot_ids
    }
    for robot in robot_ids:
        for weld_interval in weld_intervals[robot]:
            if not _covered_by(weld_interval, active_intervals[robot]):
                raise ValueError(
                    f"robot {robot} weld interval is not covered by active motion"
                )

    total_end = max(robot_program_end)
    if total_end <= 0.0:
        raise ValueError("RAPID total end time must be positive")
    robot_active = [
        _interval_duration(active_intervals[robot]) for robot in robot_ids
    ]
    robot_weld = [
        _interval_duration(weld_intervals[robot]) for robot in robot_ids
    ]
    concurrent_two_plus, concurrent_three = _concurrent_durations(active_intervals)
    robot_count = len(robot_ids)
    robot_initial_wait: list[float] = []
    robot_process_wait: list[float] = []
    robot_completion_wait: list[float] = []
    robot_total_wait: list[float] = []
    for index, active in enumerate(robot_active):
        first = robot_first_active_start[index]
        last = robot_last_active_end[index]
        if first is None or last is None:
            initial_wait = 0.0
            process_wait = 0.0
            completion_wait = total_end
        else:
            initial_wait = first
            process_wait = _nonnegative_difference(
                last - first,
                active,
                f"robot {robot_ids[index]} RAPID process wait",
            )
            completion_wait = _nonnegative_difference(
                total_end,
                last,
                f"robot {robot_ids[index]} RAPID completion wait",
            )
        total_wait = _nonnegative_difference(
            total_end,
            active,
            f"robot {robot_ids[index]} RAPID total wait",
        )
        if not math.isclose(
            initial_wait + process_wait + completion_wait,
            total_wait,
            rel_tol=1e-9,
            abs_tol=1e-7,
        ):
            raise ValueError(
                f"robot {robot_ids[index]} RAPID wait decomposition is inconsistent"
            )
        robot_initial_wait.append(initial_wait)
        robot_process_wait.append(process_wait)
        robot_completion_wait.append(completion_wait)
        robot_total_wait.append(total_wait)

    return {
        "source": "rapid_waypoints_and_beads",
        "timing": "accumulated",
        "n_robots": robot_count,
        "robot_ids": robot_ids,
        "rapid_total_end_sec": total_end,
        "robot_program_end_sec": robot_program_end,
        "robot_active_sec": robot_active,
        "robot_weld_sec": robot_weld,
        "robot_first_active_start_sec": robot_first_active_start,
        "robot_last_active_end_sec": robot_last_active_end,
        "robot_initial_wait_sec": robot_initial_wait,
        "robot_process_wait_sec": robot_process_wait,
        "robot_completion_wait_sec": robot_completion_wait,
        "robot_total_wait_sec": robot_total_wait,
        "robot_wait_ratio_pct": [
            100.0 * value / total_end for value in robot_total_wait
        ],
        "total_active_sec": sum(robot_active),
        "total_weld_sec": sum(robot_weld),
        "total_initial_wait_sec": sum(robot_initial_wait),
        "total_process_wait_sec": sum(robot_process_wait),
        "total_completion_wait_sec": sum(robot_completion_wait),
        "total_wait_sec": sum(robot_total_wait),
        "average_active_sec": sum(robot_active) / robot_count,
        "average_process_wait_sec": sum(robot_process_wait) / robot_count,
        "average_total_wait_sec": sum(robot_total_wait) / robot_count,
        "rapid_wait_ratio_pct": (
            100.0 * sum(robot_total_wait) / (robot_count * total_end)
        ),
        "rapid_process_wait_share_pct": (
            100.0 * sum(robot_process_wait) / sum(robot_total_wait)
            if sum(robot_total_wait) > 0.0
            else 0.0
        ),
        "robot_active_utilization_pct": [
            100.0 * value / total_end for value in robot_active
        ],
        "robot_weld_utilization_pct": [
            100.0 * value / total_end for value in robot_weld
        ],
        "rapid_active_utilization_pct": (
            100.0 * sum(robot_active) / (robot_count * total_end)
        ),
        "rapid_weld_utilization_pct": (
            100.0 * sum(robot_weld) / (robot_count * total_end)
        ),
        "rapid_concurrent_2plus_sec": concurrent_two_plus,
        "rapid_concurrent_3_sec": concurrent_three,
        "rapid_concurrent_2plus_pct": 100.0 * concurrent_two_plus / total_end,
        "rapid_concurrent_3_pct": 100.0 * concurrent_three / total_end,
    }
