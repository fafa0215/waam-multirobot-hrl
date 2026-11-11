# 포인트 단위 HRL 이동·대기 결과를 원래 꼭짓점을 보존한 로봇별 RAPID로 변환하는 모듈
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from config import DEFAULT_CONFIG, KitechConfig
from pipeline.S3_path_planning.S31_types import N_ROBOTS, classify_bead
from pipeline.S3_path_planning.S33_rapid_writer import (
    DEFAULT_RAPID_CONFIG,
    RapidConfig,
    _SPEEDDATA_TAIL,
    _WEAVE_LINE,
    _WELD_S0_LINE,
    _WELD_WEAVE_LINE,
    _robtarget_line,
    _weld_for,
)
from pipeline.S3_path_planning.S39_hrl_paper_training import PointLayerTrainingResult


def _distance(first, second) -> float:
    return float(np.linalg.norm(np.asarray(second, float) - np.asarray(first, float)))


def _same_point(first, second, tolerance: float = 1e-6) -> bool:
    return _distance(first, second) <= tolerance


def _step_edge_maps(result: PointLayerTrainingResult) -> Dict[Tuple[int, int], object]:
    if result.evaluation_stats.get("time_model") == "independent_event_time":
        maps = {}
        for robot, queue in enumerate(result.queues):
            records = sorted((record for record in result.records if record["robot"] == robot),
                             key=lambda record: record["edge_index"])
            if [record["edge_index"] for record in records] != list(range(len(queue.edges))):
                raise ValueError("event records do not cover each robot edge exactly once")
            for record in records:
                maps[(record["edge_index"] + 1, robot)] = queue.edges[record["edge_index"]]
        return maps
    maps: Dict[Tuple[int, int], object] = {}
    counters = [0, 0, 0]
    for step in result.step_log:
        moves = list(step["effective_move"])
        for robot, moved in enumerate(moves):
            if not moved:
                continue
            queue = result.queues[robot]
            if counters[robot] >= len(queue.edges):
                raise ValueError(
                    f"layer {result.layer_idx} robot {robot} advances past its queue"
                )
            maps[(int(step["t"]), robot)] = queue.edges[counters[robot]]
            counters[robot] += 1
    for robot, queue in enumerate(result.queues):
        if counters[robot] != len(queue.edges):
            raise ValueError(
                f"layer {result.layer_idx} robot {robot} emitted "
                f"{counters[robot]}/{len(queue.edges)} edges"
            )
    return maps


def _robot_steps(result, robot):
    if result.evaluation_stats.get("time_model") != "independent_event_time":
        return list(result.step_log)
    steps = []
    cursor = 0.0
    for record in sorted((record for record in result.records if record["robot"] == robot),
                         key=lambda record: record["edge_index"]):
        start, end = float(record["start_sec"]), float(record["end_sec"])
        if start < cursor - 1e-7 or end <= start:
            raise ValueError("overlapping or invalid robot event records")
        if start > cursor + 1e-9:
            steps.append({"t": -(record["edge_index"] + 1), "step_start_sec": cursor,
                          "step_end_sec": start})
        steps.append({"t": record["edge_index"] + 1, "step_start_sec": start,
                      "step_end_sec": end})
        cursor = end
    return steps


def _step_duration(step: dict) -> float:
    if "step_duration_sec" in step:
        return float(step["step_duration_sec"])
    return float(step["step_end_sec"]) - float(step["step_start_sec"])


def _edge_duration_by_robot(step: dict, edge_maps: dict, robot: int) -> float:
    values = step.get("edge_duration_sec")
    if isinstance(values, (list, tuple)) and len(values) > robot:
        return float(values[robot])
    edge = edge_maps.get((int(step["t"]), robot))
    return 0.0 if edge is None else float(edge.duration_sec)


def _continues_weld(
    steps: Sequence[dict],
    position: int,
    robot: int,
    edge_maps: dict,
) -> bool:
    step = steps[position]
    edge = edge_maps.get((int(step["t"]), robot))
    if edge is None or edge.motion_kind != "WELD":
        return False
    barrier_wait = _step_duration(step) - _edge_duration_by_robot(step, edge_maps, robot)
    if barrier_wait > 1e-6 or position + 1 >= len(steps):
        return False
    next_step = steps[position + 1]
    next_edge = edge_maps.get((int(next_step["t"]), robot))
    return bool(
        next_edge is not None
        and next_edge.motion_kind == "WELD"
        and next_edge.path_id == edge.path_id
        and _same_point(edge.end, next_edge.start)
    )


def _target_prefix(bead_type: str | None) -> str:
    value = str(bead_type or "i_f")
    for known in ("wall", "s_f", "s_o", "i_f", "i_o"):
        if value.startswith(known):
            return known
    return "i_f"


def _task_type(bead_type: str | None) -> str:
    return "infill" if str(bead_type or "").startswith("i_") else "shell"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def generate_pointwise_rapid_per_robot(
    layer_results: Sequence[PointLayerTrainingResult],
    out_dir: str | os.PathLike,
    *,
    config: KitechConfig = DEFAULT_CONFIG,
    rapid_config: RapidConfig = DEFAULT_RAPID_CONFIG,
) -> dict:
    """층별 포인트 타임라인을 3개 MOD로 방출하고 실제 명령시간 요약을 반환한다."""

    if N_ROBOTS != 3:
        raise ValueError("pointwise RAPID writer requires exactly three robots")
    results = list(layer_results)
    if not results:
        raise ValueError("at least one completed layer result is required")
    if any(not result.complete for result in results):
        raise ValueError("incomplete layer result cannot be converted to RAPID")
    expected_layers = list(range(len(results)))
    actual_layers = [int(result.layer_idx) for result in results]
    if actual_layers != expected_layers:
        raise ValueError(
            f"layer results must be contiguous from zero, got {actual_layers}"
        )

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    edge_maps_by_layer = {
        result.layer_idx: _step_edge_maps(result) for result in results
    }
    v_fly = float(config.process.flyby_speed)
    v_weld = float(config.weaving.weld_speed_normal)
    v_weave = float(config.weaving.weld_speed_weave)
    speed_values = {"v_weld": v_weld, "v_weave": v_weave}
    park_transition_sec = {}
    for layer_position, result in enumerate(results):
        if layer_position == 0:
            durations = [0.0, 0.0, 0.0]
        else:
            previous = results[layer_position - 1]
            durations = [
                _distance(
                    previous.queues[robot].park_xyz,
                    result.queues[robot].park_xyz,
                ) / v_fly
                for robot in range(3)
            ]
        barrier = max(durations)
        for robot, duration in enumerate(durations):
            park_transition_sec[(result.layer_idx, robot)] = (duration, barrier)
    summary = {
        "generator": "S40_hrl_pointwise_rapid.generate_pointwise_rapid_per_robot",
        "sync": ("independent_event_time" if all(result.evaluation_stats.get("time_model") ==
                 "independent_event_time" for result in results) else "pointwise_barrier_waittime"),
        "transit": "continuous_layer_path",
        "layers": len(results),
        "speeds": {"flyby": v_fly, "weld": v_weld, "weave": v_weave},
        "files": {},
        "robots": {},
    }

    for robot in range(3):
        targets: List[str] = []
        commands: List[str] = []
        target_counts = {
            "wall": 0,
            "s_f": 0,
            "s_o": 0,
            "i_f": 0,
            "i_o": 0,
            "fly": 0,
        }
        cursor = 0.0
        current_position = None
        previous_park = None
        arc_open = False
        arc_path_id = None
        arc_runs = 0
        weld_length = 0.0
        weld_seconds = 0.0
        transit_length = 0.0
        wait_seconds = 0.0
        emitted_edges = 0
        layer_summaries = []

        def add_target(prefix: str, point, *, home: bool = False) -> str:
            index = target_counts[prefix]
            target_counts[prefix] += 1
            name = f"{prefix}{index}" if prefix != "fly" else f"Flyby_cw{index}"
            targets.append(_robtarget_line(
                name,
                float(point[0]),
                float(point[1]),
                float(point[2]),
                rapid_config,
                rapid_config.confdata_home if home else None,
            ))
            return name

        for result in results:
            queue = result.queues[robot]
            layer_cursor_start = cursor
            park_name = f"park_R{robot}_L{result.layer_idx}"
            targets.append(_robtarget_line(
                park_name,
                queue.park_xyz[0],
                queue.park_xyz[1],
                queue.park_xyz[2],
                rapid_config,
                rapid_config.confdata_home,
            ))
            commands.append(
                f"                ! LAYER index={result.layer_idx} z={result.layer_z}"
            )
            commands.append(
                f"                MoveJ {park_name},v_fly,fine,"
                f"{rapid_config.tool_name}\\WObj:={rapid_config.wobj_name};"
            )
            if previous_park is not None:
                park_move, park_barrier = park_transition_sec[
                    (result.layer_idx, robot)
                ]
                cursor += park_move
                transit_length += _distance(previous_park, queue.park_xyz)
                park_wait = max(0.0, park_barrier - park_move)
                if park_wait > 1e-9:
                    commands.append(f"                WaitTime {park_wait:.6f};")
                    cursor += park_wait
                    wait_seconds += park_wait
            current_position = np.asarray(queue.park_xyz, float)

            steps = _robot_steps(result, robot)
            edge_maps = edge_maps_by_layer[result.layer_idx]
            for step_position, step in enumerate(steps):
                step_seconds = _step_duration(step)
                edge = edge_maps.get((int(step["t"]), robot))
                if edge is None:
                    if arc_open:
                        raise ValueError("an open arc reached a wait-only decision step")
                    if step_seconds > 1e-9:
                        commands.append(f"                WaitTime {step_seconds:.6f};")
                        cursor += step_seconds
                        wait_seconds += step_seconds
                    continue

                emitted_edges += 1
                edge_seconds = float(edge.duration_sec)
                barrier_wait = max(0.0, step_seconds - edge_seconds)
                geometry = [np.asarray(point, float) for point in edge.geometry]
                if not geometry or not _same_point(geometry[0], current_position):
                    raise ValueError(
                        f"layer {result.layer_idx} robot {robot} edge geometry is discontinuous"
                    )

                if edge.motion_kind == "TRANSIT":
                    if arc_open:
                        raise ValueError("a transit edge started before ArcLEnd")
                    for point in geometry[1:]:
                        is_park = _same_point(point, queue.park_xyz)
                        name = park_name if is_park else add_target("fly", point)
                        operation = "MoveJ" if is_park else "MoveL"
                        zone = "fine" if is_park else "z10"
                        commands.append(
                            f"                {operation} {name},v_fly,{zone},"
                            f"{rapid_config.tool_name}\\WObj:={rapid_config.wobj_name};"
                        )
                        transit_length += _distance(current_position, point)
                        current_position = point
                elif edge.motion_kind == "WELD":
                    prefix = _target_prefix(edge.bead_type)
                    weld_data, weave_suffix, speed_name = _weld_for(prefix)
                    if not arc_open:
                        commands.append(
                            f"                ! TASK layer={result.layer_idx} z={result.layer_z} "
                            f"type={_task_type(prefix)} id={edge.path_id} "
                            f"cluster={edge.path_id} sched_start={float(step['step_start_sec']):.6f} "
                            f"sched_dur={edge_seconds:.6f}"
                        )
                        start_name = add_target(prefix, geometry[0])
                        commands.append(
                            f"                ArcLStart {start_name},v_fly,"
                            f"{rapid_config.seam_name},{weld_data}{weave_suffix},"
                            f"fine,{rapid_config.tool_name}\\WObj:={rapid_config.wobj_name};"
                        )
                        arc_open = True
                        arc_path_id = edge.path_id
                        arc_runs += 1
                    elif arc_path_id != edge.path_id:
                        raise ValueError("a weld run crossed natural-path ids")

                    continuing = _continues_weld(
                        steps,
                        step_position,
                        robot,
                        edge_maps,
                    )
                    for point_index, point in enumerate(geometry[1:], start=1):
                        is_final = point_index == len(geometry) - 1
                        operation = "ArcLEnd" if is_final and not continuing else "ArcL"
                        zone = "fine" if operation == "ArcLEnd" else "z0"
                        name = add_target(prefix, point)
                        commands.append(
                            f"                {operation} {name},{speed_name},"
                            f"{rapid_config.seam_name},{weld_data}{weave_suffix},"
                            f"{zone},{rapid_config.tool_name}\\WObj:={rapid_config.wobj_name};"
                        )
                        segment_length = _distance(current_position, point)
                        weld_length += segment_length
                        weld_seconds += segment_length / speed_values[speed_name]
                        current_position = point
                    if not continuing:
                        arc_open = False
                        arc_path_id = None
                else:
                    raise ValueError(f"unknown point motion kind {edge.motion_kind!r}")

                cursor += edge_seconds
                if barrier_wait > 1e-9:
                    if arc_open:
                        raise ValueError("barrier wait reached an open arc")
                    commands.append(f"                WaitTime {barrier_wait:.6f};")
                    cursor += barrier_wait
                    wait_seconds += barrier_wait

            if arc_open:
                raise ValueError(f"layer {result.layer_idx} ended with an open arc")
            if not _same_point(current_position, queue.park_xyz):
                raise ValueError(
                    f"layer {result.layer_idx} robot {robot} did not return to Park"
                )
            previous_park = np.asarray(queue.park_xyz, float)
            layer_summaries.append({
                "layer_idx": int(result.layer_idx),
                "planned_makespan_sec": float(result.makespan_sec),
                "derived_robot_span_sec": cursor - layer_cursor_start,
            })

        lines = [f"MODULE MainModule_R{robot}"]
        lines.append(
            f"        ! META robot={robot} n_robots=3 frame=cell "
            f"sync={summary['sync']} transit=continuous layers={len(results)}"
        )
        lines.append("        ! 판단점 이동·대기를 V4 실제 경로와 현재 속도로 변환한 적용 버전")
        lines.extend(targets)
        lines.append(
            f"        TASK PERS speeddata v_fly:=[{v_fly},{_SPEEDDATA_TAIL}];"
        )
        lines.append(
            f"        TASK PERS speeddata v_weld:=[{v_weld},{_SPEEDDATA_TAIL}];"
        )
        lines.append(
            f"        TASK PERS speeddata v_weave:=[{v_weave},{_SPEEDDATA_TAIL}];"
        )
        lines.append(_WELD_S0_LINE)
        lines.append(_WELD_WEAVE_LINE)
        lines.append(_WEAVE_LINE)
        lines.append("        PROC main()")
        lines.extend(commands)
        lines.append("        ENDPROC")
        lines.append("ENDMODULE")

        output_path = target_dir / f"robot{robot}.mod"
        _write_text(output_path, "\n".join(lines) + "\n")
        n_paths = sum(len(result.queues[robot].path_ids) for result in results)
        summary["files"][str(robot)] = str(output_path)
        summary["robots"][str(robot)] = {
            "n_layers": len(results),
            "n_natural_paths": n_paths,
            "n_point_edges": emitted_edges,
            "n_targets": len(targets),
            "arc_runs": arc_runs,
            "arc_restarts": max(0, arc_runs - n_paths),
            "weld_length_mm": weld_length,
            "weld_sec": weld_seconds,
            "transit_length_mm": transit_length,
            "wait_sec": wait_seconds,
            "derived_end_sec": cursor,
            "layers": layer_summaries,
        }

    return summary
