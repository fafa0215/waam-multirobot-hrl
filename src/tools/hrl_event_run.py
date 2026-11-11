# 독립 시간 HRL의 입력·소스·체크포인트를 보존하고 단층 학습과 평가를 실행하는 도구
from __future__ import annotations

import argparse
import copy
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "docs/experiments/2026-08-08_poster-contracts/beads/V4"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    result = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", check=True)
    return result.stdout.strip()


def prepare_source(out):
    files = {}
    for source in sorted((REPO / "code").rglob("*.py")):
        relative = source.relative_to(REPO / "code")
        if any(part in {"results", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        target = out / "source" / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        files[relative.as_posix()] = digest(target)
        if files[relative.as_posix()] != digest(source):
            raise RuntimeError("source changed during snapshot")
    manifest = {"repository": str(REPO), "commit": git("rev-parse", "HEAD"),
                "code_status": git("status", "--short", "--", "code"), "files": files}
    save(out / "source_manifest.json", manifest)
    sys.path.insert(0, str(out / "source" / "code"))
    return manifest


def prepare_input(args, out):
    from config import DEFAULT_CONFIG
    from pipeline.S3_path_planning.S31_cpsat_scheduler import apply_cell_frame
    from pipeline.S3_path_planning.S31_types import load_beads
    from pipeline.S3_path_planning.S36_hrl_layerwise import _positive_v4_csvs
    from pipeline.S3_path_planning.S42_hrl_event_time import csv_natural_paths, curve_flags, sample_adaptive_path

    sources = [path for _, _, path in _positive_v4_csvs(args.input)]
    if not sources:
        raise ValueError("no positive layer CSV inputs")
    hashes = {}
    raw = []
    for source in sources:
        target = out / "input_csv" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[source.name] = digest(target)
        raw.append(load_beads(str(target)))
    variant = args.input / "_variant.json"
    if variant.exists():
        shutil.copy2(variant, out / "input_csv" / variant.name)
        hashes[variant.name] = digest(variant)
    transformed, offset, rotation = apply_cell_frame(raw, DEFAULT_CONFIG.cell_frame)
    beads, layer_z = transformed[0], float(sources[0].stem[7:])
    paths = csv_natural_paths(beads, layer_z=layer_z)
    covered = [index for path in paths for index in path.bead_indices]
    if sorted(covered) != sorted(int(bead["index"]) for bead in beads) or len(set(covered)) != len(covered):
        raise ValueError("natural path coverage differs from CSV")
    records = []
    for path in paths:
        edges = sample_adaptive_path(path, straight_mm=args.straight_mm, curve_mm=10.0)
        if abs(sum(edge.distance_mm for edge in edges) - path.length_mm) > 1e-6:
            raise ValueError("sampling changed deposited length")
        flags = curve_flags(path)
        records.append({"path_id": path.path_id, "bead_indices": list(path.bead_indices),
                        "bead_type": path.bead_type, "length_mm": path.length_mm,
                        "curve_bead_indices": [segment.bead_index for segment, flag in zip(path.segments, flags) if flag],
                        "decision_edges": len(edges)})
    if len(beads) != 280 or len(paths) != 138:
        raise ValueError(f"expected agreed V4 280 rows / 138 paths, got {len(beads)} / {len(paths)}")
    data = {"source_csv_sha256": hashes, "frame_scope": [source.name for source in sources],
            "offset_mm": offset, "rotation_deg": rotation, "layer_z": layer_z, "beads": beads,
            "curve_rule": "both adjacent chords of 0.5..15 degree nonzero turn, closed loop wrap included",
            "paths": records, "total_length_mm": sum(path.length_mm for path in paths)}
    save(out / "prepared_input.json", data)
    return beads, layer_z, paths, data


def plot_preflight(paths, out, straight_mm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pipeline.S3_path_planning.S38_hrl_pointwise import sample_natural_path
    from pipeline.S3_path_planning.S42_hrl_event_time import curve_flags, sample_adaptive_path

    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for axis, adaptive in zip(axes, (False, True)):
        for path in paths:
            for segment, curved in zip(path.segments, curve_flags(path)):
                axis.plot([segment.start[0], segment.end[0]], [segment.start[1], segment.end[1]],
                          color="#d76c18" if curved else "#2475a8", linewidth=1)
            edges = (sample_adaptive_path(path, straight_mm=straight_mm) if adaptive
                     else sample_natural_path(path, max_edge_mm=50.0))
            axis.scatter([edge.end[0] for edge in edges], [edge.end[1] for edge in edges],
                         s=3, color="#222222", alpha=0.6)
        axis.set_aspect("equal")
        axis.set_xlabel("X (mm)")
        axis.set_ylabel("Y (mm)")
    axes[0].set_title("Previous adapter: all WELD decisions <=50 mm")
    axes[1].set_title(f"New: straight <={straight_mm:g} mm / curved <=10 mm")
    axes[1].legend(handles=[Line2D([0], [0], color="#d76c18", label="Curved chord span"),
                           Line2D([0], [0], color="#2475a8", label="Straight span")])
    figure.savefig(out / "sampling_before_after.png", dpi=160)
    plt.close(figure)


def run(args, out):
    # 과거 실험의 환경변수로 이번 계약이 바뀌지 않도록 격리한다.
    inherited = {key: value for key, value in os.environ.items() if key.startswith("KITECH_")}
    for key in inherited:
        del os.environ[key]
    os.environ.update(KITECH_N_ROBOTS="3", KITECH_SECTOR_ASSIGN="off", MPLBACKEND="Agg")
    source = prepare_source(out)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from config import DEFAULT_CONFIG
    from pipeline.S3_path_planning.S39_hrl_paper_training import train_point_hrl_layer
    from pipeline.S3_path_planning.S40_hrl_pointwise_rapid import generate_pointwise_rapid_per_robot
    from pipeline.S3_path_planning.S41_execution_metrics import rapid_execution_metrics, paper_schedule_metrics
    from pipeline.S3_path_planning.S42_hrl_event_time import (
        EventPointSchedulingEnv, MetaEventPointSchedulingEnv, build_event_queues,
    )
    budget = (1, 1, 1) if args.mode == "smoke" else (100, 100, 200)
    save(out / "settings.json", {"version": "hrl-kitech-event-time-v1", "mode": args.mode,
         "seed": 42, "meta_episodes": budget[0], "allocation_episodes": budget[1],
         "scheduling_episodes": budget[2], "straight_mm": args.straight_mm, "curve_mm": 10,
         "transit_mm": 50, "all_idle_wait_sec": 1, "arc_restart_cost_sec": 0,
         "device": "cpu", "torch_threads": 1, "config": dataclasses.asdict(DEFAULT_CONFIG),
         "ignored_inherited_kitech_environment": sorted(inherited),
         "python": sys.version, "executable": sys.executable,
         "packages": {name: importlib.metadata.version(name) for name in
                      ("numpy", "torch", "gymnasium", "stable-baselines3", "sb3-contrib")},
         "method_deviations": ["connected KITECH paths with native speeds and Park/transit",
             "30D state and independent event clock", "completion-edge/event reward frequency",
             "work-plane-gated XY endpoint poses, maximum feasible action subset",
             "completion/makespan-first observed rollout selection; deterministic score also reported"],
         "audit": {"dt_sec": 0.01, "threshold_mm": 100, "pose_source": "rapid",
                   "source_commit": source["commit"], "C1_version_alignment_verified": False,
                   "continuous_safety_guarantee": False}})
    beads, layer_z, paths, data = prepare_input(args, out)
    plot_preflight(paths, out, args.straight_mm)
    started = time.perf_counter()

    def status(row):
        save(out / "status.json", {"status": "training", "utc": datetime.now(timezone.utc).isoformat(),
                                    "wall_sec": time.perf_counter() - started, **row})

    def progress(row):
        with (out / "meta_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        status({"phase": "meta_complete", **row})
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def checkpoint(meta, best, allocation_model, scheduler_model, metrics, wall, allocation, scheduling, allocation_parameters):
        folder = out / "checkpoints" / f"meta_{meta:03d}"
        folder.mkdir(parents=True)
        allocation_model.save(str(folder / "allocation_after_update"))
        scheduler_model.save(str(folder / "scheduler"))
        save(folder / "evaluation.json", {"assignment": allocation["assignment"], **scheduling})
        if best["meta_episode"] == meta:
            shutil.copy2(folder / "scheduler.zip", out / "best_scheduler.zip")
            current = copy.deepcopy(allocation_model.get_parameters())
            allocation_model.set_parameters(allocation_parameters)
            allocation_model.save(str(out / "best_allocation"))
            allocation_model.set_parameters(current)
            save(out / "best_observed.json", {"meta_episode": meta, "assignment": allocation["assignment"], **scheduling})

    status({"phase": "initializing", "meta_episode": 0})
    result = train_point_hrl_layer(
        paths, beads, layer_idx=0, layer_z=layer_z, seed=42,
        meta_episodes=budget[0], allocation_episodes=budget[1], scheduling_episodes=budget[2],
        model_dir=out / "models", device="cpu", progress_callback=progress,
        queue_builder=functools.partial(build_event_queues, straight_mm=args.straight_mm),
        scheduling_environment_factory=MetaEventPointSchedulingEnv,
        evaluation_environment_factory=EventPointSchedulingEnv,
        checkpoint_callback=checkpoint, rollout_callback=status,
    )
    result.evaluation_stats["time_model"] = "independent_event_time"
    save(out / "result.json", dataclasses.asdict(result))
    if not all(row["allocation_ppo_updates"] > 0 and row["scheduling_ppo_updates"] > 0 for row in result.meta_metrics):
        raise RuntimeError("PPO did not update")
    if not result.complete:
        save(out / "status.json", {"status": "training_finished_incomplete", "complete": False,
             "training_wall_sec": result.training_wall_sec, "evaluation_stats": result.evaluation_stats,
             "note": "partial elapsed time is not a completed makespan; no executable RAPID emitted"})
        return
    summary = generate_pointwise_rapid_per_robot([result], out / "rapid")
    save(out / "rapid_summary.json", summary)
    metrics = rapid_execution_metrics(out / "rapid")
    if abs(metrics["rapid_total_end_sec"] - result.makespan_sec) > 0.05:
        raise RuntimeError("RAPID and event makespan differ")
    if abs(sum(robot["weld_length_mm"] for robot in summary["robots"].values()) - data["total_length_mm"]) > 0.05:
        raise RuntimeError("RAPID deposited length differs from input")
    motion_sec = [sum(edge.duration_sec for edge in queue.edges) for queue in result.queues]
    ends = [max((record["end_sec"] for record in result.records if record["robot"] == robot), default=0.0)
            for robot in range(3)]
    save(out / "execution_metrics.json", {"rapid": metrics, "motion": paper_schedule_metrics(
        motion_sec, ends, result.makespan_sec), "scope": "modeled seconds, not hardware measurements"})
    status({"phase": "final_3d_audit", "meta_episode": budget[0]})
    from verify.run_verify import audit_mod_dir
    audit = audit_mod_dir(str(out / "rapid"), timing="accumulated", dt_sec=0.01,
                          threshold_mm=100.0, collect_series=False, pose_source="rapid")
    save(out / "audit.json", audit)
    if audit.get("error") or audit.get("n_samples", 0) <= 0:
        raise RuntimeError(f"audit produced no samples: {audit.get('error')}")
    violations = sum(int(value) for value in (audit.get("collision_by_statepair") or {}).values())
    save(out / "status.json", {"status": "finished", "complete": True,
         "training_wall_sec": result.training_wall_sec, "makespan_sec": result.makespan_sec,
         "sampled_3d_pass": violations == 0 and int(audit["ik_fail_samples"]) == 0,
         "clearance_violation_samples": violations, "ik_fail_samples": audit["ik_fail_samples"],
         "note": "sampled pairwise model audit; C1 version alignment not yet verified"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--straight-mm", type=float, choices=(50.0, 25.0, 10.0), default=50.0)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(args, out)
    except BaseException as error:
        save(out / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        save(out / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
