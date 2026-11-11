# 현재 형상용 계층적 강화학습 어댑터의 핵심 계약을 검증하는 테스트
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipeline.S3_path_planning.S31_types import ROBOT_BASES
from pipeline.S3_path_planning.S33_rapid_writer import _task_arc_segments, scheduled_tasks
from pipeline.S3_path_planning.S36_hrl_layerwise import (
    HrlTask,
    LayerHrlResult,
    TaskAllocationEnv,
    TaskSchedulingEnv,
    build_layered_schedule_result,
    prepare_v4_layers,
)


def _task(idx: int, *, layer: int = 0, duration: float = 10.0,
          predecessor=None, eligible=(True, True, True)) -> HrlTask:
    center = np.array([float(idx * 100), float((idx % 2) * 50)])
    return HrlTask(
        layer_idx=layer,
        task_type="infill",
        task_id=idx,
        cluster_id=idx,
        centroid=center,
        bbox_min=center - 5.0,
        bbox_max=center + 5.0,
        bead_indices=(idx + 1,),
        duration_by_robot=(duration, duration + 1.0, duration + 2.0),
        eligible_robots=eligible,
        predecessor=predecessor,
    )


def _analysis(layer_idx: int, tasks):
    clusters = [SimpleNamespace(id=t.cluster_id, bead_indices=list(t.bead_indices))
                for t in tasks]
    return SimpleNamespace(layer_idx=layer_idx, clusters=clusters, shell_tasks=[])


def test_current_robot_bases_and_allocation_mask_are_used():
    np.testing.assert_allclose(
        ROBOT_BASES,
        np.array([[1500.0, 0.0], [-750.0, 1299.03810568],
                  [-750.0, -1299.03810568]]),
        atol=1e-6,
    )
    env = TaskAllocationEnv([_task(0, eligible=(False, True, False))],
                            shuffle_tasks=False)
    obs, _ = env.reset(seed=42)
    assert obs.shape == (13,)
    assert env.action_masks().tolist() == [False, True, False]
    _obs, _reward, terminated, truncated, info = env.step(1)
    assert terminated and not truncated
    assert info["assignment"] == [1]


def test_scheduler_blocks_conflicting_simultaneous_start():
    tasks = [_task(0, duration=10.0), _task(1, duration=10.0)]
    conflicts = {("infill", 0, 0, "infill", 1, 1): True}
    env = TaskSchedulingEnv(tasks, [0, 1], conflicts)
    env.reset(seed=42)
    _obs, _reward, terminated, truncated, info = env.step(0b011)
    assert not terminated and not truncated
    assert info["started"] == 1
    assert info["blocked"] == 1
    _obs, _reward, terminated, truncated, _info = env.step(0b010)
    assert terminated and not truncated
    assert len(env.records) == 2
    assert env.records[1]["start_sec"] >= env.records[0]["end_sec"]


def test_scheduler_is_reproducible_for_fixed_actions():
    tasks = [_task(0), _task(1), _task(2)]

    def run():
        env = TaskSchedulingEnv(tasks, [0, 1, 2])
        env.reset(seed=42)
        env.step(0b111)
        while env.active:
            env.step(0)
        return env.records, env.now_sec

    records_a, makespan_a = run()
    records_b, makespan_b = run()
    assert records_a == records_b
    assert makespan_a == pytest.approx(makespan_b)


def test_layer_barrier_and_exactly_once_contract():
    tasks0 = [_task(0, layer=0, duration=10.0), _task(1, layer=0, duration=20.0)]
    tasks1 = [_task(0, layer=1, duration=5.0)]
    result0 = LayerHrlResult(
        0, tasks0, [0, 1],
        [{"task_index": 0, "robot": 0, "start_sec": 0.0, "end_sec": 10.0},
         {"task_index": 1, "robot": 1, "start_sec": 0.0, "end_sec": 21.0}],
        21.0, True,
    )
    result1 = LayerHrlResult(
        1, tasks1, [2],
        [{"task_index": 0, "robot": 2, "start_sec": 0.0, "end_sec": 7.0}],
        7.0, True,
    )
    beads = [[{"index": 1}], [{"index": 1}]]
    result = build_layered_schedule_result(
        ["z3.9", "z7.3"], beads,
        [_analysis(0, tasks0), _analysis(1, tasks1)], [result0, result1])
    assert len(result["all_entries"]) == 3
    layer0_end = max(e.end for e in result["all_entries"] if e.layer_idx == 0)
    layer1_start = min(e.start for e in result["all_entries"] if e.layer_idx == 1)
    assert layer1_start >= layer0_end
    assert result["config"]["layer_barrier"] is True


def test_duplicate_task_record_is_rejected():
    tasks = [_task(0)]
    duplicate = LayerHrlResult(
        0, tasks, [0],
        [{"task_index": 0, "robot": 0, "start_sec": 0.0, "end_sec": 10.0},
         {"task_index": 0, "robot": 0, "start_sec": 10.0, "end_sec": 20.0}],
        20.0, True,
    )
    with pytest.raises(ValueError, match="scheduled twice"):
        build_layered_schedule_result(
            ["z3.9"], [[{"index": 1}]], [_analysis(0, tasks)], [duplicate])


def test_current_writer_preserves_horizontal_vertical_and_diagonal_directions():
    beads = [{
        "index": 1, "type": "i_f", "x": 0.0, "y": 0.0,
        "dx": 1.0, "dy": 0.0, "kappa": 0.0, "length": 20.0, "width": 8.0,
    }, {
        "index": 2, "type": "i_f", "x": 100.0, "y": 0.0,
        "dx": 0.0, "dy": 1.0, "kappa": 0.0, "length": 20.0, "width": 8.0,
    }, {
        "index": 3, "type": "i_f", "x": 200.0, "y": 0.0,
        "dx": 2 ** -0.5, "dy": 2 ** -0.5, "kappa": 0.0,
        "length": 20.0, "width": 8.0,
    }]
    analysis = SimpleNamespace(
        layer_idx=0,
        clusters=[SimpleNamespace(id=0, bead_indices=[1, 2, 3])],
        shell_tasks=[],
    )
    entry = SimpleNamespace(
        layer_idx=0, cluster_id=0, robot=0, start=0, duration_sec=3.0,
        task_type="infill", shell_id=None,
    )
    result = {
        "layer_names": ["z3.9"], "all_beads": [beads],
        "all_analyses": [analysis], "all_entries": [entry],
    }
    scheduled = scheduled_tasks(result)
    segments = _task_arc_segments(scheduled[0])
    vectors = [np.asarray(points[-1])[:2] - np.asarray(points[0])[:2]
               for _kind, points in segments]
    assert any(abs(v[0]) > 0 and abs(v[1]) < 1e-9 for v in vectors)
    assert any(abs(v[1]) > 0 and abs(v[0]) < 1e-9 for v in vectors)
    assert any(abs(v[0]) > 0 and abs(v[1]) > 0 for v in vectors)


def test_v4_csv_uses_current_schema_and_builds_phase0_tasks():
    repo = Path(__file__).resolve().parents[3]
    v4 = (repo / "docs" / "experiments" / "2026-08-08_poster-contracts"
          / "beads" / "V4")
    names, beads, analyses, _offset, _rotation = prepare_v4_layers(
        v4, limit=1, with_conflict=False)
    assert names == ["z3.9"]
    assert len(beads[0]) == 280
    assert len(analyses[0].clusters) + len(analyses[0].shell_tasks) == 8
    assert all({"type", "x", "y", "dx", "dy", "kappa", "length", "width"}
               <= set(bead) for bead in beads[0])
    assert any(abs(bead["dx"]) > 1e-3 for bead in beads[0])


def test_first_eleven_v4_layers_each_build_exactly_eight_tasks():
    repo = Path(__file__).resolve().parents[3]
    v4 = (repo / "docs" / "experiments" / "2026-08-08_poster-contracts"
          / "beads" / "V4")
    names, _beads, analyses, _offset, _rotation = prepare_v4_layers(
        v4, limit=11, with_conflict=False)
    assert len(names) == 11
    assert [len(analysis.clusters) + len(analysis.shell_tasks)
            for analysis in analyses] == [8] * 11
