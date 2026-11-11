# 논문식 자연 경로 배정·메타 이어하기·포인트 RAPID 왕복을 검증하는 테스트
from __future__ import annotations

import math

import numpy as np

from pipeline.S3_path_planning.S38_hrl_pointwise import (
    NaturalPath,
    PathSegment,
    PointSchedulingEnv,
    build_robot_point_queues,
)
from pipeline.S3_path_planning.S39_hrl_paper_training import (
    MetaPointSchedulingEnv,
    PathAllocationEnv,
    PointLayerTrainingResult,
    _choose_final_scheduling,
    _meta_selection_key,
    _restore_model_parameters,
    _snapshot_model_parameters,
    train_point_hrl_layer,
)
from pipeline.S3_path_planning.S40_hrl_pointwise_rapid import (
    generate_pointwise_rapid_per_robot,
)
from verify.rapid_timeline import events_from_mod_dir


def _path(path_id: int, start, end, bead_type: str = "i_f") -> NaturalPath:
    segment = PathSegment(
        start=tuple(start),
        end=tuple(end),
        bead_index=path_id + 1,
        bead_type=bead_type,
    )
    return NaturalPath(
        path_id=path_id,
        root_id=path_id,
        bead_indices=(path_id + 1,),
        bead_type=bead_type,
        segments=(segment,),
    )


def _beads(paths):
    rows = []
    for path in paths:
        start = np.asarray(path.start, float)
        end = np.asarray(path.end, float)
        delta = end - start
        length = float(np.linalg.norm(delta))
        direction = delta / length
        center = 0.5 * (start + end)
        rows.append({
            "index": path.path_id + 1,
            "type": path.bead_type,
            "x": float(center[0]),
            "y": float(center[1]),
            "dx": float(direction[0]),
            "dy": float(direction[1]),
            "length": length,
            "width": 0.0,
        })
    return rows


def _three_sector_paths():
    return [
        _path(0, (650.0, -20.0, 3.9), (650.0, 20.0, 3.9)),
        _path(1, (-342.3, 572.9, 3.9), (-307.7, 592.9, 3.9)),
        _path(2, (-307.7, -592.9, 3.9), (-342.3, -572.9, 3.9)),
    ]


def test_path_allocation_uses_actual_seconds_and_finishes_once():
    paths = _three_sector_paths()
    environment = PathAllocationEnv(paths, _beads(paths), seed=42)
    observation, _ = environment.reset(seed=42)
    assert observation.shape == (13,)
    terminated = False
    while not terminated:
        observation, reward, terminated, truncated, _ = environment.step(0)
        assert math.isfinite(reward)
        assert not truncated
    assert np.count_nonzero(environment.assignments == 0) == len(paths)
    assert np.all(environment.proc_times > 0.0)


def test_meta_environment_continues_unfinished_indices():
    paths = _three_sector_paths()
    queues = build_robot_point_queues(paths, [0, 1, 2], _beads(paths))
    environment = MetaPointSchedulingEnv(queues, meta_reset_probability=0.0)
    environment.horizon = 1
    environment.reset(seed=42)
    _obs, _reward, terminated, truncated, _info = environment.step(7)
    assert truncated and not terminated
    assert _info["termination_reason"] == "horizon"
    assert _info["continuation_feasible"] is True
    before = dict(environment.indices)
    environment.reset(seed=43)
    assert environment.indices == before
    assert environment.continue_resets >= 1


def test_meta_environment_full_resets_after_noncontinuable_truncation(monkeypatch):
    paths = _three_sector_paths()
    queues = build_robot_point_queues(paths, [0, 1, 2], _beads(paths))
    environment = MetaPointSchedulingEnv(queues, meta_reset_probability=0.0)
    environment.reset(seed=42)
    full_resets_before = environment.full_resets

    def truncate_as_deadlock(base_environment, _action):
        base_environment.indices[0] = 1
        observation = np.zeros(base_environment.observation_space.shape, dtype=np.float32)
        info = {
            "termination_reason": "deadlock",
            "continuation_feasible": False,
        }
        return observation, -1.0, False, True, info

    monkeypatch.setattr(PointSchedulingEnv, "step", truncate_as_deadlock)
    _obs, _reward, terminated, truncated, info = environment.step(0)
    assert truncated and not terminated
    assert info["termination_reason"] == "deadlock"
    assert info["continuation_feasible"] is False
    assert environment.last_state is None

    environment.reset(seed=43)
    assert environment.indices == {0: 0, 1: 0, 2: 0}
    assert environment.full_resets == full_resets_before + 1
    assert environment.continue_resets == 0


def test_pointwise_rapid_preserves_weld_length_and_returns_to_park(tmp_path):
    paths = _three_sector_paths()
    beads = _beads(paths)
    queues = build_robot_point_queues(paths, [0, 1, 2], beads)
    environment = PointSchedulingEnv(queues)
    environment.reset(seed=42)
    terminated = truncated = False
    for _ in range(1000):
        _obs, _reward, terminated, truncated, _info = environment.step(7)
        if terminated or truncated:
            break
    assert terminated and not truncated
    result = PointLayerTrainingResult(
        layer_idx=0,
        layer_z=3.9,
        paths=paths,
        assignment=[0, 1, 2],
        queues=queues,
        records=list(environment.records),
        step_log=list(environment.step_log),
        makespan_sec=environment.now_sec,
        complete=True,
        training_wall_sec=0.0,
        best_meta_episode=1,
    )
    summary = generate_pointwise_rapid_per_robot([result], tmp_path)
    events, _tasks, info = events_from_mod_dir(str(tmp_path), timing="accumulated")
    emitted_length = sum(float(np.linalg.norm(event.end_pt - event.start_pt)) for event in events)
    expected_length = sum(path.length_mm for path in paths)
    assert math.isclose(emitted_length, expected_length, abs_tol=1e-6)
    assert info["n_events"] >= 3
    assert all(summary["robots"][str(robot)]["arc_runs"] >= 1 for robot in range(3))
    assert all(
        math.isclose(
            summary["robots"][str(robot)]["weld_length_mm"],
            paths[robot].length_mm,
            abs_tol=1e-6,
        )
        for robot in range(3)
    )


def test_tiny_joint_training_completes_rollouts_after_minimum_episode_count():
    paths = _three_sector_paths()
    result = train_point_hrl_layer(
        paths,
        _beads(paths),
        layer_idx=0,
        layer_z=3.9,
        seed=42,
        meta_episodes=1,
        allocation_episodes=1,
        scheduling_episodes=1,
    )
    assert len(result.meta_metrics) == 1
    assert result.evaluation_stats["allocation_episodes_completed"] >= 1
    assert result.evaluation_stats["scheduling_episodes_completed"] >= 1
    assert result.evaluation_stats["best_scheduler_parameters_restored"] is True
    assert set(result.evaluation_stats["best_meta_scheduling_evaluation"]) == {
        "complete",
        "reward",
        "progress",
        "makespan_sec",
        "blocked_attempts",
        "total_wait_steps",
        "decision_steps",
        "truncated",
    }
    assert result.evaluation_stats["final_evaluation_source"] in {
        "restored_scheduler_deterministic",
        "best_meta_observed_rollout",
    }


def _scheduling_result(*, complete, progress, makespan, reward=0.0):
    return {
        "completed": complete,
        "progress": progress,
        "makespan_sec": makespan,
        "reward": reward,
        "collisions": 0,
        "total_wait_steps": 0,
        "decision_steps": 1,
        "truncated": not complete,
        "records": [],
        "step_log": [],
    }


def test_meta_selection_is_lexicographic_completion_time_progress_then_reward():
    incomplete_high_reward = _scheduling_result(
        complete=False, progress=0.99, makespan=1.0, reward=1000.0,
    )
    complete_low_reward = _scheduling_result(
        complete=True, progress=1.0, makespan=100.0, reward=-1000.0,
    )
    assert _meta_selection_key(complete_low_reward, -1000.0) > _meta_selection_key(
        incomplete_high_reward, 1000.0,
    )

    complete_short = _scheduling_result(
        complete=True, progress=1.0, makespan=80.0,
    )
    complete_long = _scheduling_result(
        complete=True, progress=1.0, makespan=90.0,
    )
    assert _meta_selection_key(complete_short, -100.0) > _meta_selection_key(
        complete_long, 100.0,
    )
    assert _meta_selection_key(complete_short, 2.0) > _meta_selection_key(
        complete_short, 1.0,
    )

    incomplete_far = _scheduling_result(
        complete=False, progress=0.8, makespan=500.0,
    )
    incomplete_near = _scheduling_result(
        complete=False, progress=0.9, makespan=900.0,
    )
    assert _meta_selection_key(incomplete_near, -100.0) > _meta_selection_key(
        incomplete_far, 100.0,
    )


def test_scheduler_parameter_snapshot_is_deep_and_restore_is_exact():
    class FakeModel:
        def __init__(self):
            self.parameters = {"policy": {"weights": [1, 2, 3]}}
            self.restored = None

        def get_parameters(self):
            return self.parameters

        def set_parameters(self, parameters, *, exact_match):
            self.restored = (parameters, exact_match)

    model = FakeModel()
    snapshot = _snapshot_model_parameters(model)
    model.parameters["policy"]["weights"][0] = 99
    assert snapshot["policy"]["weights"] == [1, 2, 3]

    _restore_model_parameters(model, snapshot)
    restored, exact_match = model.restored
    assert exact_match is True
    assert restored == snapshot
    restored["policy"]["weights"][1] = 88
    assert snapshot["policy"]["weights"] == [1, 2, 3]


def test_final_gate_reuses_better_observed_rollout_and_prefers_equal_deterministic():
    observed = _scheduling_result(
        complete=True, progress=1.0, makespan=80.0,
    )
    deterministic = _scheduling_result(
        complete=False, progress=0.95, makespan=70.0,
    )
    selected, source = _choose_final_scheduling(deterministic, observed)
    assert selected["completed"] is True
    assert source == "best_meta_observed_rollout"

    equal = _scheduling_result(
        complete=True, progress=1.0, makespan=80.0,
    )
    selected, source = _choose_final_scheduling(equal, observed)
    assert selected is equal
    assert source == "restored_scheduler_deterministic"
