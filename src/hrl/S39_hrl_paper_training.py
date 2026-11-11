# 논문식 상위 배정과 하위 이동·대기 PPO를 V4 자연 경로에 결합해 학습하는 모듈
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from config import DEFAULT_CONFIG, KitechConfig
from pipeline.S3_path_planning.S31_types import N_ROBOTS, ROBOT_BASES, classify_bead
from pipeline.S3_path_planning.S38_hrl_pointwise import (
    NaturalPath,
    PointSchedulingEnv,
    RobotPointQueue,
    build_robot_point_queues,
)


ALPHA_CV = 0.3
BETA_COMP = 0.3
GAMMA_SPLIT = 0.2
DELTA_PROX = 0.4
LAMBDA_BALANCE = 0.5
LAMBDA_ADJ = 1.2
LAMBDA_SPLIT = 0.5
LAMBDA_PROX = 0.6
KNN_K = 5
K_ALLOW = 3
EPS_SCALE = 0.8
META_RESET_PROB = 0.1
MAX_TIMESTEPS_PER_LEARN = 10_000_000
META_SELECTION_CRITERION = {
    "mode": "lexicographic max",
    "priority_1": "completed descending",
    "priority_2_completed": "makespan_sec ascending",
    "priority_2_incomplete": "progress descending",
    "priority_3": "joint_reward descending",
}


@dataclass
class PointLayerTrainingResult:
    """한 물리 층의 마지막 정책과 최상 배정을 결합한 최종 평가 결과."""

    layer_idx: int
    layer_z: float
    paths: List[NaturalPath]
    assignment: List[int]
    queues: Tuple[RobotPointQueue, ...]
    records: List[dict]
    step_log: List[dict]
    makespan_sec: float
    complete: bool
    training_wall_sec: float
    best_meta_episode: int
    meta_metrics: List[dict] = field(default_factory=list)
    reward_history: dict = field(default_factory=dict)
    evaluation_stats: dict = field(default_factory=dict)


def set_global_seed(seed: int) -> None:
    """원본 코드와 같은 Python·NumPy·PyTorch 난수 초기화를 수행한다."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _coefficient_of_variation(loads: np.ndarray) -> float:
    mean = float(np.mean(loads))
    return 0.0 if mean <= 1e-9 else float(np.std(loads) / mean)


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 0), dtype=float)
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def _connected_components(points: np.ndarray, eps: float) -> np.ndarray:
    count = len(points)
    if count <= 1:
        return np.zeros(count, dtype=int)
    distances = _pairwise_distances(points)
    parent = np.arange(count)
    rank = np.zeros(count, dtype=int)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first == root_second:
            return
        if rank[root_first] < rank[root_second]:
            parent[root_first] = root_second
        elif rank[root_first] > rank[root_second]:
            parent[root_second] = root_first
        else:
            parent[root_second] = root_first
            rank[root_first] += 1

    for first in range(count):
        for second in range(first + 1, count):
            if distances[first, second] <= eps:
                union(first, second)
    roots = sorted({find(index) for index in range(count)})
    labels = {root: label for label, root in enumerate(roots)}
    return np.asarray([labels[find(index)] for index in range(count)], dtype=int)


def _knn_spread(points: np.ndarray, k: int, distance_reference: float) -> float:
    count = len(points)
    if count <= 1:
        return 0.0
    distances = _pairwise_distances(points)
    np.fill_diagonal(distances, np.inf)
    effective_k = min(k, count - 1)
    nearest = np.partition(
        distances,
        kth=effective_k - 1,
        axis=1,
    )[:, :effective_k]
    return float(np.mean(np.mean(nearest, axis=1)) / max(distance_reference, 1e-9))


def _cluster_compactness(
    points: np.ndarray,
    *,
    k: int,
    distance_reference: float,
    eps: float,
) -> Tuple[float, int]:
    count = len(points)
    if count == 0:
        return 0.0, 0
    labels = _connected_components(points, eps)
    component_count = int(labels.max() + 1)
    compactness = 0.0
    for label in range(component_count):
        members = points[np.where(labels == label)[0]]
        spread = _knn_spread(members, k, distance_reference)
        compactness += (len(members) / count) * spread
    return float(compactness), component_count


def _path_centroid(path: NaturalPath) -> np.ndarray:
    weighted = np.zeros(2, dtype=float)
    total = 0.0
    for segment in path.segments:
        length = segment.length_mm
        weighted += 0.5 * (
            np.asarray(segment.start[:2], float)
            + np.asarray(segment.end[:2], float)
        ) * length
        total += length
    if total <= 1e-9:
        return np.asarray(path.start[:2], float)
    return weighted / total


class PathAllocationEnv(gym.Env):
    """원본 13차원 배정 MDP를 실제 방향을 가진 자연 경로에 적용한다."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        paths: Sequence[NaturalPath],
        beads: Sequence[dict],
        *,
        config: KitechConfig = DEFAULT_CONFIG,
        seed: int = 42,
    ):
        super().__init__()
        if N_ROBOTS != 3:
            raise ValueError("paper allocation environment requires three robots")
        if not paths:
            raise ValueError("PathAllocationEnv requires at least one natural path")
        self.paths = list(paths)
        self.config = config
        self.rng = np.random.RandomState(seed)
        bead_by_index = {int(bead["index"]): bead for bead in beads}
        self.coords = np.asarray([_path_centroid(path) for path in self.paths], float)
        self.lengths = np.asarray([path.length_mm for path in self.paths], float)
        self.widths = np.asarray([
            float(np.average(
                [float(bead_by_index[index].get("width", 0.0)) for index in path.bead_indices],
                weights=[float(bead_by_index[index]["length"]) for index in path.bead_indices],
            ))
            for path in self.paths
        ], dtype=float)
        self.proc_times = np.asarray([
            path.length_mm / (
                float(config.weaving.weld_speed_weave)
                if classify_bead(path.bead_type) == "WALL"
                else float(config.weaving.weld_speed_normal)
            )
            for path in self.paths
        ], dtype=float)
        self.starts = np.asarray([path.start[:2] for path in self.paths], float)
        self.ends = np.asarray([path.end[:2] for path in self.paths], float)
        self.n_paths = len(self.paths)
        self.x_min, self.x_max = self.coords[:, 0].min(), self.coords[:, 0].max()
        self.y_min, self.y_max = self.coords[:, 1].min(), self.coords[:, 1].max()
        self.len_min, self.len_max = self.lengths.min(), self.lengths.max()
        self.width_min, self.width_max = self.widths.min(), self.widths.max()
        self.distance_reference = self._compute_distance_reference()
        self.action_space = spaces.Discrete(3)
        # 부하 비율과 기준거리 대비 거리는 1을 넘을 수 있다.
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )
        self.order = np.arange(self.n_paths, dtype=int)
        self.external_bonus = 0.0
        self.reset()

    def _compute_distance_reference(self) -> float:
        if self.n_paths <= 1:
            return 1.0
        indices = self.rng.choice(
            self.n_paths,
            size=min(200, self.n_paths),
            replace=False,
        )
        sample = self.coords[indices]
        distances = []
        for index in range(len(sample) - 1):
            distances.extend(
                np.linalg.norm(sample[index + 1:] - sample[index], axis=1).tolist()
            )
        return float(np.median(distances)) if distances else 1.0

    @staticmethod
    def _normalize(value: float, lower: float, upper: float) -> float:
        return 0.0 if upper - lower <= 1e-9 else float((value - lower) / (upper - lower))

    @property
    def current_action_mask(self) -> np.ndarray:
        # 원본 코드와 현재 기본 셀 계약 모두 세 로봇을 허용한다.
        return np.ones(3, dtype=bool)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.t = 0
        self.assignments = -np.ones(self.n_paths, dtype=int)
        self.robot_sets: List[List[int]] = [[], [], []]
        self.robot_loads = np.zeros(3, dtype=float)
        self.order = np.arange(self.n_paths, dtype=int)
        self.rng.shuffle(self.order)
        return self._observation(), {}

    def _cv(self) -> float:
        return _coefficient_of_variation(self.robot_loads)

    def _compactness_and_split(self) -> Tuple[float, float]:
        compactness_values, split_values = [], []
        eps = self.distance_reference * EPS_SCALE
        for robot in range(3):
            indices = self.robot_sets[robot]
            points = self.coords[indices] if indices else np.zeros((0, 2), dtype=float)
            compactness, components = _cluster_compactness(
                points,
                k=KNN_K,
                distance_reference=self.distance_reference,
                eps=eps,
            )
            compactness_values.append(compactness)
            split_values.append(max(0, components - K_ALLOW))
        return float(np.mean(compactness_values)), float(np.mean(split_values))

    def _proximity(self) -> float:
        assigned = np.where(self.assignments >= 0)[0]
        if assigned.size == 0:
            return 0.0
        distances, weights = [], []
        for path_index in assigned:
            robot = int(self.assignments[path_index])
            base = np.asarray(ROBOT_BASES[robot], float)
            distance = 0.5 * (
                np.linalg.norm(self.starts[path_index] - base)
                + np.linalg.norm(self.ends[path_index] - base)
            )
            distances.append(float(distance))
            weights.append(float(self.lengths[path_index]))
        return float(np.average(distances, weights=np.maximum(weights, 1e-9))) / max(
            self.distance_reference,
            1e-6,
        )

    def step(self, action: int):
        cv_before = self._cv()
        compactness_before, split_before = self._compactness_and_split()
        proximity_before = self._proximity()

        path_index = int(self.order[self.t])
        robot = int(action)
        self.assignments[path_index] = robot
        self.robot_sets[robot].append(path_index)
        self.robot_loads[robot] += self.proc_times[path_index]

        cv_after = self._cv()
        compactness_after, split_after = self._compactness_and_split()
        proximity_after = self._proximity()
        reward = -(
            ALPHA_CV * (cv_after - cv_before)
            + BETA_COMP * (compactness_after - compactness_before)
            + GAMMA_SPLIT * (split_after - split_before)
            + DELTA_PROX * (proximity_after - proximity_before)
        )
        self.t += 1
        terminated = self.t >= self.n_paths
        if terminated:
            reward += -(
                LAMBDA_BALANCE * cv_after
                + LAMBDA_ADJ * compactness_after
                + LAMBDA_SPLIT * split_after
                + LAMBDA_PROX * proximity_after
            )
            reward += float(self.external_bonus)
        observation = (
            np.zeros(13, dtype=np.float32)
            if terminated
            else self._observation()
        )
        info = {
            "allocation": tuple(len(values) for values in self.robot_sets),
            "robot_loads_sec": self.robot_loads.tolist(),
        }
        return observation, float(reward), terminated, False, info

    def _observation(self) -> np.ndarray:
        if self.t >= self.n_paths:
            return np.zeros(13, dtype=np.float32)
        index = int(self.order[self.t])
        loads = self.robot_loads / max(float(np.mean(self.robot_loads)) + 1e-6, 1.0)
        endpoint_distances = []
        for robot in range(3):
            base = np.asarray(ROBOT_BASES[robot], float)
            endpoint_distances.append(0.5 * (
                np.linalg.norm(self.starts[index] - base)
                + np.linalg.norm(self.ends[index] - base)
            ) / max(self.distance_reference, 1e-6))
        remaining = self.order[self.t:]
        return np.asarray([
            self._normalize(self.coords[index, 0], self.x_min, self.x_max),
            self._normalize(self.coords[index, 1], self.y_min, self.y_max),
            self._normalize(self.lengths[index], self.len_min, self.len_max),
            self._normalize(self.widths[index], self.width_min, self.width_max),
            self.t / max(self.n_paths - 1, 1),
            loads[0], loads[1], loads[2],
            endpoint_distances[0], endpoint_distances[1], endpoint_distances[2],
            (self.n_paths - self.t) / self.n_paths,
            float(self.lengths[remaining].sum() / max(self.lengths.sum(), 1e-6)),
        ], dtype=np.float32)


class MetaPointSchedulingEnv(PointSchedulingEnv):
    """원본의 10% 전체 초기화·90% 미완료 상태 이어하기를 보존한다."""

    def __init__(
        self,
        queues: Sequence[RobotPointQueue],
        *,
        meta_reset_probability: float = META_RESET_PROB,
    ):
        self.meta_reset_probability = float(meta_reset_probability)
        self.is_first_episode = True
        self.last_state: Optional[dict] = None
        self.full_resets = 0
        self.continue_resets = 0
        super().__init__(queues)

    def _full_reset(self) -> None:
        self.indices = {robot: 0 for robot in range(3)}
        self.done_robot = {
            robot: len(self.queues[robot].edges) == 0 for robot in range(3)
        }
        self.gave_bonus = {robot: False for robot in range(3)}
        self.wait_steps = np.zeros(3, dtype=np.int64)
        self.collision_steps = np.zeros(3, dtype=np.int64)
        self.full_resets += 1

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        gym.Env.reset(self, seed=seed)
        should_reset = (
            self.is_first_episode
            or self.last_state is None
            or all(self.last_state["done_robot"].values())
            or np.random.random() < self.meta_reset_probability
        )
        if should_reset:
            self._full_reset()
        else:
            self.indices = dict(self.last_state["indices"])
            self.done_robot = dict(self.last_state["done_robot"])
            self.gave_bonus = dict(self.last_state["gave_bonus"])
            self.wait_steps = self.last_state["wait_steps"].copy()
            self.collision_steps = self.last_state["collision_steps"].copy()
            self.continue_resets += 1
        self.is_first_episode = False
        self.t = 0
        self.now_sec = 0.0
        self.collision_count = 0
        self.no_progress_counter = 0
        self.records = []
        self.step_log = []
        self.traj_indices = [[self.indices[robot] for robot in range(3)]]
        return self._build_obs(), {}

    def step(self, action: int):
        result = super().step(action)
        _observation, _reward, terminated, truncated, info = result
        termination_reason = info.get("termination_reason")
        continuation_feasible = bool(info.get("continuation_feasible", True))
        requires_full_reset = truncated and (
            not continuation_feasible
            or termination_reason in {"deadlock", "no_valid_action"}
        )
        if requires_full_reset:
            self.last_state = None
        elif terminated or truncated:
            self.last_state = {
                "indices": dict(self.indices),
                "done_robot": dict(self.done_robot),
                "gave_bonus": dict(self.gave_bonus),
                "wait_steps": self.wait_steps.copy(),
                "collision_steps": self.collision_steps.copy(),
            }
        return result


def _callback_class():
    from stable_baselines3.common.callbacks import BaseCallback

    class EpisodeCountCallback(BaseCallback):
        def __init__(self, target_episodes: int):
            super().__init__(verbose=0)
            self.target_episodes = int(target_episodes)
            self.episodes_this_meta = 0
            self.current_episode_reward = 0.0
            self.episode_rewards: List[float] = []
            self.episode_timesteps: List[int] = []

        def _on_step(self) -> bool:
            reward = self.locals.get("rewards")
            if reward is not None:
                self.current_episode_reward += float(np.asarray(reward).reshape(-1)[0])
            done = self.locals.get("dones")
            if done is not None and bool(np.asarray(done).reshape(-1)[0]):
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_timesteps.append(int(self.num_timesteps))
                self.current_episode_reward = 0.0
                self.episodes_this_meta += 1
            return True

        def reset_meta_counter(self) -> None:
            self.episodes_this_meta = 0
            self.current_episode_reward = 0.0

    return EpisodeCountCallback


def _learn_episode_budget(model, callback, progress_callback=None) -> None:
    """최소 에피소드 수를 채우고 마지막 rollout까지 PPO 갱신을 완료한다."""
    callback.reset_meta_counter()
    started = int(model.num_timesteps)
    while callback.episodes_this_meta < callback.target_episodes:
        if int(model.num_timesteps) - started >= MAX_TIMESTEPS_PER_LEARN:
            raise RuntimeError("episode budget was not reached within the timestep limit")
        model.learn(
            total_timesteps=model.n_steps,
            reset_num_timesteps=False,
            callback=callback,
        )
        if progress_callback is not None:
            progress_callback({"episodes_this_meta": callback.episodes_this_meta,
                               "timesteps": int(model.num_timesteps),
                               "ppo_updates": int(model._n_updates)})


def run_allocation(model, environment: PathAllocationEnv) -> dict:
    """원본과 같이 확률적 정책으로 배정 한 에피소드를 평가한다."""

    observation, _ = environment.reset()
    terminated = False
    reward_sum = 0.0
    while not terminated:
        try:
            action, _ = model.predict(
                observation,
                action_masks=environment.current_action_mask,
                deterministic=False,
            )
        except TypeError:
            action, _ = model.predict(observation, deterministic=False)
        observation, reward, terminated, _truncated, _info = environment.step(
            int(np.asarray(action).item())
        )
        reward_sum += float(reward)
    return {
        "assignment": [int(value) for value in environment.assignments],
        "robot_sets": [list(values) for values in environment.robot_sets],
        "reward": reward_sum,
    }


def run_scheduling(
    model,
    queues: Sequence[RobotPointQueue],
    *,
    max_steps: int = 5000,
    deterministic: bool = False,
    environment_factory=PointSchedulingEnv,
) -> dict:
    """정책을 새 전체 초기 상태에서 평가한다."""

    environment = environment_factory(queues)
    observation, _ = environment.reset()
    terminated = truncated = False
    reward_sum = 0.0
    steps = 0
    while not terminated and not truncated and steps < max_steps:
        action, _ = model.predict(observation, deterministic=deterministic)
        observation, reward, terminated, truncated, _info = environment.step(
            int(np.asarray(action).item())
        )
        reward_sum += float(reward)
        steps += 1
    total_edges = sum(len(queue.edges) for queue in queues)
    remaining = environment._total_remaining()
    return {
        "reward": reward_sum,
        "completed": bool(terminated and remaining == 0),
        "progress": (total_edges - remaining) / max(total_edges, 1),
        "collisions": int(environment.collision_count),
        "total_wait_steps": int(environment.wait_steps.sum()),
        "records": list(environment.records),
        "step_log": list(environment.step_log),
        "makespan_sec": float(environment.now_sec),
        "decision_steps": int(steps),
        "truncated": bool(truncated or (steps >= max_steps and not terminated)),
    }


def _scheduling_quality_key(result: dict) -> tuple:
    """완주 우선이며 완주 시 짧은 시간, 미완주 시 높은 진행률을 고른다."""

    if bool(result["completed"]):
        return 1, -float(result["makespan_sec"])
    return 0, float(result["progress"])


def _meta_selection_key(result: dict, joint_reward: float) -> tuple:
    """논문식 결합 보상은 완주·시간·진행률이 같은 마지막 동률에만 쓴다."""

    return (*_scheduling_quality_key(result), float(joint_reward))


def _snapshot_model_parameters(model) -> dict:
    """후속 PPO 학습이 best 시점 파라미터를 바꾸지 못하도록 깊은 사본을 만든다."""

    return copy.deepcopy(model.get_parameters())


def _restore_model_parameters(model, parameters: dict) -> None:
    """best 시점의 scheduler 파라미터를 정확히 복원한다."""

    model.set_parameters(copy.deepcopy(parameters), exact_match=True)


def _scheduling_evaluation_stats(result: dict) -> dict:
    return {
        "complete": bool(result["completed"]),
        "reward": float(result["reward"]),
        "progress": float(result["progress"]),
        "makespan_sec": float(result["makespan_sec"]),
        "blocked_attempts": int(result["collisions"]),
        "total_wait_steps": int(result["total_wait_steps"]),
        "decision_steps": int(result["decision_steps"]),
        "truncated": bool(result["truncated"]),
    }


def _choose_final_scheduling(
    deterministic_result: dict,
    best_observed_result: dict,
) -> tuple[dict, str]:
    """복원 정책의 결정론적 결과가 나쁠 때만 이미 관측한 best rollout을 쓴다."""

    if _scheduling_quality_key(best_observed_result) > _scheduling_quality_key(
        deterministic_result
    ):
        return copy.deepcopy(best_observed_result), "best_meta_observed_rollout"
    return deterministic_result, "restored_scheduler_deterministic"


def train_point_hrl_layer(
    paths: Sequence[NaturalPath],
    beads: Sequence[dict],
    *,
    layer_idx: int,
    layer_z: float,
    config: KitechConfig = DEFAULT_CONFIG,
    seed: int = 42,
    meta_episodes: int = 100,
    allocation_episodes: int = 100,
    scheduling_episodes: int = 200,
    model_dir: Optional[str | Path] = None,
    device: str = "cpu",
    progress_callback=None,
    queue_builder=None,
    checkpoint_callback=None,
    scheduling_environment_factory=MetaPointSchedulingEnv,
    evaluation_environment_factory=PointSchedulingEnv,
    rollout_callback=None,
) -> PointLayerTrainingResult:
    """원본 episode 수·결합 순서·확률적 평가로 한 층을 독립 학습한다."""

    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    if min(meta_episodes, allocation_episodes, scheduling_episodes) <= 0:
        raise ValueError("all training episode counts must be positive")
    set_global_seed(seed)
    make_queues = queue_builder or build_robot_point_queues
    path_list = list(paths)
    allocation_base = PathAllocationEnv(path_list, beads, config=config, seed=seed)
    allocation_wrapper = ActionMasker(
        allocation_base,
        lambda env: env.current_action_mask,
    )
    allocation_model = MaskablePPO(
        "MlpPolicy",
        Monitor(allocation_wrapper),
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        device=device,
    )

    initial_environment = PathAllocationEnv(path_list, beads, config=config, seed=seed)
    initial_allocation = run_allocation(allocation_model, initial_environment)
    initial_queues = make_queues(
        path_list,
        initial_allocation["assignment"],
        beads,
        config=config,
    )
    scheduler_core = scheduling_environment_factory(initial_queues)
    scheduler_model = PPO(
        "MlpPolicy",
        Monitor(scheduler_core),
        verbose=0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        seed=seed,
        clip_range=0.2,
        n_epochs=10,
        device=device,
    )

    Callback = _callback_class()
    allocation_callback = Callback(allocation_episodes)
    scheduling_callback = Callback(scheduling_episodes)
    meta_metrics: List[dict] = []
    allocation_rewards: List[float] = []
    scheduling_rewards: List[float] = []
    joint_rewards: List[float] = []
    scheduling_bonuses: List[float] = []
    allocation_boundaries: List[int] = []
    scheduling_boundaries: List[int] = []
    best_selection_key: Optional[tuple] = None
    best: Optional[dict] = None
    started_at = time.perf_counter()

    for meta_index in range(meta_episodes):
        evaluation_environment = PathAllocationEnv(
            path_list,
            beads,
            config=config,
            seed=seed + meta_index,
        )
        allocation_result = run_allocation(allocation_model, evaluation_environment)
        queues = make_queues(
            path_list,
            allocation_result["assignment"],
            beads,
            config=config,
        )

        scheduler_core = scheduling_environment_factory(queues)
        scheduler_model.set_env(Monitor(scheduler_core))
        _learn_episode_budget(scheduler_model, scheduling_callback,
                              None if rollout_callback is None else lambda row: rollout_callback(
                                  {"meta_episode": meta_index + 1, "phase": "scheduling", **row}))
        scheduling_boundaries.append(
            scheduling_callback.episode_timesteps[-1]
            if scheduling_callback.episode_timesteps
            else 0
        )
        scheduling_result = run_scheduling(
            scheduler_model, queues, max_steps=5000,
            environment_factory=evaluation_environment_factory,
        )
        if scheduling_result["completed"]:
            scheduling_bonus = 40.0
        else:
            scheduling_bonus = 40.0 * (scheduling_result["progress"] - 0.5)

        allocation_base.external_bonus = scheduling_bonus
        allocation_parameters = (
            _snapshot_model_parameters(allocation_model)
            if checkpoint_callback is not None else None
        )
        _learn_episode_budget(allocation_model, allocation_callback,
                              None if rollout_callback is None else lambda row: rollout_callback(
                                  {"meta_episode": meta_index + 1, "phase": "allocation", **row}))
        allocation_boundaries.append(
            allocation_callback.episode_timesteps[-1]
            if allocation_callback.episode_timesteps
            else 0
        )

        allocation_reward = float(allocation_result["reward"])
        scheduling_reward = float(scheduling_result["reward"])
        joint_reward = allocation_reward + scheduling_reward + scheduling_bonus
        allocation_rewards.append(allocation_reward)
        scheduling_rewards.append(scheduling_reward)
        scheduling_bonuses.append(scheduling_bonus)
        joint_rewards.append(joint_reward)
        row = {
            "meta_episode": meta_index + 1,
            "allocation_reward": allocation_reward,
            "scheduling_reward": scheduling_reward,
            "scheduling_bonus": scheduling_bonus,
            "joint_reward": joint_reward,
            "complete": bool(scheduling_result["completed"]),
            "progress": float(scheduling_result["progress"]),
            "makespan_sec": float(scheduling_result["makespan_sec"]),
            "blocked_attempts": int(scheduling_result["collisions"]),
            "total_wait_steps": int(scheduling_result["total_wait_steps"]),
            "assignment_counts": [len(values) for values in allocation_result["robot_sets"]],
            "allocation_episodes_this_meta": allocation_callback.episodes_this_meta,
            "scheduling_episodes_this_meta": scheduling_callback.episodes_this_meta,
            "allocation_timesteps": int(allocation_model.num_timesteps),
            "scheduling_timesteps": int(scheduler_model.num_timesteps),
            "allocation_ppo_updates": int(allocation_model._n_updates),
            "scheduling_ppo_updates": int(scheduler_model._n_updates),
            "training_wall_sec": time.perf_counter() - started_at,
        }
        selection_key = _meta_selection_key(scheduling_result, joint_reward)
        row["selection_key"] = list(selection_key)
        meta_metrics.append(row)
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best = {
                "meta_episode": meta_index + 1,
                "assignment": list(allocation_result["assignment"]),
                "queues": queues,
                "selection_key": selection_key,
                "joint_reward": joint_reward,
                "scheduling_result": copy.deepcopy(scheduling_result),
                "scheduler_parameters": _snapshot_model_parameters(scheduler_model),
                "allocation_parameters": allocation_parameters,
            }
        if progress_callback is not None:
            progress_callback(row)
        if checkpoint_callback is not None:
            checkpoint_callback(
                meta_index + 1, best, allocation_model, scheduler_model,
                meta_metrics, time.perf_counter() - started_at,
                allocation_result, scheduling_result, allocation_parameters,
            )

    if best is None:
        raise RuntimeError("joint training produced no allocation result")
    for row in meta_metrics:
        row["selected_as_best"] = row["meta_episode"] == best["meta_episode"]
    final_queues = make_queues(
        path_list,
        best["assignment"],
        beads,
        config=config,
    )
    _restore_model_parameters(scheduler_model, best["scheduler_parameters"])
    deterministic_scheduling = run_scheduling(
        scheduler_model,
        final_queues,
        max_steps=5000,
        deterministic=True,
        environment_factory=evaluation_environment_factory,
    )
    final_scheduling, final_evaluation_source = _choose_final_scheduling(
        deterministic_scheduling,
        best["scheduling_result"],
    )
    if model_dir is not None:
        target = Path(model_dir)
        target.mkdir(parents=True, exist_ok=True)
        allocation_model.save(str(target / "allocation_model"))
        scheduler_model.save(str(target / "scheduling_model"))

    reward_history = {
        "meta_allocation_rewards": allocation_rewards,
        "meta_scheduling_rewards": scheduling_rewards,
        "meta_joint_rewards": joint_rewards,
        "scheduling_bonuses": scheduling_bonuses,
        "allocation_meta_timestep_boundaries": allocation_boundaries,
        "scheduling_meta_timestep_boundaries": scheduling_boundaries,
        "allocation_episode_rewards": allocation_callback.episode_rewards,
        "allocation_episode_timesteps": allocation_callback.episode_timesteps,
        "scheduling_episode_rewards": scheduling_callback.episode_rewards,
        "scheduling_episode_timesteps": scheduling_callback.episode_timesteps,
    }
    stats = {
        "final_reward": float(final_scheduling["reward"]),
        "progress": float(final_scheduling["progress"]),
        "blocked_attempts": int(final_scheduling["collisions"]),
        "total_wait_steps": int(final_scheduling["total_wait_steps"]),
        "decision_steps": int(final_scheduling["decision_steps"]),
        "truncated": bool(final_scheduling["truncated"]),
        "selection_criterion": copy.deepcopy(META_SELECTION_CRITERION),
        "best_meta_selection_key": list(best["selection_key"]),
        "best_meta_scheduling_evaluation": _scheduling_evaluation_stats(
            best["scheduling_result"]
        ),
        "restored_scheduler_deterministic_evaluation": _scheduling_evaluation_stats(
            deterministic_scheduling
        ),
        "final_evaluation_source": final_evaluation_source,
        "best_scheduler_parameters_restored": True,
        "best_joint_reward": float(best["joint_reward"]),
        "max_joint_reward": float(max(joint_rewards)),
        "allocation_episodes_completed": len(allocation_callback.episode_rewards),
        "scheduling_episodes_completed": len(scheduling_callback.episode_rewards),
        "episode_budget_semantics": "minimum episodes, completed PPO rollouts",
    }
    return PointLayerTrainingResult(
        layer_idx=int(layer_idx),
        layer_z=float(layer_z),
        paths=path_list,
        assignment=list(best["assignment"]),
        queues=tuple(final_queues),
        records=list(final_scheduling["records"]),
        step_log=list(final_scheduling["step_log"]),
        makespan_sec=float(final_scheduling["makespan_sec"]),
        complete=bool(final_scheduling["completed"]),
        training_wall_sec=time.perf_counter() - started_at,
        best_meta_episode=int(best["meta_episode"]),
        meta_metrics=meta_metrics,
        reward_history=reward_history,
        evaluation_stats=stats,
    )
