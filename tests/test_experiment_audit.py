# HRL 관측 범위와 짧은 학습 예산의 실제 PPO 갱신을 검증한다.
from pipeline.S3_path_planning.S39_hrl_paper_training import PathAllocationEnv, train_point_hrl_layer
from tests.s36_hrl.test_hrl_paper_training_rapid import _beads, _three_sector_paths


def test_allocation_observations_are_inside_declared_space():
    paths = _three_sector_paths()
    env = PathAllocationEnv(paths, _beads(paths))
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    done = False
    while not done:
        obs, _, done, _, _ = env.step(0)
        assert env.observation_space.contains(obs)



def test_short_hrl_budget_updates_both_policies(monkeypatch):
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    updates = {"allocation": 0, "scheduling": 0}
    for cls, key in ((MaskablePPO, "allocation"), (PPO, "scheduling")):
        original = cls.train
        def counted(model, original=original, key=key):
            updates[key] += 1
            return original(model)
        monkeypatch.setattr(cls, "train", counted)
    paths = _three_sector_paths()
    result = train_point_hrl_layer(paths, _beads(paths), layer_idx=0, layer_z=3.9,
                                  meta_episodes=1, allocation_episodes=1, scheduling_episodes=1)
    assert updates["allocation"] > 0
    assert updates["scheduling"] > 0
    assert result.evaluation_stats["allocation_episodes_completed"] >= 1

