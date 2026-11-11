# 충돌 기하의 FK 연결·TCP 높이와 전체 장면 평행이동 불변성을 검증하는 테스트
import numpy as np
import pytest

from pipeline.S3_path_planning import S35_collision_audit as audit


@pytest.mark.parametrize("tilt", [0.0, 36.0])
def test_tip_and_flange_share_fk_world_frame(monkeypatch, tilt):
    monkeypatch.setattr(audit, "TORCH_TILT_DEG", tilt)
    bases, facing, transforms = audit._robot_frames(np.zeros(3))
    for robot in range(3):
        tip = np.array([150.0, -100.0, 3.9])
        arm, torch, ok = audit._arm_torch_geometry(tip, bases[robot], *facing[robot], transforms[robot])
        assert ok
        np.testing.assert_allclose(torch[-1], tip + [0, 0, 450], atol=1e-6)
        np.testing.assert_allclose(arm[-1], torch[0], atol=1e-6)
        if tilt:
            direction = tip[:2] - bases[robot, :2]
            direction /= np.linalg.norm(direction)
            th = np.radians(tilt)
            expected = tip + np.r_[-direction * audit._TORCH_FLANGE_TIP_MM * np.sin(th),
                                    audit._TORCH_FLANGE_TIP_MM * np.cos(th) + 450]
            np.testing.assert_allclose(arm[-1], expected, atol=1e-6)


def test_collision_clearance_is_invariant_to_world_translation():
    bases, facing, transforms = audit._robot_frames(np.zeros(3))
    shift = np.array([810.0, -360.0, 720.0])
    arms, torches, moved_arms, moved_torches = {}, {}, {}, {}
    for robot in range(3):
        tip = np.array([150.0 * robot, -100.0, 3.9])
        arm, torch, ok = audit._arm_torch_geometry(tip, bases[robot], *facing[robot], transforms[robot])
        assert ok
        moved = transforms[robot].copy()
        moved[:3, 3] += shift
        # raw Z는 베이스 상대값이고 표시 월드 높이는 T_wb에서만 바뀐다.
        moved_tip = tip + [shift[0], shift[1], 0]
        arm2, torch2, ok2 = audit._arm_torch_geometry(
            moved_tip, bases[robot] + [shift[0], shift[1], 0], *facing[robot], moved)
        assert ok2
        np.testing.assert_allclose(arm2, np.asarray(arm) + shift, atol=1e-6)
        np.testing.assert_allclose(torch2, np.asarray(torch) + shift, atol=1e-6)
        arms[robot], torches[robot] = arm, torch
        moved_arms[robot], moved_torches[robot] = arm2, torch2
    before = audit._detect_collisions(arms, 150, 1e18, torch_geometry=torches, torch_radius=80)
    after = audit._detect_collisions(moved_arms, 150, 1e18, torch_geometry=moved_torches, torch_radius=80)
    for pair in before:
        assert after[pair][0] == pytest.approx(before[pair][0], abs=1e-6)


def test_failed_ik_fallback_uses_same_world_height():
    bases, facing, transforms = audit._robot_frames(np.zeros(3))
    tip = np.array([1e6, 1e6, 3.9])
    arm, torch, ok = audit._arm_torch_geometry(tip, bases[0], *facing[0], transforms[0])
    assert not ok and torch is None
    np.testing.assert_allclose(arm[-1], tip + [0, 0, 450])
