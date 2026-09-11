import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_arm_lab import ArmLab  # noqa: E402


class ArmLabTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lab = ArmLab()

    def test_model_has_expected_control_dimensions(self) -> None:
        self.assertEqual(self.lab.model.nu, 4)
        self.assertEqual(len(self.lab.joint_targets), 4)

    def test_target_is_clamped_to_workspace(self) -> None:
        self.lab.set_target((99, -99, 99))
        np.testing.assert_allclose(self.lab.target, (0.62, -0.52, 0.92))

    def test_manual_adjustment_changes_mode_and_target(self) -> None:
        original = self.lab.joint_targets[0]
        self.lab.adjust_joint(0, 0.1)
        self.assertFalse(self.lab.auto_mode)
        self.assertAlmostEqual(self.lab.joint_targets[0], original + 0.1)

    def test_inverse_kinematics_reaches_default_target(self) -> None:
        initial_error = self.lab.position_error
        self.lab.run_steps(800)
        self.assertLess(self.lab.position_error, initial_error)
        self.assertLess(self.lab.position_error, 0.06)

    def test_inverse_kinematics_reaches_seeded_random_targets(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(3):
            lab = ArmLab()
            lab.randomize_target(rng)
            lab.run_steps(1000)
            self.assertLess(lab.position_error, 0.06)

    def test_target_behind_arm_uses_direct_route_when_safe(self) -> None:
        self.lab.run_steps(800)
        self.lab.set_target((-0.42, 0.0, 0.52))
        self.lab.run_steps(20)
        self.assertFalse(self.lab.last_plan_used_safe_route)
        self.assertEqual(self.lab.last_plan_detour_scale, 0.0)
        self.assertEqual(self.lab.motion_status, "coordinated reaching")
        largest_contact_count = 0
        for _ in range(1200):
            self.lab.step()
            largest_contact_count = max(largest_contact_count, self.lab.data.ncon)
        self.assertLess(self.lab.position_error, 0.06)
        self.assertGreater(abs(self.lab.data.qpos[self.lab.qpos_ids[0]]), 2.5)
        self.assertEqual(largest_contact_count, 0)

    def test_planned_joint_targets_change_smoothly(self) -> None:
        self.lab.set_target((-0.42, 0.0, 0.52))
        previous = self.lab.joint_targets.copy()
        largest_step = 0.0
        for _ in range(700):
            self.lab.step()
            largest_step = max(
                largest_step,
                float(np.max(np.abs(self.lab.joint_targets - previous))),
            )
            previous = self.lab.joint_targets.copy()
        self.assertLessEqual(
            largest_step,
            self.lab.trajectory_speed * self.lab.model.opt.timestep * 1.05,
        )

    def test_joint_servo_settles_without_wobble(self) -> None:
        self.lab.set_auto_mode(False)
        self.lab.adjust_joint(1, 0.7)
        self.lab.run_steps(900)
        samples = []
        velocities = []
        for _ in range(150):
            self.lab.step()
            samples.append(self.lab.data.qpos[self.lab.qpos_ids].copy())
            velocities.append(self.lab.data.qvel[self.lab.dof_ids].copy())
        motion_span = np.ptp(np.asarray(samples), axis=0)
        peak_velocity = np.max(np.abs(np.asarray(velocities)))
        self.assertLess(float(np.max(motion_span)), 0.001)
        self.assertLess(float(peak_velocity), 0.01)

    def test_large_move_coordinates_multiple_joints_at_once(self) -> None:
        self.lab.run_steps(800)
        self.lab.set_target((-0.34, 0.25, 0.34))
        coordinated_steps = 0
        previous = self.lab.joint_targets.copy()
        for _ in range(1000):
            self.lab.step()
            moving_joints = np.count_nonzero(
                np.abs(self.lab.joint_targets - previous) > 1e-5
            )
            if moving_joints >= 3:
                coordinated_steps += 1
            previous = self.lab.joint_targets.copy()
        self.assertGreater(coordinated_steps, 100)
        self.assertLess(self.lab.position_error, 0.06)


if __name__ == "__main__":
    unittest.main()
