from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Deque, Iterable, Tuple

import mujoco
import numpy as np


JOINT_NAMES = ("shoulder_yaw", "shoulder_pitch", "elbow", "wrist")
TRAVEL_PITCH_POSE = np.array([-0.4, 1.6, 1.2])


class ArmLab:
    """MuJoCo arm with smooth, current-state-aware target planning."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parents[2] / "models" / "arm.xml"

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.planner_data = mujoco.MjData(self.model)
        self.site_id = self.model.site("end_effector").id
        target_body_id = self.model.body("target").id
        self.target_mocap_id = int(self.model.body_mocapid[target_body_id])

        self.joint_ids = np.array([self.model.joint(name).id for name in JOINT_NAMES])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids]
        self.joint_ranges = self.model.jnt_range[self.joint_ids].copy()

        self.auto_mode = True
        self.ik_gain = 4.5
        self.max_joint_speed = 1.25
        self.damping = 0.08
        self.trajectory_speed = 2.0
        self.replan_delay = 0.08
        # Begin in a bent, non-singular pose so the Jacobian solver can move
        # toward targets on either side of the starting position.
        self.joint_targets = np.array([0.0, 0.35, 0.5, 1.0])

        self._plan_dirty = True
        self._waypoints: Deque[Tuple[str, np.ndarray]] = deque()
        self._trajectory_start: np.ndarray | None = None
        self._trajectory_goal: np.ndarray | None = None
        self._trajectory_elapsed = 0.0
        self._trajectory_duration = 0.0
        self._replan_countdown = 0.0
        self._last_observed_target = np.zeros(3)
        self.motion_status = "planning"
        self.last_plan_used_safe_route = False
        self.last_plan_detour_scale = 0.0

        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_ids] = self.joint_targets
        self.data.ctrl[:] = self.joint_targets
        self.set_target((0.42, 0.0, 0.52))
        mujoco.mj_forward(self.model, self.data)
        self.request_replan()

    @property
    def target(self) -> np.ndarray:
        return self.data.mocap_pos[self.target_mocap_id].copy()

    @property
    def end_effector(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    @property
    def position_error(self) -> float:
        return float(np.linalg.norm(self.target - self.end_effector))

    def set_target(self, xyz: Iterable[float]) -> None:
        target = np.asarray(tuple(xyz), dtype=float)
        if target.shape != (3,):
            raise ValueError("Target must contain exactly three coordinates")

        # Keep targets over the table and within the arm's useful workspace.
        target[0] = np.clip(target[0], -0.62, 0.62)
        target[1] = np.clip(target[1], -0.52, 0.52)
        target[2] = np.clip(target[2], 0.22, 0.92)
        self.data.mocap_pos[self.target_mocap_id] = target
        self._last_observed_target = target.copy()
        self.request_replan(self.replan_delay)

    def move_target(self, delta: Iterable[float]) -> None:
        self.set_target(self.target + np.asarray(tuple(delta), dtype=float))

    def randomize_target(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        radius = rng.uniform(0.28, 0.57)
        azimuth = rng.uniform(-1.0, 1.0)
        height = rng.uniform(0.28, 0.72)
        self.set_target((radius * np.cos(azimuth), radius * np.sin(azimuth), height))

    def adjust_joint(self, index: int, delta: float) -> None:
        if not 0 <= index < len(self.joint_targets):
            raise IndexError("Joint index is out of range")
        self.auto_mode = False
        self._clear_plan()
        self.joint_targets[index] = np.clip(
            self.joint_targets[index] + delta,
            self.joint_ranges[index, 0],
            self.joint_ranges[index, 1],
        )

    def set_auto_mode(self, enabled: bool) -> None:
        self.auto_mode = enabled
        if enabled:
            self.request_replan()
        else:
            self._clear_plan()
            self.motion_status = "manual"

    def request_replan(self, delay: float = 0.0) -> None:
        self._plan_dirty = True
        self._replan_countdown = max(0.0, delay)

    def _detect_external_target_change(self) -> None:
        target = self.target
        if np.linalg.norm(target - self._last_observed_target) > 1e-6:
            self._last_observed_target = target
            self.request_replan(self.replan_delay)

    def _clear_plan(self) -> None:
        self._waypoints.clear()
        self._trajectory_start = None
        self._trajectory_goal = None
        self._trajectory_elapsed = 0.0
        self._trajectory_duration = 0.0

    @staticmethod
    def _angle_difference(target: float, current: float) -> float:
        return float((target - current + np.pi) % (2 * np.pi) - np.pi)

    def _desired_yaw(self, target: np.ndarray) -> float:
        yaw = float(np.arctan2(target[1], target[0]))
        return float(np.clip(yaw, self.joint_ranges[0, 0], self.joint_ranges[0, 1]))

    def _solve_ik_candidate(
        self,
        seed: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        data = self.planner_data
        mujoco.mj_resetData(self.model, data)
        data.qpos[self.qpos_ids] = np.clip(
            seed,
            self.joint_ranges[:, 0],
            self.joint_ranges[:, 1],
        )

        jacobian_pos = np.zeros((3, self.model.nv))
        jacobian_rot = np.zeros((3, self.model.nv))
        for _ in range(350):
            mujoco.mj_forward(self.model, data)
            error = target - data.site_xpos[self.site_id]
            if np.linalg.norm(error) < 0.002:
                break

            mujoco.mj_jacSite(
                self.model,
                data,
                jacobian_pos,
                jacobian_rot,
                self.site_id,
            )
            jacobian = jacobian_pos[:, self.dof_ids]
            regularizer = 0.03**2 * np.eye(3)
            update = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + regularizer,
                error,
            )
            update = np.clip(update, -0.08, 0.08)
            data.qpos[self.qpos_ids] += update
            data.qpos[self.qpos_ids] = np.clip(
                data.qpos[self.qpos_ids],
                self.joint_ranges[:, 0],
                self.joint_ranges[:, 1],
            )

        mujoco.mj_forward(self.model, data)
        result = data.qpos[self.qpos_ids].copy()
        error = float(np.linalg.norm(target - data.site_xpos[self.site_id]))
        return result, error

    def _configuration_has_collision(self, joint_positions: np.ndarray) -> bool:
        data = self.planner_data
        mujoco.mj_resetData(self.model, data)
        data.qpos[self.qpos_ids] = joint_positions
        mujoco.mj_forward(self.model, data)
        return data.ncon > 0

    def _path_has_collision(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        samples: int = 48,
    ) -> bool:
        for blend in np.linspace(0.0, 1.0, samples):
            joint_positions = start + blend * (goal - start)
            if self._configuration_has_collision(joint_positions):
                return True
        return False

    def _solve_target_configuration(self, current: np.ndarray) -> np.ndarray | None:
        target = self.target
        desired_yaw = self._desired_yaw(target)
        seeds = (
            np.array([desired_yaw, *current[1:]]),
            np.array([desired_yaw, 0.35, 0.5, 1.0]),
            np.array([desired_yaw, *TRAVEL_PITCH_POSE]),
            np.array([desired_yaw, 0.2, 0.9, 1.5]),
            np.array([desired_yaw, 1.0, -1.0, -0.5]),
        )

        candidates: list[tuple[float, float, np.ndarray]] = []
        for seed in seeds:
            solution, error = self._solve_ik_candidate(seed, target)
            yaw_mismatch = abs(self._angle_difference(desired_yaw, solution[0]))
            if (
                error < 0.035
                and yaw_mismatch < np.deg2rad(25)
                and not self._configuration_has_collision(solution)
            ):
                travel = float(np.linalg.norm(solution - current))
                candidates.append((travel, error, solution))

        if not candidates:
            return None
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        return candidates[0][2]

    def _build_motion_plan(self) -> None:
        self._clear_plan()
        current = self.data.qpos[self.qpos_ids].copy()
        goal = self._solve_target_configuration(current)
        self._plan_dirty = False

        if goal is None:
            self.motion_status = "local IK fallback"
            self.last_plan_used_safe_route = False
            return

        signed_yaw_change = self._angle_difference(goal[0], current[0])
        equivalent_goal_yaw = current[0] + signed_yaw_change
        if self.joint_ranges[0, 0] <= equivalent_goal_yaw <= self.joint_ranges[0, 1]:
            goal[0] = equivalent_goal_yaw

        self.last_plan_used_safe_route = False
        self.last_plan_detour_scale = 0.0

        # Rotation size alone no longer triggers the old fixed travel pose.
        # Execute the shortest all-joint path whenever it is contact-free.
        if not self._path_has_collision(current, goal):
            self._waypoints.append(("coordinated reaching", goal))
            self.motion_status = "planned"
            return

        # If the direct path is blocked, search for the smallest useful bend
        # toward the compact pose instead of immediately retracting fully.
        midpoint = 0.5 * (current + goal)
        for detour_scale in (0.15, 0.3, 0.5, 0.75, 1.0):
            clearance = midpoint.copy()
            clearance[1:] = (
                (1.0 - detour_scale) * midpoint[1:]
                + detour_scale * TRAVEL_PITCH_POSE
            )
            if (
                not self._path_has_collision(current, clearance)
                and not self._path_has_collision(clearance, goal)
            ):
                self._waypoints.append(("minimal clearance", clearance))
                self._waypoints.append(("coordinated reaching", goal))
                self.last_plan_used_safe_route = True
                self.last_plan_detour_scale = detour_scale
                self.motion_status = "planned"
                return

        # Rare final fallback for a configuration where sampled clearance
        # routes are all blocked.
        retract = np.array([current[0], *TRAVEL_PITCH_POSE])
        rotate = np.array([goal[0], *TRAVEL_PITCH_POSE])
        self._waypoints.append(("retracting", retract))
        self._waypoints.append(("rotating", rotate))
        self._waypoints.append(("reaching", goal))
        self.last_plan_used_safe_route = True
        self.last_plan_detour_scale = 1.0
        self.motion_status = "planned"

    def _start_next_trajectory(self) -> None:
        if not self._waypoints:
            self._trajectory_start = None
            self._trajectory_goal = None
            self.motion_status = "tracking"
            return

        label, goal = self._waypoints.popleft()
        # Continue from the preceding command, not the slightly lagging
        # physical pose, so adjacent trajectory segments remain continuous.
        start = self.joint_targets.copy()
        largest_move = float(np.max(np.abs(goal - start)))
        # A quintic blend peaks at 1.875 times its average speed.
        duration = 1.875 * largest_move / self.trajectory_speed
        self._trajectory_start = start
        self._trajectory_goal = goal
        self._trajectory_elapsed = 0.0
        self._trajectory_duration = max(0.35, duration)
        self.motion_status = label

    def _update_trajectory(self) -> None:
        if self._trajectory_goal is None or self._trajectory_start is None:
            self._start_next_trajectory()
            return

        self._trajectory_elapsed += self.model.opt.timestep
        progress = min(1.0, self._trajectory_elapsed / self._trajectory_duration)
        blend = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
        self.joint_targets = self._trajectory_start + blend * (
            self._trajectory_goal - self._trajectory_start
        )

        if progress >= 1.0:
            actual_error = np.max(
                np.abs(self.data.qpos[self.qpos_ids] - self._trajectory_goal)
            )
            if actual_error < 0.06:
                self._trajectory_start = None
                self._trajectory_goal = None

    def update_inverse_kinematics(self) -> None:
        error = self.target - self.end_effector
        jacobian_pos = np.zeros((3, self.model.nv))
        jacobian_rot = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacobian_pos,
            jacobian_rot,
            self.site_id,
        )
        jacobian = jacobian_pos[:, self.dof_ids]

        regularizer = (self.damping**2) * np.eye(3)
        joint_velocity = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + regularizer,
            self.ik_gain * error,
        )
        joint_velocity = np.clip(
            joint_velocity,
            -self.max_joint_speed,
            self.max_joint_speed,
        )
        self.joint_targets += joint_velocity * self.model.opt.timestep
        self.joint_targets = np.clip(
            self.joint_targets,
            self.joint_ranges[:, 0],
            self.joint_ranges[:, 1],
        )

    def step(self) -> None:
        if self.auto_mode:
            self._detect_external_target_change()
            if self._plan_dirty:
                self._replan_countdown -= self.model.opt.timestep
                if self._replan_countdown <= 0:
                    self._build_motion_plan()
            if self._trajectory_goal is not None or self._waypoints:
                self._update_trajectory()
            elif self._plan_dirty:
                self.motion_status = "target settling"
            else:
                self.motion_status = "tracking"
        self.data.ctrl[:] = self.joint_targets
        mujoco.mj_step(self.model, self.data)

    def run_steps(self, count: int) -> None:
        for _ in range(count):
            self.step()
