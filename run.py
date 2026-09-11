from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from robot_arm_lab import ArmLab  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive virtual robot arm lab")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a viewer and print the final reaching error",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="Simulation duration in headless mode",
    )
    return parser.parse_args()


def run_headless(lab: ArmLab, seconds: float) -> int:
    steps = max(1, int(seconds / lab.model.opt.timestep))
    lab.run_steps(steps)
    print(f"Target:       {np.round(lab.target, 3)}")
    print(f"End effector: {np.round(lab.end_effector, 3)}")
    print(f"Final error:  {lab.position_error:.4f} m")
    return 0 if lab.position_error < 0.06 else 1


def run_interactive(lab: ArmLab) -> int:
    try:
        import mujoco.viewer
        from mujoco.glfw import glfw
    except ImportError as exc:
        print(f"Could not load the MuJoCo viewer: {exc}", file=sys.stderr)
        return 2

    target_step = 0.035
    joint_step = np.deg2rad(4)

    def on_key(keycode: int) -> None:
        if keycode == glfw.KEY_SPACE:
            lab.set_auto_mode(not lab.auto_mode)
        elif keycode == glfw.KEY_R:
            lab.randomize_target()
        elif keycode == glfw.KEY_UP:
            lab.move_target((target_step, 0, 0))
        elif keycode == glfw.KEY_DOWN:
            lab.move_target((-target_step, 0, 0))
        elif keycode == glfw.KEY_LEFT:
            lab.move_target((0, target_step, 0))
        elif keycode == glfw.KEY_RIGHT:
            lab.move_target((0, -target_step, 0))
        elif keycode == glfw.KEY_PAGE_UP:
            lab.move_target((0, 0, target_step))
        elif keycode == glfw.KEY_PAGE_DOWN:
            lab.move_target((0, 0, -target_step))
        elif keycode in (glfw.KEY_1, glfw.KEY_2):
            lab.adjust_joint(0, joint_step if keycode == glfw.KEY_2 else -joint_step)
        elif keycode in (glfw.KEY_3, glfw.KEY_4):
            lab.adjust_joint(1, joint_step if keycode == glfw.KEY_4 else -joint_step)
        elif keycode in (glfw.KEY_5, glfw.KEY_6):
            lab.adjust_joint(2, joint_step if keycode == glfw.KEY_6 else -joint_step)
        elif keycode in (glfw.KEY_7, glfw.KEY_8):
            lab.adjust_joint(3, joint_step if keycode == glfw.KEY_8 else -joint_step)

    print("Virtual Robot Arm Lab")
    print("Space: auto/manual | Arrows + PgUp/PgDn: target | R: random target")
    print("Manual joints: 1/2, 3/4, 5/6, 7/8")

    with mujoco.viewer.launch_passive(
        lab.model,
        lab.data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=True,
    ) as viewer:
        while viewer.is_running():
            frame_start = time.time()
            lab.step()
            viewer.sync()

            remaining = lab.model.opt.timestep - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    return 0


def main() -> int:
    args = parse_args()
    lab = ArmLab()
    if args.headless:
        return run_headless(lab, args.seconds)
    return run_interactive(lab)


if __name__ == "__main__":
    raise SystemExit(main())
