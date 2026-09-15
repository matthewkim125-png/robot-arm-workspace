# Robot Arm Workspace

An interactive MuJoCo simulation of a four-joint tabletop robot arm. Move the joints manually, reposition a target in 3D, or let a damped least-squares inverse-kinematics controller guide the gripper to the target.

This is milestone 1 of a larger pick-and-place project. It deliberately starts with reliable arm motion before adding grasping, loose objects, and path planning.

## What works now

- A custom four-joint arm with shoulder yaw, shoulder pitch, elbow, and wrist
- Smooth actuator-driven movement rather than teleporting the arm
- Automatic target reaching using the arm's live MuJoCo Jacobian
- Current-state-aware motion planning with smooth quintic trajectories
- Shortest-path coordinated motion even for large rotations
- Adaptive collision detours instead of a mandatory fixed travel pose
- Critically damped joint servos that settle without oscillating
- Target-change debouncing so rapid edits produce one clean replan
- Manual joint control for understanding the arm's motion
- Movable and randomized targets
- Headless demo mode for quick verification
- Automated checks for model structure and reaching behavior

## Setup

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run it

Launch the interactive viewer:

```bash
python run.py
```

On macOS, MuJoCo's interactive viewer may require its supplied launcher:

```bash
mjpython run.py
```

Run a viewer-free reaching check:

```bash
python run.py --headless --seconds 5
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Controls

| Key | Action |
|---|---|
| `Space` | Toggle automatic/manual control |
| Arrow keys | Move target in the horizontal plane |
| `Page Up` / `Page Down` | Move target up/down |
| `R` | Move target to a random reachable location |
| `1` / `2` | Decrease/increase shoulder yaw |
| `3` / `4` | Decrease/increase shoulder pitch |
| `5` / `6` | Decrease/increase elbow angle |
| `7` / `8` | Decrease/increase wrist angle |

The numbered joint controls switch the simulation into manual mode. Target controls remain available in either mode.

## How it works

When the target changes, a multi-start inverse-kinematics solver searches for a clean target-facing joint configuration and favors the solution nearest the arm's current pose. The arm follows a time-scaled quintic joint trajectory, so its speed eases in and out instead of changing abruptly. The trajectory speed is tuned for a brisk response, while critically damped position servos prevent the links from overshooting and shaking around their commands.

Rapid target edits are grouped over a very short settling window before a new route is generated. This prevents held keys or target dragging from repeatedly restarting the motion while remaining visually immediate.

Large rotation no longer forces the arm through a fixed retract–rotate–extend pose. The planner first samples the shortest direct all-joint route from the arm's current configuration to the selected target configuration. If that route is contact-free, it executes it even when the target is directly behind the arm.

When the direct route is blocked, the planner tests progressively larger clearance bends and uses the smallest collision-free one it finds. Full retraction is retained only as a rare final fallback when every smaller sampled detour is blocked.

MuJoCo contact checks reject routes and final configurations that collide with the table, floor, or non-neighboring parts of the arm. Gravity compensation lets the position servos hold the planned configuration without a reactive controller continually changing the route.

This is classical robotics, not machine learning. It establishes the environment and control baseline that a later RL project could be compared against.

## Project structure

```text
virtual-robot-arm-lab/
├── models/arm.xml
├── src/robot_arm_lab/
│   ├── __init__.py
│   └── simulation.py
├── tests/test_simulation.py
├── run.py
└── requirements.txt
```

