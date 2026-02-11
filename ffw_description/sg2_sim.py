#!/usr/bin/env python3
"""
Isaac Sim IK demo (pose tracking):
- Phase-based motion
- Each phase defines a fixed pose (position + orientation)
- Box pose via SingleXFormPrim
- EE current via kinematic FK
- Quaternion-based phase check (wxyz)
- Target quaternion continuity + rate limiting
- Per-joint step limiting
- Per-phase completion checks
"""

import numpy as np
from pathlib import Path
from isaacsim.simulation_app import SimulationApp

# ======================================================
# Simulation
# ======================================================
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction

# ======================================================
# Paths
# ======================================================
USD_PATH = "/home/youngho/git/ai-worker/ffw_description/temp.usd"
LULA_DESC_PATH = "/home/youngho/git/ai-worker/ffw_description/sg2.yaml"
ROBOT_URDF_PATH = "/home/youngho/git/ai-worker/ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"

# ======================================================
# Box
# ======================================================
BOX_ROOT_PATH = "/World/Box"
BOX_CENTER_FALLBACK = (0.55, 0.0, 0.35)
BOX_CENTER_Z_BIAS = 0.1
BOX_SAFETY_RADIUS_XY = 0.20
BOX_TOP_SAFE_Z_OFFSET = 0.20
BOX_TOP_SAFE_Z_OFFSET_MOVE_ONLY = 0.00

# ======================================================
# IK
# ======================================================
LEFT_EE_FRAME = "eepoint_l"
RIGHT_EE_FRAME = "eepoint_r"
IK_POS_TOL = 0.02
ROTATE_IK_POS_TOL = 0.05
MOVE_ONLY_IK_POS_TOL = 0.005
IK_UPDATE_EVERY_N = 1

# ======================================================
# Phase config
# ======================================================
PHASE_HOLD_STEPS = 100
HOLD_EPS = 0.1
ORI_EPS_RAD = np.deg2rad(5.0)
ROTATE_PHASE = 1
LOCK_POSITION_DURING_ROTATE = True
ROTATE_POS_EPS = 0.12
MOVE_ONLY_PHASES = {2}
PHASE_COMPLETE_POS_EPS = 0.04
PHASE_COMPLETE_ROTATE_POS_EPS = 0.07
PHASE_COMPLETE_ORI_EPS_RAD = np.deg2rad(2.5)
PHASE_COMPLETE_POS_EPS_MOVE_ONLY = 0.04
DEBUG_PRINT_EVERY = 100

# ======================================================
# Trajectory config
# ======================================================
TRAJ_DURATION_SEC = 28.0
ROTATE_SPLITS = 5

MAX_TARGET_ROT_STEP_RAD = np.deg2rad(1.0)
MAX_JOINT_STEP = 0.0035
MAX_JOINT_STEP_ROTATE = 0.0025
CMD_SMOOTHING_ALPHA = 0.28
CMD_SMOOTHING_ALPHA_ROTATE = 0.20
MAX_JOINT_STEP_SINGLE_ARM = 0.006
CMD_SMOOTHING_ALPHA_SINGLE_ARM = 0.45

# ======================================================
# Orientation (wxyz)
# ======================================================
L_EE_APPROACH_QUAT = np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float32)
L_EE_GRIP_QUAT     = np.array([0.0, 0.0, -0.7071068, 0.7071068], dtype=np.float32)
R_EE_APPROACH_QUAT = np.array([0.5, -0.5, -0.5, -0.5], dtype=np.float32)
R_EE_GRIP_QUAT     = np.array([0.7071068, -0.7071068, 0.0, 0.0], dtype=np.float32)

PHASE_POSES = {
    0: {
        "left":  {"pos": np.array([-0.1,  0.3, 0.3]), "quat": L_EE_APPROACH_QUAT},
        "right": {"pos": np.array([-0.1, -0.3, 0.3]), "quat": R_EE_APPROACH_QUAT},
    },
    1: {
        "left":  {"pos": np.array([-0.1,  0.3, 0.3]), "quat": L_EE_GRIP_QUAT},
        "right": {"pos": np.array([-0.1, -0.3, 0.3]), "quat": R_EE_GRIP_QUAT},
    },
    2: {
        "left":  {"pos": np.array([0.0,  0.2, 0.1]), "quat": L_EE_GRIP_QUAT},
        "right": {"pos": np.array([0.0, -0.2, 0.1]), "quat": R_EE_GRIP_QUAT},
    },
}
PHASE_MAX = max(PHASE_POSES.keys())

LEFT_ARM_JOINT_NAMES  = [f"arm_l_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINT_NAMES = [f"arm_r_joint{i}" for i in range(1, 8)]

# ======================================================
# Utils
# ======================================================
def quat_angle_error_wxyz(q1, q2):
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.clip(abs(np.dot(q1, q2)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)


def rotmat_to_quat_wxyz(R):
    """
    Convert rotation matrix to quaternion (w, x, y, z)
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return q / np.linalg.norm(q)


def find_articulation_root():
    import omni.usd
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    return None


def compute_phase_poses(phase, box_center):
    cfg = PHASE_POSES[phase]
    return (
        box_center + cfg["left"]["pos"],
        cfg["left"]["quat"],
        box_center + cfg["right"]["pos"],
        cfg["right"]["quat"],
    )


def apply_box_clearance(target_pos, box_center, top_safe_z_offset):
    p = target_pos.copy()
    delta_xy = p[:2] - box_center[:2]
    dist_xy = np.linalg.norm(delta_xy)
    if dist_xy < BOX_SAFETY_RADIUS_XY:
        if dist_xy < 1e-8:
            delta_xy = np.array([1.0, 0.0], dtype=np.float32)
            dist_xy = 1.0
        p[:2] = box_center[:2] + (delta_xy / dist_xy) * BOX_SAFETY_RADIUS_XY
    p[2] = max(p[2], box_center[2] + top_safe_z_offset)
    return p


def get_box_center(box_prim):
    box_pos, _ = box_prim.get_world_pose()
    box_center = np.array(box_pos) if box_pos is not None else np.array(BOX_CENTER_FALLBACK)
    box_center[2] += BOX_CENTER_Z_BIAS
    return box_center


def get_fk_state(ik_left, ik_right):
    fk_l_pos, fk_l_rot = ik_left.compute_end_effector_pose()
    fk_r_pos, fk_r_rot = ik_right.compute_end_effector_pose()
    curr_l_quat = rotmat_to_quat_wxyz(fk_l_rot)
    curr_r_quat = rotmat_to_quat_wxyz(fk_r_rot)
    return fk_l_pos, fk_r_pos, curr_l_quat, curr_r_quat, fk_l_rot, fk_r_rot


def merge_lr_actions_by_dof(robot, action_l, action_r):
    joint_indices, joint_positions = [], []

    def extract(action, joint_names):
        if action is None:
            return
        ik_map = {int(i): p for i, p in zip(action.joint_indices, action.joint_positions)}
        for name in joint_names:
            dof = robot.get_dof_index(name)
            if dof in ik_map:
                joint_indices.append(dof)
                joint_positions.append(ik_map[dof])

    extract(action_l, LEFT_ARM_JOINT_NAMES)
    extract(action_r, RIGHT_ARM_JOINT_NAMES)

    if not joint_indices:
        return None

    return ArticulationAction(
        joint_indices=np.array(joint_indices, dtype=np.int32),
        joint_positions=np.array(joint_positions, dtype=np.float32),
    )

def lerp(a, b, t):
    return (1.0 - t) * a + t * b

# 회전 보간 (쿼터니언 slerp)
def slerp_quat_wxyz(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return (q0 + t * (q1 - q0)) / np.linalg.norm(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1

# 회전 보간 (분할 slerp)
def slerp_quat_wxyz_split(q0, q1, t, splits=2):
    splits = max(1, int(splits))
    if splits <= 1:
        return slerp_quat_wxyz(q0, q1, t)
    t = np.clip(t, 0.0, 1.0)
    seg = min(splits - 1, int(t * splits))
    t0 = seg / splits
    t1 = (seg + 1) / splits
    local_t = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    q_start = slerp_quat_wxyz(q0, q1, t0)
    q_end = slerp_quat_wxyz(q0, q1, t1)
    return slerp_quat_wxyz(q_start, q_end, local_t)


def clamp_quat_target_step(prev_q, target_q, max_step_rad):
    target_q = target_q / np.linalg.norm(target_q)
    if prev_q is None:
        return target_q
    prev_q = prev_q / np.linalg.norm(prev_q)
    if np.dot(prev_q, target_q) < 0.0:
        target_q = -target_q
    ang = quat_angle_error_wxyz(prev_q, target_q)
    if ang <= max_step_rad:
        return target_q
    t = max_step_rad / max(ang, 1e-8)
    return slerp_quat_wxyz(prev_q, target_q, t)


def align_quat_sign(target_q, ref_q):
    if np.dot(target_q, ref_q) < 0.0:
        return -target_q
    return target_q

# 관절 변화량 제한
def clamp_joint_step(curr, target, max_step):
    delta = target - curr
    delta = np.clip(delta, -max_step, max_step)
    return curr + delta


def smooth_joint_targets(indices, curr, target, prev_cmd_map, alpha):
    smoothed = target.copy()
    for k, dof in enumerate(indices):
        dof_i = int(dof)
        prev_cmd = prev_cmd_map.get(dof_i, float(curr[dof_i]))
        smoothed[k] = prev_cmd + alpha * (target[k] - prev_cmd)
    return smoothed


# ======================================================
# Main
# ======================================================
def main():
    usd_path = Path(USD_PATH)
    if not usd_path.is_file():
        print("USD not found")
        return

    from isaacsim.core.api.world import World
    from isaacsim.core.utils.stage import open_stage
    import omni.timeline
    from isaacsim.core.api.robots import Robot
    from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver
    from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

    open_stage(str(usd_path))
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    omni.timeline.get_timeline_interface().play()

    for _ in range(10):
        world.step(render=True)

    robot = Robot(find_articulation_root(), name="sg2")
    world.scene.add(robot)
    world.reset()
    robot.initialize()

    kin = LulaKinematicsSolver(
        robot_description_path=LULA_DESC_PATH,
        urdf_path=ROBOT_URDF_PATH,
    )

    base_pos, base_quat = robot.get_world_pose()
    kin.set_robot_base_pose(base_pos, base_quat)

    ik_left  = ArticulationKinematicsSolver(robot, kin, LEFT_EE_FRAME)
    ik_right = ArticulationKinematicsSolver(robot, kin, RIGHT_EE_FRAME)

    box_prim = SingleXFormPrim(prim_path=BOX_ROOT_PATH)

    phase, phase_hold, step = 0, 0, 0
    traj_step = 0
    traj_steps = max(1, int(TRAJ_DURATION_SEC / (1.0 / 60.0)))

    # Initialize trajectory start/end
    box_center = get_box_center(box_prim)
    dl_pos_end, dl_quat_end, dr_pos_end, dr_quat_end = compute_phase_poses(phase, box_center)

    fk_l_pos, fk_r_pos, curr_l_quat, curr_r_quat, fk_l_rot, fk_r_rot = get_fk_state(
        ik_left, ik_right
    )

    dl_pos_start = fk_l_pos
    dl_quat_start = curr_l_quat
    dr_pos_start = fk_r_pos
    dr_quat_start = curr_r_quat

    prev_target_l_quat = None
    prev_target_r_quat = None
    last_action_l = None
    last_action_r = None
    prev_cmd_map = {}

    while simulation_app.is_running():
        world.step(render=True)

        if step % IK_UPDATE_EVERY_N != 0:
            step += 1
            continue

        box_center = get_box_center(box_prim)
        fk_l_pos, fk_r_pos, curr_l_quat, curr_r_quat, fk_l_rot, fk_r_rot = get_fk_state(
            ik_left, ik_right
        )

        # Trajectory target for current step
        t = min(1.0, traj_step / traj_steps)
        if phase == ROTATE_PHASE and LOCK_POSITION_DURING_ROTATE:
            dl_pos = dl_pos_start
            dr_pos = dr_pos_start
        else:
            dl_pos = lerp(dl_pos_start, dl_pos_end, t)
            dr_pos = lerp(dr_pos_start, dr_pos_end, t)

        top_safe_z_offset = (
            BOX_TOP_SAFE_Z_OFFSET_MOVE_ONLY if phase in MOVE_ONLY_PHASES else BOX_TOP_SAFE_Z_OFFSET
        )
        dl_pos = apply_box_clearance(dl_pos, box_center, top_safe_z_offset)
        dr_pos = apply_box_clearance(dr_pos, box_center, top_safe_z_offset)

        # Keep move-only phases orientation-fixed and always align quaternion sign to start.
        if phase in MOVE_ONLY_PHASES:
            dl_quat_goal = dl_quat_start
            dr_quat_goal = dr_quat_start
        else:
            dl_quat_goal = align_quat_sign(dl_quat_end, dl_quat_start)
            dr_quat_goal = align_quat_sign(dr_quat_end, dr_quat_start)

        dl_quat = slerp_quat_wxyz_split(dl_quat_start, dl_quat_goal, t, splits=ROTATE_SPLITS)
        dr_quat = slerp_quat_wxyz_split(dr_quat_start, dr_quat_goal, t, splits=ROTATE_SPLITS)

        # Keep target orientation continuous and rate-limited.
        dl_quat = clamp_quat_target_step(prev_target_l_quat, dl_quat, MAX_TARGET_ROT_STEP_RAD)
        dr_quat = clamp_quat_target_step(prev_target_r_quat, dr_quat, MAX_TARGET_ROT_STEP_RAD)
        prev_target_l_quat = dl_quat.copy()
        prev_target_r_quat = dr_quat.copy()

        ori_err_l = quat_angle_error_wxyz(dl_quat_goal, curr_l_quat)
        ori_err_r = quat_angle_error_wxyz(dr_quat_goal, curr_r_quat)
        ori_ok = (
            ori_err_l < PHASE_COMPLETE_ORI_EPS_RAD
            and ori_err_r < PHASE_COMPLETE_ORI_EPS_RAD
        )

        # Completion checks:
        # - rotate phase: evaluate against current trajectory point
        # - move phases: evaluate against current point while moving, end pose after trajectory finishes
        if phase == ROTATE_PHASE:
            pos_ref_l = dl_pos
            pos_ref_r = dr_pos
            eps = PHASE_COMPLETE_ROTATE_POS_EPS
        elif traj_step < traj_steps:
            pos_ref_l = dl_pos
            pos_ref_r = dr_pos
            eps = PHASE_COMPLETE_POS_EPS
        else:
            pos_ref_l = dl_pos_end
            pos_ref_r = dr_pos_end
            eps = PHASE_COMPLETE_POS_EPS_MOVE_ONLY if phase in MOVE_ONLY_PHASES else PHASE_COMPLETE_POS_EPS

        pos_err_l = np.linalg.norm(pos_ref_l - fk_l_pos)
        pos_err_r = np.linalg.norm(pos_ref_r - fk_r_pos)
        pos_ok = (pos_err_l < eps and pos_err_r < eps)

        reach_eps = eps
        left_reached = pos_err_l < reach_eps and ori_err_l < PHASE_COMPLETE_ORI_EPS_RAD
        right_reached = pos_err_r < reach_eps and ori_err_r < PHASE_COMPLETE_ORI_EPS_RAD

        if traj_step >= traj_steps:
            if pos_ok and ori_ok:
                if phase < PHASE_MAX:
                    phase_hold += 1
                    if phase_hold >= PHASE_HOLD_STEPS:
                        phase += 1
                        phase_hold = 0
                        fk_l_pos, fk_r_pos, _, _, fk_l_rot, fk_r_rot = get_fk_state(
                            ik_left, ik_right
                        )
                        dl_pos_start = fk_l_pos
                        dr_pos_start = fk_r_pos
                        dl_quat_start = rotmat_to_quat_wxyz(fk_l_rot)
                        dr_quat_start = rotmat_to_quat_wxyz(fk_r_rot)
                        dl_pos_end, dl_quat_end, dr_pos_end, dr_quat_end = compute_phase_poses(
                            phase, box_center
                        )
                        traj_step = 1
                else:
                    # Final phase: hold the end target without restarting the trajectory loop.
                    phase_hold = PHASE_HOLD_STEPS
                    traj_step = traj_steps
            else:
                phase_hold = 0
        else:
            traj_step += 1

        if phase == ROTATE_PHASE:
            ik_pos_tol = ROTATE_IK_POS_TOL
        elif phase in MOVE_ONLY_PHASES:
            ik_pos_tol = MOVE_ONLY_IK_POS_TOL
        else:
            ik_pos_tol = IK_POS_TOL

        action_l, ok_l = ik_left.compute_inverse_kinematics(
            target_position=dl_pos,
            target_orientation=dl_quat,
            position_tolerance=ik_pos_tol,
        )
        action_r, ok_r = ik_right.compute_inverse_kinematics(
            target_position=dr_pos,
            target_orientation=dr_quat,
            position_tolerance=ik_pos_tol,
        )

        # If orientation-constrained IK fails, retry with current orientation
        # to keep Cartesian position progress instead of stalling.
        if not ok_l:
            action_l_fallback, ok_l_fallback = ik_left.compute_inverse_kinematics(
                target_position=dl_pos,
                target_orientation=curr_l_quat,
                position_tolerance=ik_pos_tol,
            )
            if ok_l_fallback and action_l_fallback is not None:
                action_l, ok_l = action_l_fallback, True
        if not ok_r:
            action_r_fallback, ok_r_fallback = ik_right.compute_inverse_kinematics(
                target_position=dr_pos,
                target_orientation=curr_r_quat,
                position_tolerance=ik_pos_tol,
            )
            if ok_r_fallback and action_r_fallback is not None:
                action_r, ok_r = action_r_fallback, True

        # Keep last valid per-arm IK to avoid arm drops when one step fails.
        # Reached-arm suppression is only used after trajectory finishes.
        suppress_reached_action = (
            phase != ROTATE_PHASE and phase < PHASE_MAX and traj_step >= traj_steps
        )
        if suppress_reached_action and left_reached:
            action_l = None
        elif ok_l and action_l is not None:
            last_action_l = action_l
        else:
            action_l = last_action_l
        if suppress_reached_action and right_reached:
            action_r = None
        elif ok_r and action_r is not None:
            last_action_r = action_r
        else:
            action_r = last_action_r

        if step % DEBUG_PRINT_EVERY == 0:
            single_arm_catchup = (
                suppress_reached_action
                and ((left_reached and not right_reached) or (right_reached and not left_reached))
            )
            print("dl_pos", dl_pos)
            print("dl_quat", dl_quat)
            print("dr_pos", dr_pos)
            print("dr_quat", dr_quat)
            print("ok_l", ok_l)
            print("ok_r", ok_r)
            print("fk_l_pos", fk_l_pos)
            print("fk_r_pos", fk_r_pos)
            print("curr_l_quat", curr_l_quat)
            print("curr_r_quat", curr_r_quat)
            print("phase:", phase)
            print(
                "traj_step", traj_step,
                "pos_ok", pos_ok,
                "ori_ok", ori_ok,
                "pos_err_l", pos_err_l,
                "pos_err_r", pos_err_r,
                "ori_err_l", ori_err_l,
                "ori_err_r", ori_err_r,
                "left_reached", left_reached,
                "right_reached", right_reached,
                "single_arm_catchup", single_arm_catchup,
            )

        merged = merge_lr_actions_by_dof(
            robot,
            action_l,
            action_r,
        )

        if merged is not None:
            curr = robot.get_joint_positions()
            single_arm_catchup = (
                suppress_reached_action
                and ((left_reached and not right_reached) or (right_reached and not left_reached))
            )
            if single_arm_catchup:
                max_step = MAX_JOINT_STEP_SINGLE_ARM
                cmd_alpha = CMD_SMOOTHING_ALPHA_SINGLE_ARM
            else:
                max_step = MAX_JOINT_STEP_ROTATE if phase == ROTATE_PHASE else MAX_JOINT_STEP
                cmd_alpha = CMD_SMOOTHING_ALPHA_ROTATE if phase == ROTATE_PHASE else CMD_SMOOTHING_ALPHA
            limited = clamp_joint_step(
                curr[merged.joint_indices],
                merged.joint_positions,
                max_step=max_step
            )
            limited = smooth_joint_targets(
                merged.joint_indices,
                curr,
                limited,
                prev_cmd_map,
                cmd_alpha,
            )
            merged = ArticulationAction(
                joint_indices=merged.joint_indices,
                joint_positions=limited.astype(np.float32)
            )
            robot.apply_action(merged)
            for dof, pos in zip(merged.joint_indices, merged.joint_positions):
                prev_cmd_map[int(dof)] = float(pos)

        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
