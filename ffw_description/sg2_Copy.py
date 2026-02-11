#!/usr/bin/env python3
"""Isaac Sim: phase0 approach above box, phase1 move to flap sides, then hold."""

import os
from pathlib import Path
import numpy as np

# Robot USD
USD_PATH = "/home/youngho/git/ai-worker/ffw_description/sg2simul.usd"

# Box location in USD (adjust if needed)
BOX_ROOT_PATH = "/World/Box"
BOX_CENTER_FALLBACK = (0.55, 0.0, 0.35)
BOX_CENTER_Z_BIAS = 0.1  # add height if box is on a pedestal

# Flap side offsets from box center
FLAP_SIDE_OFFSET = (0.0, 0.35, 0.0)  # left = +Y, right = -Y

# Motion targets (relative to box center)
APPROACH_Z = 0.45  # above box center
SIDE_Z = 0.30      # height to put grippers beside flaps

# IK control
USE_IK = True
LULA_DESC_PATH = "/home/youngho/git/ai-worker/ffw_description/sg2.yaml"
ROBOT_URDF_PATH = "/home/youngho/git/ai-worker/ffw_description/urdf/ffw_sg2_rev1_follower/ffw_sg2_follower.urdf"
LEFT_EE_FRAME = "eepoint_l"
RIGHT_EE_FRAME = "eepoint_r"
IK_POS_TOL = 0.02
IK_UPDATE_EVERY_N = 1
JOINT_BLEND_ALPHA = 0.6
HOLD_EPS = 0.05
INWARD_OFFSET = 0.0  # meters toward box center in final phase

# Visualization
DEBUG_SPHERES = True
DEBUG_SPHERE_RADIUS = 0.02
DEBUG_SPHERE_LEFT_PATH = "/World/DebugTargets/left_target"
DEBUG_SPHERE_RIGHT_PATH = "/World/DebugTargets/right_target"


# Phase timing
PHASE_HOLD_STEPS = 10  # require steady reach before next phase
PHASE_MAX = 2  # stop at phase2

# IK failure retry
FAIL_STREAK_MAX = 20  # consecutive failures before retry
RETRY_MAX = 5  # max number of retries
RETRY_HOLD_STEPS = 50  # keep retry offset for these steps
RETRY_POS_STD = 0.03  # meters
RETRY_Z_STD = 0.015  # meters
RETRY_SEED_BASE = 1000
RETRY_ROLLBACK_PHASE = False  # keep current phase on retry
RETRY_ROLLBACK_STEPS = 1  # how many phases to roll back
FAIL_TERMINATE = True  # stop simulation after max retries

# Upper-joint exploration (shoulder -> elbow) on retry
SEED_JOINT_STD = 0.25  # radians
SEED_JOINT_NAMES_L = ("arm_l_joint1", "arm_l_joint2", "arm_l_joint3")
SEED_JOINT_NAMES_R = ("arm_r_joint1", "arm_r_joint2", "arm_r_joint3")


def _apply_blended_action(robot, action, alpha):
    if action is None or robot is None:
        return
    joint_indices = getattr(action, "joint_indices", None)
    joint_positions = getattr(action, "joint_positions", None)
    if joint_indices is None or joint_positions is None:
        return
    curr = robot.get_joint_positions(joint_indices=joint_indices)
    if curr is None:
        return
    target = (1.0 - alpha) * curr + alpha * joint_positions
    robot.set_joint_positions(target, joint_indices=joint_indices)


def _find_articulation_root():
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    return None


def _get_joint_indices(robot, names):
    try:
        all_names = list(robot.get_joint_names())
    except Exception:
        return []
    indices = []
    for n in names:
        try:
            indices.append(all_names.index(n))
        except ValueError:
            continue
    return indices


def _get_prim_world_pos(stage, path):
    try:
        from pxr import UsdGeom
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        xformable = UsdGeom.Xformable(prim)
        xform = xformable.ComputeLocalToWorldTransform()
        return np.array(xform.ExtractTranslation(), dtype=np.float32)
    except Exception:
        return None


def main() -> int:
    usd_path = Path(USD_PATH).expanduser().resolve()
    headless = os.environ.get("ISAACSIM_HEADLESS", "0") == "1"

    from isaacsim.simulation_app import SimulationApp

    simulation_app = SimulationApp({"headless": headless})

    if not usd_path.is_file():
        print(f"ERROR: USD file not found: {usd_path}")
        simulation_app.close()
        return 1

    try:
        from isaacsim.core.api.world import World
        from isaacsim.core.utils.stage import open_stage
        import omni.usd
        import omni.timeline
        from pxr import UsdGeom, Gf
    except Exception as exc:
        print(f"ERROR: failed to import Isaac Sim modules: {exc}")
        simulation_app.close()
        return 1

    open_stage(str(usd_path))

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    def _ensure_sphere(path, color_rgb):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            sphere = UsdGeom.Sphere.Define(stage, path)
            sphere.GetRadiusAttr().Set(DEBUG_SPHERE_RADIUS)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color_rgb)])
            xform = UsdGeom.Xformable(sphere)
            xform.AddTranslateOp()
            return sphere
        return UsdGeom.Sphere(prim)

    def _set_sphere_pos(sphere, pos):
        if sphere is None:
            return
        xform = UsdGeom.Xformable(sphere)
        ops = xform.GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        else:
            xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    left_sphere = None
    right_sphere = None
    if DEBUG_SPHERES:
        left_sphere = _ensure_sphere(DEBUG_SPHERE_LEFT_PATH, (1.0, 0.1, 0.1))
        right_sphere = _ensure_sphere(DEBUG_SPHERE_RIGHT_PATH, (0.1, 0.3, 1.0))

    for _ in range(10):
        world.step(render=True)

    robot = None
    ik_left = None
    ik_right = None
    seed_joint_indices_l = []
    seed_joint_indices_r = []
    if USE_IK:
        try:
            from isaacsim.core.api.robots import Robot
            from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver
            from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver

            robot_root = _find_articulation_root()
            if robot_root is None:
                raise RuntimeError("Articulation root not found in USD.")

            robot = Robot(robot_root, name="sg2_robot")
            world.scene.add(robot)
            world.reset()
            robot.initialize()
            seed_joint_indices_l = _get_joint_indices(robot, SEED_JOINT_NAMES_L)
            seed_joint_indices_r = _get_joint_indices(robot, SEED_JOINT_NAMES_R)

            kin = LulaKinematicsSolver(
                robot_description_path=LULA_DESC_PATH,
                urdf_path=ROBOT_URDF_PATH,
            )
            base_pos, base_quat = robot.get_world_pose()
            kin.set_robot_base_pose(base_pos, base_quat)
            frame_names = kin.get_all_frame_names()
            if LEFT_EE_FRAME not in frame_names or RIGHT_EE_FRAME not in frame_names:
                raise RuntimeError(f"EE frame not found. left={LEFT_EE_FRAME} right={RIGHT_EE_FRAME}")
            ik_left = ArticulationKinematicsSolver(robot, kin, LEFT_EE_FRAME)
            ik_right = ArticulationKinematicsSolver(robot, kin, RIGHT_EE_FRAME)
        except Exception as exc:
            print(f"WARNING: IK init failed, falling back to no IK: {exc}")
            robot = None
            ik_left = None
            ik_right = None

    try:
        step = 0
        phase = 0
        phase_hold = 0
        fail_streak = 0
        retry_count = 0
        retry_ttl = 0
        retry_offset_l = np.zeros(3, dtype=np.float32)
        retry_offset_r = np.zeros(3, dtype=np.float32)
        curr_left = None
        curr_right = None
        desired_left = None
        desired_right = None
        final_orient_ready = False
        final_orient_set_once = False
        while simulation_app.is_running():
            world.step(render=True)
            if USE_IK and ik_left and ik_right and step % IK_UPDATE_EVERY_N == 0:
                box_center = _get_prim_world_pos(stage, BOX_ROOT_PATH)
                if box_center is None:
                    box_center = np.array(BOX_CENTER_FALLBACK, dtype=np.float32)
                else:
                    box_center = box_center + np.array([0.0, 0.0, BOX_CENTER_Z_BIAS], dtype=np.float32)

                left_side = box_center + np.array(FLAP_SIDE_OFFSET, dtype=np.float32)
                right_side = box_center - np.array(FLAP_SIDE_OFFSET, dtype=np.float32)
                if phase == 0:
                    left_t = left_side + np.array([0.0, 0.0, APPROACH_Z], dtype=np.float32)
                    right_t = right_side + np.array([0.0, 0.0, APPROACH_Z], dtype=np.float32)
                elif phase == 1:
                    left_t = left_side + np.array([0.0, 0.0, SIDE_Z], dtype=np.float32)
                    right_t = right_side + np.array([0.0, 0.0, SIDE_Z], dtype=np.float32)
                else:
                    left_t = left_side + np.array([0.0, -0.10, SIDE_Z], dtype=np.float32)
                    right_t = right_side + np.array([0.0, 0.10, SIDE_Z], dtype=np.float32)

                if retry_ttl > 0:
                    left_t = left_t + retry_offset_l
                    right_t = right_t + retry_offset_r
                    retry_ttl -= 1

                if DEBUG_SPHERES:
                    try:
                        _set_sphere_pos(left_sphere, left_t)
                        _set_sphere_pos(right_sphere, right_t)
                    except Exception:
                        pass

                try:
                    fk_l, _ = ik_left.compute_end_effector_pose()
                    fk_r, _ = ik_right.compute_end_effector_pose()
                    curr_left = np.array(fk_l, dtype=np.float32)
                    curr_right = np.array(fk_r, dtype=np.float32)
                except Exception:
                    if curr_left is None:
                        curr_left = left_t.copy()
                    if curr_right is None:
                        curr_right = right_t.copy()

                delta_l = left_t - curr_left
                delta_r = right_t - curr_right
                if np.linalg.norm(delta_l) < HOLD_EPS and np.linalg.norm(delta_r) < HOLD_EPS:
                    phase_hold += 1
                    if phase_hold >= PHASE_HOLD_STEPS:
                        if phase == PHASE_MAX and not final_orient_set_once:
                            final_orient_ready = True
                            final_orient_set_once = True
                        phase = min(phase + 1, PHASE_MAX)
                        phase_hold = 0
                    step += 1
                    if not final_orient_ready:
                        continue
                else:
                    phase_hold = 0

                desired_left = left_t.copy()
                desired_right = right_t.copy()
                if final_orient_ready and INWARD_OFFSET > 0.0:
                    dl_dir = box_center - desired_left
                    dr_dir = box_center - desired_right
                    dl_norm = np.linalg.norm(dl_dir)
                    dr_norm = np.linalg.norm(dr_dir)
                    if dl_norm > 1e-6:
                        desired_left = desired_left + (INWARD_OFFSET * dl_dir / dl_norm)
                    if dr_norm > 1e-6:
                        desired_right = desired_right + (INWARD_OFFSET * dr_dir / dr_norm)
                try:
                    action_l, ok_l = ik_left.compute_inverse_kinematics(
                        desired_left,
                        position_tolerance=IK_POS_TOL,
                    )
                    action_r, ok_r = ik_right.compute_inverse_kinematics(
                        desired_right,
                        position_tolerance=IK_POS_TOL,
                    )
                except Exception as exc:
                    print(f"ERROR: IK compute failed: {exc}")
                    ok_l = False
                    ok_r = False
                    action_l = None
                    action_r = None

                if ok_l and ok_r:
                    fail_streak = 0
                else:
                    fail_streak += 1
                    if fail_streak >= FAIL_STREAK_MAX:
                        if retry_count < RETRY_MAX:
                            retry_count += 1
                            fail_streak = 0
                            phase_hold = 0
                            if RETRY_ROLLBACK_PHASE:
                                phase = max(0, phase - RETRY_ROLLBACK_STEPS)
                            rng = np.random.default_rng(RETRY_SEED_BASE + retry_count)
                            noise_xy = rng.normal(0.0, RETRY_POS_STD, size=2).astype(np.float32)
                            noise_z = float(rng.normal(0.0, RETRY_Z_STD))
                            retry_offset_l = np.array([noise_xy[0], noise_xy[1], noise_z], dtype=np.float32)
                            retry_offset_r = np.array([noise_xy[0], -noise_xy[1], noise_z], dtype=np.float32)
                            retry_ttl = RETRY_HOLD_STEPS
                            if robot is not None:
                                if seed_joint_indices_l:
                                    curr = robot.get_joint_positions(joint_indices=seed_joint_indices_l)
                                    if curr is not None:
                                        curr = np.array(curr, dtype=np.float32)
                                        curr += rng.normal(0.0, SEED_JOINT_STD, size=curr.shape)
                                        robot.set_joint_positions(curr, joint_indices=seed_joint_indices_l)
                                if seed_joint_indices_r:
                                    curr = robot.get_joint_positions(joint_indices=seed_joint_indices_r)
                                    if curr is not None:
                                        curr = np.array(curr, dtype=np.float32)
                                        curr += rng.normal(0.0, SEED_JOINT_STD, size=curr.shape)
                                        robot.set_joint_positions(curr, joint_indices=seed_joint_indices_r)
                            print(
                                "WARN: IK retry",
                                f"retry={retry_count}/{RETRY_MAX}",
                                f"phase={phase}",
                                f"offset={retry_offset_l}",
                            )
                        elif FAIL_TERMINATE:
                            print("ERROR: IK retry exceeded max retries, terminating loop.")
                            break

                if robot is not None:
                    if ok_l:
                        _apply_blended_action(robot, action_l, JOINT_BLEND_ALPHA)
                    if ok_r:
                        _apply_blended_action(robot, action_r, JOINT_BLEND_ALPHA)

            step += 1
    except Exception as exc:
        print(f"ERROR: loop exception: {exc}")
    finally:
        simulation_app.close()

    print("INFO: loop ended, is_running=", simulation_app.is_running())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
