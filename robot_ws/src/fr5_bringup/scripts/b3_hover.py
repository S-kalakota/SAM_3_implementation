#!/usr/bin/env python3
"""Task B3, step 2: plan or execute a slow 100 mm fingertip hover.

The input is a SURFACE point in ``base_link`` from ``b3_pick_point.py``.  This
tool adds +0.100 m along base Z, chooses the nearest taught lift orientation,
accounts for the calibrated ``wrist3_link -> tcp_link`` offset, and sends a
pose goal to MoveIt.

Safety behavior:
  * default is PLAN ONLY; the robot cannot move without ``--execute``;
  * execution speed/acceleration are capped at 5 percent;
  * the accepted B2 file, its B1 source hash, and the live static TF must agree;
  * raw targets must remain inside the calibrated B1 workspace envelope;
  * the chosen tool orientation must point the fingertip approximately down;
  * execution is refused unless the current fingertip is already clear above
    the selected surface.

MoveIt has no environment collision scene in this project.  Before execution,
review the plan, clear the physical path, use a safe starting pose, and keep a
hand on the e-stop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, MoveItErrorCodes,
                             OrientationConstraint, PositionConstraint)
from moveit_msgs.srv import GetPositionFK
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener

from repo_paths import calibration_dir


GROUP = 'fairino5_v6_group'
BASE_FRAME = 'base_link'
CAMERA_FRAME = 'zed_left_optical'
WRIST_FRAME = 'wrist3_link'
TCP_FRAME = 'tcp_link'
ARM_JOINTS = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
ORIENTATION_POSES = ('leftLift', 'rightLift')
DEFAULT_CALIBRATION = calibration_dir(Path(__file__)) / 'T_base_cam.json'
HOVER_M = 0.100
CALIBRATION_MARGIN_M = 0.075
EXECUTE_CLEARANCE_M = 0.080
MAX_SCALE = 0.05
POSITION_TOLERANCE_M = 0.003
ORIENTATION_TOLERANCE_RAD = math.radians(3.0)
CONTROLLER_GOAL_TOLERANCE_M = 0.005
MAX_TOOL_DOWN_ANGLE_DEG = 25.0
LIVE_TRANSLATION_TOLERANCE_M = 0.001
LIVE_ROTATION_TOLERANCE_RAD = math.radians(0.1)


class HoverError(RuntimeError):
    pass


def parse_xyz(text):
    try:
        values = np.asarray([float(value.strip()) for value in text.split(',')],
                            dtype=float)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            '--target must be comma-separated x,y,z metres') from exc
    if values.shape != (3,) or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError(
            '--target must contain exactly three finite numbers')
    return values


def normalize_quaternion(values):
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise HoverError('invalid quaternion')
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise HoverError('zero quaternion')
    return quaternion / norm


def quaternion_to_matrix(values):
    x, y, z, w = normalize_quaternion(values)
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z),
         2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),
         1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),
         2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])


def matrix_to_quaternion(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray([
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
            0.25 * s,
        ])
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] -
                          rotation[2, 2]) * 2.0
            quaternion = np.asarray([
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[2, 1] - rotation[1, 2]) / s,
            ])
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] -
                          rotation[2, 2]) * 2.0
            quaternion = np.asarray([
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
            ])
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] -
                          rotation[1, 1]) * 2.0
            quaternion = np.asarray([
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ])
    quaternion = normalize_quaternion(quaternion)
    return -quaternion if quaternion[3] < 0.0 else quaternion


def message_quaternion(message):
    return np.asarray([message.x, message.y, message.z, message.w], dtype=float)


def message_translation(message):
    return np.asarray([message.x, message.y, message.z], dtype=float)


def quaternion_angle(first, second):
    dot = abs(float(np.dot(normalize_quaternion(first),
                           normalize_quaternion(second))))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def tool_down_angle_deg(quaternion):
    tool_z_in_base = quaternion_to_matrix(quaternion)[:, 2]
    cosine = max(-1.0, min(1.0, float(-tool_z_in_base[2])))
    return math.degrees(math.acos(cosine))


def load_named_poses():
    srdf = (Path(get_package_share_directory('fr5_bringup')) /
            'config' / 'fr5.srdf')
    poses = {}
    for group_state in ET.parse(srdf).getroot().iter('group_state'):
        if group_state.get('group') != GROUP:
            continue
        poses[group_state.get('name')] = {
            joint.get('name'): float(joint.get('value'))
            for joint in group_state.iter('joint')
        }
    return poses


def load_calibration(path: Path):
    try:
        calibration = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoverError(f'cannot load B2 calibration {path}: {exc}') from exc
    if calibration.get('schema_version') != 1:
        raise HoverError(f'unsupported B2 schema in {path}')
    if not calibration.get('quality', {}).get('passed', False):
        raise HoverError(f'B2 calibration failed quality gates: {path}')
    if calibration.get('parent_frame') != BASE_FRAME:
        raise HoverError(f'B2 parent frame must be {BASE_FRAME}')
    if calibration.get('child_frame') != CAMERA_FRAME:
        raise HoverError(f'B2 child frame must be {CAMERA_FRAME}')

    translation = np.asarray(calibration.get('t_m'), dtype=float)
    quaternion = normalize_quaternion(calibration.get('quaternion_xyzw'))
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise HoverError('invalid B2 translation')

    source_path = Path(calibration.get('source_capture', '')).expanduser()
    if not source_path.is_file():
        sibling = path.with_name('calib_points.json')
        source_path = sibling if sibling.is_file() else source_path
    try:
        source_raw = source_path.read_bytes()
        capture = json.loads(source_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise HoverError(f'cannot load B1 source capture {source_path}: {exc}') from exc
    expected_hash = calibration.get('source_sha256')
    actual_hash = hashlib.sha256(source_raw).hexdigest()
    if not expected_hash or actual_hash != expected_hash:
        raise HoverError(
            'B1 points changed after B2; rerun b2_fit_transform.py --write')
    try:
        base_points = np.asarray(
            [pair['base_xyz'] for pair in capture['pairs']], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise HoverError('invalid B1 base points') from exc
    if base_points.ndim != 2 or base_points.shape[1] != 3:
        raise HoverError('invalid B1 base point dimensions')
    return calibration, translation, quaternion, base_points


def load_target_file(path: Path, calibration):
    try:
        target = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoverError(f'cannot load B3 target {path}: {exc}') from exc
    if target.get('schema_version') != 1:
        raise HoverError(f'unsupported B3 target schema in {path}')
    if target.get('base_frame') != BASE_FRAME:
        raise HoverError(f'B3 target frame must be {BASE_FRAME}')
    if target.get('camera_frame') != CAMERA_FRAME:
        raise HoverError(f'B3 target camera frame must be {CAMERA_FRAME}')
    if target.get('calibration_source_sha256') != calibration.get('source_sha256'):
        raise HoverError('B3 target was produced from a different B2 calibration')
    surface = np.asarray(target.get('base_surface_xyz_m'), dtype=float)
    if surface.shape != (3,) or not np.isfinite(surface).all():
        raise HoverError(f'invalid base_surface_xyz_m in {path}')
    return surface


def check_target_envelope(surface, base_points):
    lower = base_points.min(axis=0) - CALIBRATION_MARGIN_M
    upper = base_points.max(axis=0) + CALIBRATION_MARGIN_M
    if np.any(surface < lower) or np.any(surface > upper):
        raise HoverError(
            f'target {np.round(surface, 4).tolist()} is outside the calibrated '
            f'B1 envelope {np.round(lower, 4).tolist()} .. '
            f'{np.round(upper, 4).tolist()}')


class B3HoverNode(Node):
    def __init__(self, node_name='b3_hover'):
        super().__init__(node_name)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def lookup_transform(self, target_frame, source_frame, timeout_sec=5.0):
        deadline = time.monotonic() + timeout_sec
        last_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame, source_frame, rclpy.time.Time())
            except Exception as exc:  # tf2 exception classes vary by distro.
                last_error = exc
        raise HoverError(
            f'no TF {target_frame} <- {source_frame}: {last_error}')

    def validate_live_calibration(self, expected_translation,
                                  expected_quaternion):
        transform = self.lookup_transform(BASE_FRAME, CAMERA_FRAME)
        live_translation = message_translation(transform.transform.translation)
        live_quaternion = message_quaternion(transform.transform.rotation)
        translation_error = np.linalg.norm(
            live_translation - expected_translation)
        rotation_error = quaternion_angle(live_quaternion, expected_quaternion)
        if translation_error > LIVE_TRANSLATION_TOLERANCE_M or \
                rotation_error > LIVE_ROTATION_TOLERANCE_RAD:
            raise HoverError(
                'live camera TF does not match T_base_cam.json: '
                f'translation difference {translation_error * 1000:.3f} mm, '
                f'rotation difference {math.degrees(rotation_error):.4f} deg. '
                'Restart bringup with the accepted B2 calibration.')
        return translation_error, rotation_error

    def named_tcp_pose(self, name, joint_targets):
        client = self.create_client(GetPositionFK, 'compute_fk')
        if not client.wait_for_service(timeout_sec=10.0):
            raise HoverError('compute_fk unavailable; is move_group running?')
        request = GetPositionFK.Request()
        request.header.frame_id = BASE_FRAME
        request.fk_link_names = [TCP_FRAME]
        request.robot_state.joint_state.name = list(joint_targets)
        request.robot_state.joint_state.position = list(joint_targets.values())
        request.robot_state.is_diff = False
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None or not response.pose_stamped:
            raise HoverError(f'compute_fk returned no TCP pose for {name}')
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise HoverError(
                f'compute_fk failed for {name}: {response.error_code.val}')
        pose = response.pose_stamped[0].pose
        position = np.asarray(
            [pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quaternion = message_quaternion(pose.orientation)
        return position, normalize_quaternion(quaternion)

    def choose_orientation(self, choice, surface, named_poses):
        names = ORIENTATION_POSES if choice == 'auto' else (choice,)
        candidates = []
        for name in names:
            if name not in named_poses:
                raise HoverError(f'named pose {name!r} is missing from the SRDF')
            position, quaternion = self.named_tcp_pose(name, named_poses[name])
            down_angle = tool_down_angle_deg(quaternion)
            if down_angle > MAX_TOOL_DOWN_ANGLE_DEG:
                raise HoverError(
                    f'{name} tool-down angle {down_angle:.1f} deg exceeds '
                    f'{MAX_TOOL_DOWN_ANGLE_DEG:.1f} deg')
            xy_distance = float(np.linalg.norm(position[:2] - surface[:2]))
            candidates.append((xy_distance, name, position, quaternion, down_angle))
        return min(candidates, key=lambda candidate: candidate[0])

    def desired_wrist_pose(self, tcp_position, tcp_quaternion):
        wrist_to_tcp = self.lookup_transform(WRIST_FRAME, TCP_FRAME)
        fixed_translation = message_translation(
            wrist_to_tcp.transform.translation)
        fixed_rotation = quaternion_to_matrix(message_quaternion(
            wrist_to_tcp.transform.rotation))
        tcp_rotation = quaternion_to_matrix(tcp_quaternion)
        wrist_rotation = tcp_rotation @ fixed_rotation.T
        wrist_position = tcp_position - wrist_rotation @ fixed_translation
        return (wrist_position, matrix_to_quaternion(wrist_rotation),
                fixed_translation)

    def plan_or_execute(self, tcp_position, wrist_position, wrist_quaternion,
                        tcp_offset_in_wrist, execute, scale):
        client = ActionClient(self, MoveGroup, 'move_action')
        if not client.wait_for_server(timeout_sec=10.0):
            raise HoverError('move_action unavailable; is move_group running?')

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = BASE_FRAME
        position_constraint.link_name = WRIST_FRAME
        # Constrain the calibrated TCP point on wrist3_link, not wrist3_link's
        # origin. Otherwise orientation tolerance multiplied by the 232 mm tool
        # length can create a centimetre-scale fingertip position error.
        position_constraint.target_point_offset.x = float(tcp_offset_in_wrist[0])
        position_constraint.target_point_offset.y = float(tcp_offset_in_wrist[1])
        position_constraint.target_point_offset.z = float(tcp_offset_in_wrist[2])
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [POSITION_TOLERANCE_M]
        region_pose = Pose()
        region_pose.position.x = float(tcp_position[0])
        region_pose.position.y = float(tcp_position[1])
        region_pose.position.z = float(tcp_position[2])
        region_pose.orientation.w = 1.0
        position_constraint.constraint_region.primitives = [sphere]
        position_constraint.constraint_region.primitive_poses = [region_pose]
        position_constraint.weight = 1.0

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = BASE_FRAME
        orientation_constraint.link_name = WRIST_FRAME
        orientation_constraint.orientation.x = float(wrist_quaternion[0])
        orientation_constraint.orientation.y = float(wrist_quaternion[1])
        orientation_constraint.orientation.z = float(wrist_quaternion[2])
        orientation_constraint.orientation.w = float(wrist_quaternion[3])
        orientation_constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        orientation_constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        orientation_constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        orientation_constraint.parameterization = OrientationConstraint.ROTATION_VECTOR
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.name = 'B3 100 mm TCP hover'
        constraints.position_constraints = [position_constraint]
        constraints.orientation_constraints = [orientation_constraint]

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP
        goal.request.pipeline_id = 'ompl'
        goal.request.max_velocity_scaling_factor = scale
        goal.request.max_acceleration_scaling_factor = scale
        goal.request.allowed_planning_time = 10.0
        goal.request.num_planning_attempts = 10
        goal.request.start_state.is_diff = True
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = not execute
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        mode = 'EXECUTE' if execute else 'PLAN ONLY'
        self.get_logger().info(
            f'{mode}: wrist target {np.round(wrist_position, 6).tolist()} m, '
            f'scale={scale:.3f}')
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise HoverError('MoveGroup rejected the B3 goal')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise HoverError('MoveGroup returned no result')
        result = wrapped_result.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise HoverError(
                f'MoveGroup failed with error code {result.error_code.val}')
        trajectory = result.planned_trajectory.joint_trajectory
        duration = 0.0
        if trajectory.points:
            end = trajectory.points[-1].time_from_start
            duration = end.sec + end.nanosec * 1e-9
        return len(trajectory.points), duration


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        '--target', type=parse_xyz,
        help='surface point in base_link as x,y,z metres (not the hover point)')
    target_group.add_argument('--target-file', type=Path,
                              help='JSON emitted by b3_pick_point.py')
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION,
                        help=f'accepted B2 JSON (default: {DEFAULT_CALIBRATION})')
    parser.add_argument('--orientation-pose',
                        choices=('auto',) + ORIENTATION_POSES, default='auto',
                        help='taught lift orientation (default: nearest lift)')
    parser.add_argument('--scale', type=float, default=MAX_SCALE,
                        help=f'velocity/acceleration scale, max {MAX_SCALE}')
    parser.add_argument('--execute', action='store_true',
                        help='actually move; default is plan-only')
    args = parser.parse_args(argv)
    if args.scale <= 0.0 or args.scale > MAX_SCALE:
        parser.error(f'--scale must be in (0, {MAX_SCALE}]')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        calibration_path = args.calibration.expanduser()
        calibration, expected_t, expected_q, base_points = load_calibration(
            calibration_path)
        surface = (load_target_file(args.target_file.expanduser(), calibration)
                   if args.target_file else args.target)
        check_target_envelope(surface, base_points)
    except HoverError as exc:
        print(f'B3 REFUSED: {exc}', file=sys.stderr)
        return 2

    hover = surface + np.asarray([0.0, 0.0, HOVER_M])
    print('\n=== B3 100 mm HOVER ===')
    print('surface target [base_link, m]: ' +
          ' '.join(f'{value:+.6f}' for value in surface))
    print('TCP hover goal [base_link, m]: ' +
          ' '.join(f'{value:+.6f}' for value in hover))
    print(f'mode: {"EXECUTE - ROBOT WILL MOVE" if args.execute else "PLAN ONLY"}')
    print(f'velocity/acceleration scale: {args.scale:.3f}')
    print('WARNING: the environment is not represented in MoveIt. Clear the path.')

    rclpy.init()
    node = B3HoverNode()
    try:
        translation_error, rotation_error = node.validate_live_calibration(
            expected_t, expected_q)
        print(f'live B2 TF matches JSON: {translation_error * 1000:.3f} mm, '
              f'{math.degrees(rotation_error):.4f} deg difference')

        current_tf = node.lookup_transform(BASE_FRAME, TCP_FRAME)
        current_tcp = message_translation(current_tf.transform.translation)
        print('current TCP [base_link, m]: ' +
              ' '.join(f'{value:+.6f}' for value in current_tcp))
        required_z = surface[2] + EXECUTE_CLEARANCE_M
        if current_tcp[2] < required_z:
            message = (
                f'current TCP z={current_tcp[2]:.3f} m is not at least '
                f'{EXECUTE_CLEARANCE_M * 1000:.0f} mm above surface '
                f'z={surface[2]:.3f} m; jog the arm clear first')
            if args.execute:
                raise HoverError(message)
            print(f'PLAN-ONLY WARNING: {message}')

        named_poses = load_named_poses()
        distance, pose_name, _reference_position, tcp_quaternion, down_angle = \
            node.choose_orientation(args.orientation_pose, surface, named_poses)
        print(f'orientation: {pose_name} (reference XY distance '
              f'{distance * 1000:.1f} mm, tool-down angle {down_angle:.1f} deg)')
        (wrist_position, wrist_quaternion,
         tcp_offset_in_wrist) = node.desired_wrist_pose(
            hover, tcp_quaternion)
        print('derived wrist goal [base_link, m]: ' +
              ' '.join(f'{value:+.6f}' for value in wrist_position))

        if args.execute:
            print('\nEXECUTING AT <=5%. KEEP HAND ON E-STOP.')
        points, duration = node.plan_or_execute(
            hover, wrist_position, wrist_quaternion, tcp_offset_in_wrist,
            args.execute, args.scale)
        print(f'\n{"EXECUTION" if args.execute else "PLAN"} PASS: '
              f'{points} trajectory points, {duration:.2f} s')

        if args.execute:
            time.sleep(0.25)
            actual_tf = node.lookup_transform(BASE_FRAME, TCP_FRAME)
            actual = message_translation(actual_tf.transform.translation)
            controller_error = np.linalg.norm(actual - hover)
            print('actual TCP [base_link, m]: ' +
                  ' '.join(f'{value:+.6f}' for value in actual))
            print(f'robot-reported goal error: {controller_error * 1000:.2f} mm')
            if controller_error > CONTROLLER_GOAL_TOLERANCE_M:
                raise HoverError(
                    f'robot-reported TCP is {controller_error * 1000:.2f} mm '
                    f'from the hover goal (limit '
                    f'{CONTROLLER_GOAL_TOLERANCE_M * 1000:.1f} mm)')
            print('Now physically measure signed X/Y miss to the marked point. '
                  'B3 requires <=15 mm.')
        else:
            print('No robot motion occurred. Review the plan and physical path, '
                  'then rerun the same command with --execute.')
    except HoverError as exc:
        print(f'B3 REFUSED: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('B3 interrupted; requesting no further action.', file=sys.stderr)
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
