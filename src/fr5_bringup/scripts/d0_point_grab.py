#!/usr/bin/env python3
"""Experimental clicked-point grab, deliberately skipping Milestone C1-C3.

The input is the fresh target file written by ``b3_pick_point.py``.  The TCP
grasp position defaults to 5 mm below the selected camera surface point; no
table plane, object height, grasp-width, or bin-wall safety gate is applied.

Plan-only sequence (default):
  validate current state at the saved standby start
  -> exact DB standby_to_{left|right}grab trajectory
  -> short new plan from the DB grab endpoint to the 100 mm TCP hover
  -> straight descent to offset grasp point -> straight retreat
  -> short new plan back to the DB grab point
  -> exact DB {left|right}grab_to_{left|right}lift trajectory
  -> exact DB {left|right}lift_to_standby trajectory

Execution sequence (requires both explicit flags):
  open gripper to 100% -> execute the sequence above, pausing at the offset
  grasp point to command a 60% jaw target.  The pickup proceeds only when the
  measured fingers stop at least 8 percentage points more open than requested,
  indicating that the object blocked closure.  Peak motor current is recorded
  as supporting evidence.  A clean empty close reopens, retreats to hover,
  resets through the selected proven DB left/right grab pose, returns to the
  clicked hover, and retries once 5 mm deeper by default.  It does not return
  to standby between attempts.  Faults and ambiguous feedback do not trigger a
  deeper retry.  Every attempt emits one structured result record so a future
  VLA policy can consume the same observation/action/outcome loop.
  Saved database trajectories are replayed point-for-point with their recorded
  timing; MoveIt is used only for the short DB-point/hover connections and the
  Cartesian descent/retreat.

This is a narrow live trial, not the completed pick executive.  MoveIt has no
environment collision scene.  Use a freshly clicked point on the box, inspect
the plan, clear the entire path, keep a hand on the e-stop, and start with an
empty gripper.  The click process itself never has a robot-motion interface.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, MoveItErrorCodes,
                             OrientationConstraint, PositionConstraint,
                             RobotState)
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from a2_gripper import GripperNode
from b3_hover import (ARM_JOINTS, BASE_FRAME, CONTROLLER_GOAL_TOLERANCE_M,
                      DEFAULT_CALIBRATION, EXECUTE_CLEARANCE_M, GROUP, HOVER_M,
                      MAX_SCALE, MAX_TOOL_DOWN_ANGLE_DEG,
                      ORIENTATION_TOLERANCE_RAD, POSITION_TOLERANCE_M, TCP_FRAME,
                      WRIST_FRAME, B3HoverNode, HoverError,
                      check_target_envelope, load_calibration, load_target_file,
                      message_translation, tool_down_angle_deg)


DEFAULT_TARGET = Path('/tmp/fr5_b3_target.json')
DEFAULT_PLANS_DB = (Path.home() / 'fairino_ros_connector' /
                    'fairino_ros_controller' / 'db' / 'plans.sqlite')
OPEN_PCT = 100
DEFAULT_GRASP_CLOSE_PCT = 60
DEFAULT_MIN_GRASP_POSITION_DELTA_PCT = 8.0
DEFAULT_MIN_GRASP_CURRENT_PCT = 0.0
DEFAULT_GRASP_DEPTH_MM = 5.0
DEFAULT_GRASP_RETRIES = 1
DEFAULT_RETRY_STEP_MM = 5.0
MAX_GRASP_DEPTH_MM = 100.0
MAX_GRASP_RETRIES = 3
MAX_RETRY_STEP_MM = 20.0
MAX_TARGET_AGE_S = 600.0
CARTESIAN_STEP_M = 0.005
CARTESIAN_SPEED_M_S = 0.020
CARTESIAN_FRACTION_MIN = 0.999
CARTESIAN_REVOLUTE_JUMP_RAD = math.radians(12.0)
DB_START_TOLERANCE_RAD = 0.020
DB_CHAIN_TOLERANCE_RAD = 0.005
DB_EXECUTION_TIMEOUT_MARGIN_S = 20.0


@dataclass
class SavedPlan:
    name: str
    trajectory: JointTrajectory
    saved_at: str

    @property
    def duration_s(self):
        if not self.trajectory.points:
            return 0.0
        end = self.trajectory.points[-1].time_from_start
        return end.sec + end.nanosec * 1e-9

    @property
    def start(self):
        return list(self.trajectory.points[0].positions)

    @property
    def end(self):
        return list(self.trajectory.points[-1].positions)


def parse_created(value):
    if not isinstance(value, str) or not value:
        raise HoverError('target file has no creation timestamp')
    try:
        created = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise HoverError(f'invalid target creation timestamp {value!r}') from exc
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created.astimezone(timezone.utc)


def check_target_freshness(path, max_age_s, allow_stale):
    try:
        target = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise HoverError(f'cannot inspect target age in {path}: {exc}') from exc
    created = parse_created(target.get('created'))
    age_s = (datetime.now(timezone.utc) - created).total_seconds()
    if age_s < -5.0:
        raise HoverError(
            f'target timestamp is {-age_s:.1f} s in the future; check the clock')
    if age_s > max_age_s and not allow_stale:
        raise HoverError(
            f'target is {age_s:.0f} s old (limit {max_age_s:.0f} s); '
            'click the box again with b3_pick_point.py')
    return max(0.0, age_s)


def duration_from_json(raw):
    return Duration(sec=int(raw.get('sec', 0)),
                    nanosec=int(raw.get('nanosec', 0)))


def duration_ns(value):
    return value.sec * 1_000_000_000 + value.nanosec


def load_saved_plan(db_path, name):
    if not db_path.is_file():
        raise HoverError(f'plans database not found: {db_path}')
    try:
        with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as database:
            row = database.execute(
                'SELECT joint_names, points, saved_at FROM trajectories '
                'WHERE name = ?', (name,)).fetchone()
    except sqlite3.Error as exc:
        raise HoverError(f'cannot read plans database {db_path}: {exc}') from exc
    if row is None:
        raise HoverError(f'saved trajectory {name!r} is missing from {db_path}')
    try:
        stored_names = list(json.loads(row[0]))
        raw_points = list(json.loads(row[1]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HoverError(f'invalid JSON in saved trajectory {name!r}') from exc
    missing = [joint for joint in ARM_JOINTS if joint not in stored_names]
    if missing:
        raise HoverError(f'saved trajectory {name!r} is missing joints {missing}')
    if len(raw_points) < 2:
        raise HoverError(f'saved trajectory {name!r} has fewer than two points')
    indices = [stored_names.index(joint) for joint in ARM_JOINTS]

    trajectory = JointTrajectory()
    trajectory.joint_names = list(ARM_JOINTS)
    previous_time_ns = -1
    for point_index, raw in enumerate(raw_points):
        try:
            stored_positions = list(raw['positions'])
            point = JointTrajectoryPoint()
            point.positions = [float(stored_positions[index]) for index in indices]
            if raw.get('velocities'):
                stored_velocities = list(raw['velocities'])
                point.velocities = [float(stored_velocities[index])
                                    for index in indices]
            if raw.get('accelerations'):
                stored_accelerations = list(raw['accelerations'])
                point.accelerations = [float(stored_accelerations[index])
                                       for index in indices]
            point.time_from_start = duration_from_json(
                raw.get('time_from_start', {}))
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise HoverError(
                f'invalid point {point_index} in saved trajectory {name!r}') from exc
        values = (list(point.positions) + list(point.velocities) +
                  list(point.accelerations))
        if not values or not np.isfinite(np.asarray(values, dtype=float)).all():
            raise HoverError(
                f'non-finite point {point_index} in saved trajectory {name!r}')
        point_time_ns = duration_ns(point.time_from_start)
        if point_time_ns < previous_time_ns:
            raise HoverError(
                f'non-monotonic time at point {point_index} in {name!r}')
        previous_time_ns = point_time_ns
        trajectory.points.append(point)
    if previous_time_ns <= 0:
        raise HoverError(f'saved trajectory {name!r} has zero duration')
    return SavedPlan(name=name, trajectory=trajectory, saved_at=str(row[2]))


def load_side_plans(db_path, side):
    names = (
        f'standby_to_{side}grab',
        f'{side}grab_to_{side}lift',
        f'{side}lift_to_standby',
    )
    plans = tuple(load_saved_plan(db_path, name) for name in names)
    first_delta = max_joint_delta(plans[0].end, plans[1].start)
    second_delta = max_joint_delta(plans[1].end, plans[2].start)
    if first_delta > DB_CHAIN_TOLERANCE_RAD:
        raise HoverError(
            f'DB chain {plans[0].name} -> {plans[1].name} differs by '
            f'{first_delta:.4f} rad (limit {DB_CHAIN_TOLERANCE_RAD:.4f})')
    if second_delta > DB_CHAIN_TOLERANCE_RAD:
        raise HoverError(
            f'DB chain {plans[1].name} -> {plans[2].name} differs by '
            f'{second_delta:.4f} rad (limit {DB_CHAIN_TOLERANCE_RAD:.4f})')
    return plans


def max_joint_delta(first, second):
    if len(first) != len(second):
        raise HoverError('cannot compare joint vectors with different lengths')
    return max(abs(float(a) - float(b)) for a, b in zip(first, second))


def joint_map(positions):
    if len(positions) != len(ARM_JOINTS):
        raise HoverError('saved plan joint vector is not six values')
    return dict(zip(ARM_JOINTS, (float(value) for value in positions)))


def saved_plan_end_state(plan):
    state = RobotState()
    state.is_diff = True
    state.joint_state.name = list(ARM_JOINTS)
    state.joint_state.position = list(plan.end)
    return state


def print_saved_plan(plan, prefix='validated'):
    print(f'  {prefix} DB {plan.name}: '
          f'{len(plan.trajectory.points)} recorded points, '
          f'{plan.duration_s:.2f} s, saved {plan.saved_at}')


def pose_message(position, quaternion):
    pose = Pose()
    pose.position.x = float(position[0])
    pose.position.y = float(position[1])
    pose.position.z = float(position[2])
    pose.orientation.x = float(quaternion[0])
    pose.orientation.y = float(quaternion[1])
    pose.orientation.z = float(quaternion[2])
    pose.orientation.w = float(quaternion[3])
    return pose


def trajectory_duration(trajectory):
    points = trajectory.joint_trajectory.points
    if not points:
        return 0.0
    end = points[-1].time_from_start
    return end.sec + end.nanosec * 1e-9


def trajectory_end_state(trajectory):
    joint_trajectory = trajectory.joint_trajectory
    if not joint_trajectory.points:
        raise HoverError('planner returned an empty trajectory')
    state = RobotState()
    state.is_diff = True
    state.joint_state.name = list(joint_trajectory.joint_names)
    state.joint_state.position = list(joint_trajectory.points[-1].positions)
    return state


def choose_db_side(node, choice, target, plans_by_side):
    sides = ('left', 'right') if choice == 'auto' else (choice,)
    candidates = []
    for side in sides:
        inbound = plans_by_side[side][0]
        position, quaternion = node.named_tcp_pose(
            f'DB {inbound.name} endpoint', joint_map(inbound.end))
        down_angle = tool_down_angle_deg(quaternion)
        if down_angle > MAX_TOOL_DOWN_ANGLE_DEG:
            raise HoverError(
                f'DB {side} grab tool-down angle {down_angle:.1f} deg exceeds '
                f'{MAX_TOOL_DOWN_ANGLE_DEG:.1f} deg')
        xy_distance = float(np.linalg.norm(position[:2] - target[:2]))
        candidates.append(
            (xy_distance, side, position, quaternion, down_angle))
    return min(candidates, key=lambda candidate: candidate[0])


class PointGrabNode(B3HoverNode):
    def __init__(self):
        super().__init__('d0_point_grab')
        self.joint_state = None
        self.create_subscription(JointState, 'joint_states',
                                 self._on_joint_state, 10)
        self.move_client = ActionClient(self, MoveGroup, 'move_action')
        self.cartesian_client = self.create_client(
            GetCartesianPath, 'compute_cartesian_path')
        self.execute_client = ActionClient(
            self, ExecuteTrajectory, 'execute_trajectory')
        self.saved_trajectory_client = ActionClient(
            self, FollowJointTrajectory,
            'fairino5_controller/follow_joint_trajectory')

    def _on_joint_state(self, message):
        self.joint_state = message

    def current_arm_positions(self, timeout_sec=3.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.joint_state is None:
                continue
            lookup = dict(zip(self.joint_state.name,
                              self.joint_state.position))
            if all(joint in lookup for joint in ARM_JOINTS):
                return [float(lookup[joint]) for joint in ARM_JOINTS]
        raise HoverError('no complete arm joint state received')

    def verify_saved_plan_start(self, plan,
                                tolerance=DB_START_TOLERANCE_RAD):
        current = self.current_arm_positions()
        deltas = [abs(actual - expected)
                  for actual, expected in zip(current, plan.start)]
        worst_index = int(np.argmax(deltas))
        worst_delta = deltas[worst_index]
        worst_joint = ARM_JOINTS[worst_index]
        if worst_delta > tolerance:
            raise HoverError(
                f'robot is not at DB start for {plan.name}: {worst_joint} '
                f'differs by {worst_delta:.4f} rad '
                f'(limit {tolerance:.4f}); place the arm at the recorded '
                'standby/start point before running')
        print(f'  DB start check {plan.name}: max difference '
              f'{worst_delta:.4f} rad on {worst_joint}')
        return current

    def execute_saved_plan(self, plan):
        self.verify_saved_plan_start(plan)
        if not self.saved_trajectory_client.wait_for_server(timeout_sec=10.0):
            raise HoverError(
                'fairino5_controller trajectory action unavailable')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = copy.deepcopy(plan.trajectory)
        self.get_logger().info(
            f'EXECUTE EXACT DB TRAJECTORY: {plan.name} '
            f'({len(plan.trajectory.points)} points, {plan.duration_s:.2f} s)')
        send_future = self.saved_trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise HoverError(f'controller rejected DB trajectory {plan.name}')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            self, result_future,
            timeout_sec=plan.duration_s + DB_EXECUTION_TIMEOUT_MARGIN_S)
        wrapped = result_future.result()
        if wrapped is None:
            raise HoverError(f'no controller result for DB trajectory {plan.name}')
        result = wrapped.result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise HoverError(
                f'DB trajectory {plan.name} failed: '
                f'error {result.error_code} {result.error_string}')
        current = self.current_arm_positions()
        endpoint_delta = max_joint_delta(current, plan.end)
        if endpoint_delta > DB_START_TOLERANCE_RAD:
            raise HoverError(
                f'DB trajectory {plan.name} ended {endpoint_delta:.4f} rad '
                'from its recorded endpoint')
        print(f'  executed exact DB {plan.name}: '
              f'{len(plan.trajectory.points)} points, '
              f'endpoint error {endpoint_delta:.4f} rad')

    def _run_move_group(self, goal, execute, label):
        if not self.move_client.wait_for_server(timeout_sec=10.0):
            raise HoverError('move_action unavailable; is move_group running?')
        mode = 'EXECUTE' if execute else 'PLAN'
        self.get_logger().info(f'{mode}: {label}')
        send_future = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise HoverError(f'MoveGroup rejected {label}')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise HoverError(f'MoveGroup returned no result for {label}')
        result = wrapped.result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            raise HoverError(
                f'MoveGroup failed {label}: error {result.error_code.val} '
                f'{result.error_code.message}')
        trajectory = result.planned_trajectory
        if not trajectory.joint_trajectory.points:
            raise HoverError(f'MoveGroup returned an empty plan for {label}')
        print(f'  {mode.lower()} {label}: '
              f'{len(trajectory.joint_trajectory.points)} points, '
              f'{trajectory_duration(trajectory):.2f} s')
        return trajectory, trajectory_end_state(trajectory)

    def plan_tcp_pose(self, tcp_position, wrist_quaternion,
                      tcp_offset_in_wrist, execute, scale, label,
                      start_state=None):
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = BASE_FRAME
        position_constraint.link_name = WRIST_FRAME
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
        orientation_constraint.parameterization = \
            OrientationConstraint.ROTATION_VECTOR
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.name = label
        constraints.position_constraints = [position_constraint]
        constraints.orientation_constraints = [orientation_constraint]

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP
        goal.request.pipeline_id = 'ompl'
        goal.request.max_velocity_scaling_factor = scale
        goal.request.max_acceleration_scaling_factor = scale
        goal.request.allowed_planning_time = 10.0
        goal.request.num_planning_attempts = 10
        if start_state is None:
            goal.request.start_state.is_diff = True
        else:
            goal.request.start_state = copy.deepcopy(start_state)
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = not execute
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return self._run_move_group(goal, execute, label)

    def plan_joint_pose(self, joint_targets, execute, scale, label,
                        start_state=None):
        constraints = Constraints()
        constraints.name = label
        for name in ARM_JOINTS:
            if name not in joint_targets:
                raise HoverError(f'{label} is missing joint {name}')
            constraint = JointConstraint()
            constraint.joint_name = name
            constraint.position = float(joint_targets[name])
            constraint.tolerance_above = 0.005
            constraint.tolerance_below = 0.005
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP
        goal.request.pipeline_id = 'ompl'
        goal.request.max_velocity_scaling_factor = scale
        goal.request.max_acceleration_scaling_factor = scale
        goal.request.allowed_planning_time = 10.0
        goal.request.num_planning_attempts = 10
        if start_state is None:
            goal.request.start_state.is_diff = True
        else:
            goal.request.start_state = copy.deepcopy(start_state)
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = not execute
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        return self._run_move_group(goal, execute, label)

    def compute_cartesian(self, wrist_position, wrist_quaternion, scale,
                          label, start_state=None):
        if not self.cartesian_client.wait_for_service(timeout_sec=10.0):
            raise HoverError(
                'compute_cartesian_path unavailable; is move_group running?')
        request = GetCartesianPath.Request()
        request.header.frame_id = BASE_FRAME
        if start_state is None:
            request.start_state.is_diff = True
        else:
            request.start_state = copy.deepcopy(start_state)
        request.group_name = GROUP
        request.link_name = WRIST_FRAME
        request.waypoints = [pose_message(wrist_position, wrist_quaternion)]
        request.max_step = CARTESIAN_STEP_M
        request.jump_threshold = 0.0
        request.prismatic_jump_threshold = 0.0
        request.revolute_jump_threshold = CARTESIAN_REVOLUTE_JUMP_RAD
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = scale
        request.max_acceleration_scaling_factor = scale
        request.cartesian_speed_limited_link = WRIST_FRAME
        request.max_cartesian_speed = CARTESIAN_SPEED_M_S

        future = self.cartesian_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        response = future.result()
        if response is None:
            raise HoverError(f'Cartesian planner timed out for {label}')
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise HoverError(
                f'Cartesian planner failed {label}: '
                f'error {response.error_code.val} {response.error_code.message}')
        if response.fraction < CARTESIAN_FRACTION_MIN:
            raise HoverError(
                f'Cartesian planner completed only '
                f'{response.fraction * 100.0:.2f}% of {label}')
        trajectory = response.solution
        if not trajectory.joint_trajectory.points:
            raise HoverError(f'Cartesian planner returned an empty {label}')
        print(f'  planned {label}: {response.fraction * 100.0:.1f}%, '
              f'{len(trajectory.joint_trajectory.points)} points, '
              f'{trajectory_duration(trajectory):.2f} s')
        return trajectory, trajectory_end_state(trajectory)

    def execute_cartesian(self, trajectory, label):
        if not self.execute_client.wait_for_server(timeout_sec=10.0):
            raise HoverError(
                'execute_trajectory unavailable; is move_group running?')
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = trajectory
        self.get_logger().info(f'EXECUTE: {label}')
        send_future = self.execute_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            raise HoverError(f'execution rejected for {label}')
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        wrapped = result_future.result()
        if wrapped is None:
            raise HoverError(f'no execution result for {label}')
        if wrapped.result.error_code.val != MoveItErrorCodes.SUCCESS:
            code = wrapped.result.error_code
            raise HoverError(
                f'execution failed {label}: error {code.val} {code.message}')
        print(f'  executed {label}')

    def actual_tcp(self):
        transform = self.lookup_transform(BASE_FRAME, TCP_FRAME)
        return message_translation(transform.transform.translation)


def preflight(node, hover, wrist_hover, wrist_grasp_targets,
              attempt_depths_mm, wrist_quaternion, tcp_offset, plans, scale):
    inbound, grab_to_lift, lift_to_standby = plans
    print('\nPreflight validating DB choreography and planning only the new links:')
    node.verify_saved_plan_start(inbound)
    print_saved_plan(inbound)
    print_saved_plan(grab_to_lift)
    print_saved_plan(lift_to_standby)

    state = saved_plan_end_state(inbound)
    _, state = node.plan_tcp_pose(
        hover, wrist_quaternion, tcp_offset, False, scale,
        f'new {inbound.name} endpoint -> clicked hover', state)
    attempt_count = len(wrist_grasp_targets)
    for attempt_number, (depth_mm, wrist_grasp) in enumerate(
            zip(attempt_depths_mm, wrist_grasp_targets), start=1):
        _, state = node.compute_cartesian(
            wrist_grasp, wrist_quaternion, scale,
            f'attempt {attempt_number}/{attempt_count} straight descent '
            f'({depth_mm:g} mm depth)', state)
        _, state = node.compute_cartesian(
            wrist_hover, wrist_quaternion, scale,
            f'attempt {attempt_number}/{attempt_count} straight retreat '
            'to hover', state)
        if attempt_number < attempt_count:
            _, state = node.plan_joint_pose(
                joint_map(grab_to_lift.start), False, scale,
                f'retry reset: clicked hover -> DB '
                f'{grab_to_lift.name} start', state)
            _, state = node.plan_tcp_pose(
                hover, wrist_quaternion, tcp_offset, False, scale,
                f'retry re-approach: DB {grab_to_lift.name} start -> '
                'clicked hover', state)
    node.plan_joint_pose(
        joint_map(grab_to_lift.start), False, scale,
        f'new clicked hover -> DB {grab_to_lift.name} start', state)
    segment_count = 2 + 2 * attempt_count + 2 * (attempt_count - 1)
    print('PREFLIGHT PASS: three exact DB trajectories validated and all '
          f'{segment_count} new motion segments planned '
          f'({attempt_count} grasp attempt'
          f'{"" if attempt_count == 1 else "s"}).')


def verify_tcp(node, expected, label):
    time.sleep(0.25)
    actual = node.actual_tcp()
    error = float(np.linalg.norm(actual - expected))
    print(f'  actual TCP after {label}: '
          + ' '.join(f'{value:+.6f}' for value in actual) + ' m')
    print(f'  robot-reported {label} error: {error * 1000.0:.2f} mm')
    if error > CONTROLLER_GOAL_TOLERANCE_M:
        raise HoverError(
            f'TCP is {error * 1000.0:.2f} mm from {label} goal '
            f'(limit {CONTROLLER_GOAL_TOLERANCE_M * 1000.0:.1f} mm)')


def assess_grasp(result, min_position_delta_pct, min_current_pct):
    """Return whether measured gripper behavior indicates blocked closure."""
    if not result.completed:
        return False, 'gripper motion did not complete'
    if result.final_position_pct is None:
        return False, 'no measured final gripper position'
    if not math.isfinite(result.final_position_pct):
        return False, 'measured final gripper position is non-finite'

    position_delta = result.final_position_pct - result.target_pct
    if position_delta < min_position_delta_pct:
        return False, (
            f'jaws stopped only {position_delta:.1f} percentage points above '
            f'the {result.target_pct}% target; required '
            f'{min_position_delta_pct:.1f}')

    if min_current_pct > 0.0:
        if result.peak_current_pct is None:
            return False, 'no gripper-current samples were available'
        if not math.isfinite(result.peak_current_pct):
            return False, 'measured peak gripper current is non-finite'
        if result.peak_current_pct < min_current_pct:
            return False, (
                f'peak current {result.peak_current_pct:.1f}% is below required '
                f'{min_current_pct:.1f}%')

    current_text = ('unavailable' if result.peak_current_pct is None else
                    f'{result.peak_current_pct:.1f}%')
    return True, (
        f'blocked-closure delta {position_delta:.1f} percentage points; '
        f'peak current {current_text}')


def is_retryable_empty_close(result, min_position_delta_pct):
    """Return True only for a completed, measured close with no obstruction."""
    if (not result.completed or result.final_position_pct is None or
            not math.isfinite(result.final_position_pct)):
        return False
    return (result.final_position_pct - result.target_pct <
            min_position_delta_pct)


def grasp_depths(initial_depth_mm, retry_count, retry_step_mm):
    """Return the bounded sequence of grasp depths for all allowed attempts."""
    return [initial_depth_mm + attempt * retry_step_mm
            for attempt in range(retry_count + 1)]


def print_attempt_result(attempt_number, attempt_count, depth_mm, result,
                         verified, retryable, detail):
    """Emit a stable machine-readable observation for a future VLA executive."""
    if verified:
        outcome = 'grasp_verified'
    elif retryable:
        outcome = 'empty_close'
    else:
        outcome = 'ambiguous_or_fault'
    final_position = result.final_position_pct
    if final_position is not None and not math.isfinite(final_position):
        final_position = None
    peak_current = result.peak_current_pct
    if peak_current is not None and not math.isfinite(peak_current):
        peak_current = None
    payload = {
        'schema': 'fr5.grasp_attempt.v1',
        'attempt': attempt_number,
        'attempts_allowed': attempt_count,
        'action': {
            'grasp_depth_mm': round(float(depth_mm), 3),
            'grasp_close_pct': result.target_pct,
        },
        'observation': {
            'motion_completed': bool(result.completed),
            'final_position_pct': final_position,
            'peak_current_pct': peak_current,
            'sample_count': result.sample_count,
        },
        'outcome': outcome,
        'detail': detail,
    }
    print('GRASP_ATTEMPT_RESULT ' + json.dumps(payload, sort_keys=True))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--target-file', type=Path, default=DEFAULT_TARGET,
                        help=f'fresh JSON from b3_pick_point.py '
                             f'(default: {DEFAULT_TARGET})')
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION,
                        help=f'accepted B2 JSON (default: {DEFAULT_CALIBRATION})')
    parser.add_argument('--plans-db', type=Path, default=DEFAULT_PLANS_DB,
                        help=f'SQLite database containing the proven paths '
                             f'(default: {DEFAULT_PLANS_DB})')
    parser.add_argument('--side', choices=('auto', 'left', 'right'),
                        default='auto',
                        help='DB choreography side (default: nearest DB grab '
                             'endpoint)')
    parser.add_argument('--grasp-depth-mm', type=float,
                        default=DEFAULT_GRASP_DEPTH_MM,
                        help='distance below the clicked surface point for the '
                             f'final TCP target (default: '
                             f'{DEFAULT_GRASP_DEPTH_MM:g} mm; 0 disables)')
    parser.add_argument('--grasp-close-pct', type=int,
                        default=DEFAULT_GRASP_CLOSE_PCT,
                        help='commanded jaw target after descent; 0 is closed '
                             f'and 100 is open (default: '
                             f'{DEFAULT_GRASP_CLOSE_PCT})')
    parser.add_argument('--min-grasp-position-delta-pct', type=float,
                        default=DEFAULT_MIN_GRASP_POSITION_DELTA_PCT,
                        help='minimum actual-minus-commanded jaw percentage '
                             'indicating blocked closure (default: '
                             f'{DEFAULT_MIN_GRASP_POSITION_DELTA_PCT:g})')
    parser.add_argument('--min-grasp-current-pct', type=float,
                        default=DEFAULT_MIN_GRASP_CURRENT_PCT,
                        help='optional minimum peak motor current; 0 logs '
                             'current without gating (default: 0)')
    parser.add_argument('--grasp-retries', type=int,
                        default=DEFAULT_GRASP_RETRIES,
                        help='number of deeper retries after a verified empty '
                             f'close (default: {DEFAULT_GRASP_RETRIES}; max: '
                             f'{MAX_GRASP_RETRIES})')
    parser.add_argument('--retry-step-mm', type=float,
                        default=DEFAULT_RETRY_STEP_MM,
                        help='additional depth for each retry (default: '
                             f'{DEFAULT_RETRY_STEP_MM:g} mm; max: '
                             f'{MAX_RETRY_STEP_MM:g} mm)')
    parser.add_argument('--scale', type=float, default=MAX_SCALE,
                        help=f'arm velocity/acceleration scale, max {MAX_SCALE}')
    parser.add_argument('--max-target-age-sec', type=float,
                        default=MAX_TARGET_AGE_S,
                        help=f'refuse older clicks (default: {MAX_TARGET_AGE_S:g})')
    parser.add_argument('--allow-stale-target', action='store_true',
                        help='allow an old target for plan-only testing; '
                             'forbidden with --execute')
    parser.add_argument('--execute', action='store_true',
                        help='move the real robot; default plans only')
    parser.add_argument('--confirm-ungated-grab', action='store_true',
                        help='required with --execute: acknowledge that C1-C3 '
                             'floor/object/bin safety checks are absent')
    args = parser.parse_args(argv)
    if args.scale <= 0.0 or args.scale > MAX_SCALE:
        parser.error(f'--scale must be in (0, {MAX_SCALE}]')
    if (not math.isfinite(args.grasp_depth_mm) or
            args.grasp_depth_mm < 0.0 or
            args.grasp_depth_mm > MAX_GRASP_DEPTH_MM):
        parser.error('--grasp-depth-mm must be finite and in '
                     f'[0, {MAX_GRASP_DEPTH_MM:g}]')
    if args.grasp_close_pct < 0 or args.grasp_close_pct >= OPEN_PCT:
        parser.error('--grasp-close-pct must be an integer in [0, 99]')
    if (not math.isfinite(args.min_grasp_position_delta_pct) or
            args.min_grasp_position_delta_pct <= 0.0 or
            args.grasp_close_pct + args.min_grasp_position_delta_pct >
            OPEN_PCT):
        parser.error('--min-grasp-position-delta-pct must be finite, positive, '
                     'and keep close target + delta <= 100')
    if (not math.isfinite(args.min_grasp_current_pct) or
            args.min_grasp_current_pct < 0.0 or
            args.min_grasp_current_pct > 100.0):
        parser.error('--min-grasp-current-pct must be finite and in [0, 100]')
    if args.grasp_retries < 0 or args.grasp_retries > MAX_GRASP_RETRIES:
        parser.error(f'--grasp-retries must be in [0, {MAX_GRASP_RETRIES}]')
    if (not math.isfinite(args.retry_step_mm) or
            args.retry_step_mm <= 0.0 or
            args.retry_step_mm > MAX_RETRY_STEP_MM):
        parser.error('--retry-step-mm must be finite and in '
                     f'(0, {MAX_RETRY_STEP_MM:g}]')
    deepest_depth_mm = (
        args.grasp_depth_mm + args.grasp_retries * args.retry_step_mm)
    if deepest_depth_mm > MAX_GRASP_DEPTH_MM:
        parser.error('initial depth plus retries reaches '
                     f'{deepest_depth_mm:g} mm; maximum allowed grasp depth is '
                     f'{MAX_GRASP_DEPTH_MM:g} mm')
    if args.max_target_age_sec <= 0.0 or args.max_target_age_sec > 3600.0:
        parser.error('--max-target-age-sec must be in (0, 3600]')
    if args.execute and not args.confirm_ungated_grab:
        parser.error('--execute requires --confirm-ungated-grab')
    if args.execute and args.allow_stale_target:
        parser.error('--allow-stale-target cannot be used with --execute')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        calibration_path = args.calibration.expanduser()
        target_path = args.target_file.expanduser()
        plans_db_path = args.plans_db.expanduser()
        calibration, expected_t, expected_q, base_points = load_calibration(
            calibration_path)
        surface = load_target_file(target_path, calibration)
        check_target_envelope(surface, base_points)
        target_age_s = check_target_freshness(
            target_path, args.max_target_age_sec, args.allow_stale_target)
        sides_to_load = ('left', 'right') if args.side == 'auto' else (args.side,)
        plans_by_side = {
            side: load_side_plans(plans_db_path, side)
            for side in sides_to_load
        }
    except HoverError as exc:
        print(f'POINT GRAB REFUSED: {exc}', file=sys.stderr)
        return 2

    attempt_depths_mm = grasp_depths(
        args.grasp_depth_mm, args.grasp_retries, args.retry_step_mm)
    grasp_targets = []
    for depth_mm in attempt_depths_mm:
        target = surface.copy()
        target[2] -= depth_mm / 1000.0
        grasp_targets.append(target)
    grasp = grasp_targets[0]
    hover = surface + np.asarray([0.0, 0.0, HOVER_M])
    print('\n=== EXPERIMENTAL CLICKED-POINT GRAB ===')
    print('selected surface point [base_link, m]: '
          + ' '.join(f'{value:+.6f}' for value in surface))
    print(f'grasp depth correction: {args.grasp_depth_mm:.1f} mm downward')
    print('offset TCP grasp point [base_link, m]: '
          + ' '.join(f'{value:+.6f}' for value in grasp))
    if args.grasp_retries:
        print('retry policy: '
              f'{args.grasp_retries} deeper retr'
              f'{"y" if args.grasp_retries == 1 else "ies"}, '
              f'{args.retry_step_mm:g} mm per retry; attempt depths '
              + ', '.join(f'{depth:g}' for depth in attempt_depths_mm)
              + ' mm')
    else:
        print('retry policy: disabled')
    print('TCP hover point [base_link, m]: '
          + ' '.join(f'{value:+.6f}' for value in hover))
    print(f'target age: {target_age_s:.1f} s')
    print(f'proven trajectory database: {plans_db_path}')
    minimum_blocked_position = (
        args.grasp_close_pct + args.min_grasp_position_delta_pct)
    current_gate = (
        'logged only' if args.min_grasp_current_pct == 0.0 else
        f'require >= {args.min_grasp_current_pct:g}%')
    print(f'gripper: open {OPEN_PCT}% -> command {args.grasp_close_pct}%')
    print('grasp verification: final position >= '
          f'{minimum_blocked_position:g}% '
          f'(blocked by >= {args.min_grasp_position_delta_pct:g} points); '
          f'peak current {current_gate}')
    print(f'mode: {"EXECUTE - ROBOT WILL MOVE" if args.execute else "PLAN ONLY"}')
    print('WARNING: no table plane, object-height, bin-wall, or environment '
          'collision guard is active.')

    rclpy.init()
    node = PointGrabNode()
    gripper = None
    try:
        translation_error, rotation_error = node.validate_live_calibration(
            expected_t, expected_q)
        print(f'live B2 TF matches JSON: {translation_error * 1000.0:.3f} mm, '
              f'{math.degrees(rotation_error):.4f} deg difference')

        current_tcp = node.actual_tcp()
        print('current TCP [base_link, m]: '
              + ' '.join(f'{value:+.6f}' for value in current_tcp))
        required_z = surface[2] + EXECUTE_CLEARANCE_M
        if current_tcp[2] < required_z:
            message = (
                f'current TCP z={current_tcp[2]:.3f} m is not at least '
                f'{EXECUTE_CLEARANCE_M * 1000.0:.0f} mm above selected '
                f'surface z={surface[2]:.3f} m')
            if args.execute:
                raise HoverError(message)
            print(f'PLAN-ONLY WARNING: {message}')

        distance, side, _reference_position, tcp_quaternion, down_angle = \
            choose_db_side(node, args.side, grasp, plans_by_side)
        plans = plans_by_side[side]
        inbound, grab_to_lift, lift_to_standby = plans
        print(f'DB choreography/orientation side: {side} '
              f'(DB grab-point XY distance {distance * 1000.0:.1f} mm, '
              f'tool-down angle {down_angle:.1f} deg)')

        wrist_hover, wrist_quaternion, tcp_offset = node.desired_wrist_pose(
            hover, tcp_quaternion)
        wrist_grasp_targets = []
        for target in grasp_targets:
            wrist_grasp, grasp_wrist_quaternion, _ = node.desired_wrist_pose(
                target, tcp_quaternion)
            if not np.allclose(wrist_quaternion, grasp_wrist_quaternion,
                               atol=1e-9):
                raise HoverError('internal wrist-orientation mismatch')
            wrist_grasp_targets.append(wrist_grasp)

        preflight(node, hover, wrist_hover, wrist_grasp_targets,
                  attempt_depths_mm, wrist_quaternion, tcp_offset, plans,
                  args.scale)
        if not args.execute:
            print('\nNO MOTION OCCURRED. Re-click if the box moved, then run:')
            print('ros2 run fr5_bringup d0_point_grab.py '
                  f'--target-file={target_path} '
                  f'--grasp-depth-mm={args.grasp_depth_mm:g} '
                  f'--grasp-close-pct={args.grasp_close_pct} '
                  f'--grasp-retries={args.grasp_retries} '
                  f'--retry-step-mm={args.retry_step_mm:g} '
                  '--execute --confirm-ungated-grab')
            return 0

        print('\nEXECUTION STARTING AT <=5%. KEEP HAND ON E-STOP.')
        gripper = GripperNode()
        if not gripper.activate():
            raise HoverError('gripper activation failed; no arm motion sent')
        if not gripper.move(OPEN_PCT):
            raise HoverError('gripper failed to open; no arm motion sent')

        node.execute_saved_plan(inbound)
        node.plan_tcp_pose(
            hover, wrist_quaternion, tcp_offset, True, args.scale,
            f'new {inbound.name} endpoint -> clicked hover')
        verify_tcp(node, hover, 'hover')

        grasp_verified = False
        grasp_detail = 'no grasp attempt completed'
        last_reopen_completed = None
        attempts_completed = 0
        attempt_count = len(grasp_targets)
        for attempt_index, (depth_mm, target, wrist_target) in enumerate(
                zip(attempt_depths_mm, grasp_targets, wrist_grasp_targets)):
            attempt_number = attempt_index + 1
            attempts_completed = attempt_number
            print(f'\nGRASP ATTEMPT {attempt_number}/{attempt_count}: '
                  f'depth={depth_mm:g} mm')
            descent_label = (
                f'attempt {attempt_number}/{attempt_count} straight descent '
                f'({depth_mm:g} mm depth)')
            descent, _ = node.compute_cartesian(
                wrist_target, wrist_quaternion, args.scale, descent_label)
            node.execute_cartesian(descent, descent_label)
            verify_tcp(node, target,
                       f'attempt {attempt_number} grasp point')

            close_result = gripper.move_measured(args.grasp_close_pct)
            grasp_verified, grasp_detail = assess_grasp(
                close_result, args.min_grasp_position_delta_pct,
                args.min_grasp_current_pct)
            retryable = is_retryable_empty_close(
                close_result, args.min_grasp_position_delta_pct)
            verdict = 'PASS' if grasp_verified else 'FAIL'
            print(f'GRASP VERIFICATION {verdict}: {grasp_detail}')
            print_attempt_result(
                attempt_number, attempt_count, depth_mm, close_result,
                grasp_verified, retryable, grasp_detail)

            last_reopen_completed = None
            if not grasp_verified:
                print('No object was verified between the fingers. Reopening '
                      'before retreat.', file=sys.stderr)
                last_reopen_completed = gripper.move(OPEN_PCT)
                if not last_reopen_completed:
                    print('WARNING: gripper failed to reopen; retreating and '
                          'disabling further retries.', file=sys.stderr)

            retreat_label = (
                f'attempt {attempt_number}/{attempt_count} straight retreat '
                'to hover')
            retreat, _ = node.compute_cartesian(
                wrist_hover, wrist_quaternion, args.scale, retreat_label)
            node.execute_cartesian(retreat, retreat_label)
            verify_tcp(node, hover,
                       f'attempt {attempt_number} retreat hover')

            if grasp_verified:
                break
            if attempt_number >= attempt_count:
                break
            if not last_reopen_completed:
                break
            if not retryable:
                print('Further retries disabled: the failure was not a clean '
                      'measured empty close.', file=sys.stderr)
                break
            print(f'RETRY RESET: returning only to DB {grab_to_lift.name} '
                  'start; standby return is not used between attempts.')
            node.plan_joint_pose(
                joint_map(grab_to_lift.start), True, args.scale,
                f'retry reset: clicked hover -> DB '
                f'{grab_to_lift.name} start')
            node.verify_saved_plan_start(grab_to_lift)
            node.plan_tcp_pose(
                hover, wrist_quaternion, tcp_offset, True, args.scale,
                f'retry re-approach: DB {grab_to_lift.name} start -> '
                'clicked hover')
            verify_tcp(node, hover,
                       f'attempt {attempt_number + 1} re-approach hover')
            print(f'RETRYING: next attempt is {args.retry_step_mm:g} mm '
                  f'deeper at {attempt_depths_mm[attempt_number]:g} mm total '
                  'depth.')

        node.plan_joint_pose(
            joint_map(grab_to_lift.start), True, args.scale,
            f'new clicked hover -> DB {grab_to_lift.name} start')
        node.execute_saved_plan(grab_to_lift)
        node.execute_saved_plan(lift_to_standby)

        if not grasp_verified:
            raise HoverError(
                'arm returned to standby, but grasp verification failed after '
                f'{attempts_completed}/{attempt_count} allowed attempt(s); '
                'last gripper reopen '
                f'{"completed" if last_reopen_completed else "failed"}; '
                f'last result: {grasp_detail}')

        held_position = gripper.position_pct()
        held_current = gripper.current_pct()
        held_current_text = ('?' if held_current is None else
                             f'{held_current:.1f}')
        if (held_position is None or
                held_position < minimum_blocked_position):
            held_position_text = ('?' if held_position is None else
                                  f'{held_position:.1f}')
            raise HoverError(
                'initial grasp passed, but hold verification at standby failed: '
                f'position={held_position_text}% (required >= '
                f'{minimum_blocked_position:g}%), '
                f'current={held_current_text}%')
        print('\nPOINT GRAB SEQUENCE PASS: verified object remains between the '
              f'fingers at standby; position={held_position:.1f}%, '
              f'current={held_current_text}%.')
    except HoverError as exc:
        print(f'POINT GRAB STOPPED: {exc}', file=sys.stderr)
        print('No further automatic recovery will be attempted. Inspect the '
              'arm and gripper state and use the e-stop if needed.',
              file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('POINT GRAB INTERRUPTED. No further commands will be sent.',
              file=sys.stderr)
        return 130
    finally:
        if gripper is not None:
            gripper.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
