#!/usr/bin/env python3
"""Task A1 (Second_plan.md): command the FR5 via MoveIt and read back the live TCP pose.

Usage (bringup must be running: ros2 launch fr5_bringup a1_bringup.launch.py [sim:=false]):

  a1_move_and_read.py --list-poses          # show named SRDF poses
  a1_move_and_read.py --pose standby        # DRY-RUN: plan only, then watch TCP
  a1_move_and_read.py --pose standby --execute   # actually move (hand on e-stop!)
  a1_move_and_read.py --watch-only          # no motion, just stream the TCP pose (jog test)

Safety defaults: dry-run unless --execute, velocity/acceleration scaling 0.1.

Frames (recorded for Milestone B): MoveIt plans in `base_link` (URDF root,
verified live via /compute_fk); TCP frame here is `wrist3_link` (flange -
fingertip TCP offset is Task A2). The script prints the model frame it got
from move_group so a mismatch is loud, not silent.
"""
import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

GROUP = 'fairino5_v6_group'
BASE_FRAME = 'base_link'
TCP_FRAME = 'wrist3_link'
ARM_JOINTS = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']


def load_named_poses():
    """Named group states from the installed SRDF -> {name: {joint: value}}."""
    srdf = Path(get_package_share_directory('fr5_bringup')) / 'config' / 'fr5.srdf'
    poses = {}
    for gs in ET.parse(srdf).getroot().iter('group_state'):
        if gs.get('group') != GROUP:
            continue
        poses[gs.get('name')] = {
            j.get('name'): float(j.get('value')) for j in gs.iter('joint')
        }
    return poses


def quat_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, s)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class A1Node(Node):
    def __init__(self):
        super().__init__('a1_move_and_read')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.joint_state = None
        self.create_subscription(JointState, 'joint_states', self._on_js, 10)

    def _on_js(self, msg):
        self.joint_state = msg

    # ---- planning frame verification -------------------------------------
    def query_model_frame(self):
        """Ask move_group for FK of the TCP with an empty frame_id; the response
        header carries the model (planning) frame."""
        cli = self.create_client(GetPositionFK, 'compute_fk')
        if not cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn('compute_fk service unavailable - is move_group up?')
            return None
        req = GetPositionFK.Request()
        req.fk_link_names = [TCP_FRAME]
        req.robot_state.is_diff = True
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        res = fut.result()
        if res is None or not res.pose_stamped:
            self.get_logger().warn('compute_fk returned nothing')
            return None
        return res.pose_stamped[0].header.frame_id

    # ---- motion -----------------------------------------------------------
    def move_to(self, joint_targets, execute, scale):
        client = ActionClient(self, MoveGroup, 'move_action')
        if not client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('move_action server unavailable - is move_group up?')
            return False

        goal = MoveGroup.Goal()
        goal.request.group_name = GROUP
        goal.request.max_velocity_scaling_factor = scale
        goal.request.max_acceleration_scaling_factor = scale
        goal.request.allowed_planning_time = 5.0
        goal.request.num_planning_attempts = 5
        goal.request.start_state.is_diff = True
        constraints = Constraints()
        for name, value in joint_targets.items():
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = value
            jc.tolerance_above = jc.tolerance_below = 0.005
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints = [constraints]
        goal.planning_options.plan_only = not execute

        mode = 'EXECUTING' if execute else 'dry-run (plan only)'
        self.get_logger().info(
            f'Sending goal [{mode}] scale={scale}: '
            + ' '.join(f'{k}={v:.4f}' for k, v in joint_targets.items()))

        fut = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        handle = fut.result()
        if handle is None or not handle.accepted:
            self.get_logger().error('Goal rejected')
            return False
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        result = res_fut.result().result

        if result.error_code.val != 1:  # 1 == SUCCESS
            self.get_logger().error(f'MoveGroup failed, error_code={result.error_code.val}')
            return False
        traj = result.planned_trajectory.joint_trajectory
        dur = 0.0
        if traj.points:
            end = traj.points[-1].time_from_start
            dur = end.sec + end.nanosec * 1e-9
        self.get_logger().info(
            f'{"Executed" if execute else "Planned"} OK: {len(traj.points)} points, {dur:.2f}s')
        return True

    # ---- readback ---------------------------------------------------------
    def watch_tcp(self, hz):
        print(f'\nStreaming TCP pose  {BASE_FRAME} -> {TCP_FRAME}  at {hz} Hz  (Ctrl-C to stop)')
        print(f'{"t":>10}  {"x":>8} {"y":>8} {"z":>8}   {"roll":>7} {"pitch":>7} {"yaw":>7}   joints (rad)')
        period = 1.0 / hz
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=period)
            try:
                tf = self.tf_buffer.lookup_transform(BASE_FRAME, TCP_FRAME, rclpy.time.Time())
            except Exception:
                continue
            t = tf.transform.translation
            q = tf.transform.rotation
            r, p, y = quat_to_rpy(q.x, q.y, q.z, q.w)
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            joints = ''
            if self.joint_state:
                jmap = dict(zip(self.joint_state.name, self.joint_state.position))
                joints = ' '.join(f'{jmap.get(j, float("nan")):+.3f}' for j in ARM_JOINTS)
            print(f'{stamp:10.2f}  {t.x:+.4f} {t.y:+.4f} {t.z:+.4f}   '
                  f'{r:+.3f} {p:+.3f} {y:+.3f}   {joints}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pose', default='standby', help='named SRDF pose (default: standby)')
    ap.add_argument('--joints', help='explicit "j1,j2,j3,j4,j5,j6" in rad (overrides --pose)')
    ap.add_argument('--execute', action='store_true',
                    help='actually move the robot (default is plan-only dry run)')
    ap.add_argument('--scale', type=float, default=0.1,
                    help='velocity/acceleration scaling (default 0.1)')
    ap.add_argument('--watch-only', action='store_true', help='skip motion, just stream TCP pose')
    ap.add_argument('--list-poses', action='store_true')
    ap.add_argument('--hz', type=float, default=5.0, help='TCP print rate (default 5)')
    args = ap.parse_args()

    poses = load_named_poses()
    if args.list_poses:
        for name, jv in poses.items():
            print(f'{name:12s} ' + ' '.join(f'{j}={v:+.4f}' for j, v in jv.items()))
        return 0

    rclpy.init()
    node = A1Node()
    try:
        frame = node.query_model_frame()
        if frame:
            print(f'MoveIt model/planning frame: {frame!r}'
                  + ('' if frame == BASE_FRAME else f'  (!!) expected {BASE_FRAME!r}'))

        if not args.watch_only:
            if args.joints:
                vals = [float(v) for v in args.joints.split(',')]
                if len(vals) != 6:
                    print('need exactly 6 joint values', file=sys.stderr)
                    return 2
                targets = dict(zip(ARM_JOINTS, vals))
            else:
                if args.pose not in poses:
                    print(f'unknown pose {args.pose!r}; available: {", ".join(poses)}',
                          file=sys.stderr)
                    return 2
                targets = poses[args.pose]
            if not node.move_to(targets, args.execute, args.scale):
                return 1

        node.watch_tcp(args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
