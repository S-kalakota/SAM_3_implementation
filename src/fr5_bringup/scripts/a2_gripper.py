#!/usr/bin/env python3
"""Task A2 (Second_plan.md): drive the DH PGC140 gripper from code.

The gripper is NOT a ros2_control axis: it hangs off the FR5 controller's
tool bus and is commanded through `fairino_remote_command_service`, which the
FairinoHardwareInterface hosts inside ros2_control_node (sharing its single
RPC session — never open a second SDK connection). Command strings and the
init sequence mirror fr5_telemetry_node in the production stack:

    SetGripperConfig(4,0,0,0)   # vendor 4 = DH/Dahuan PGC/PGI series
    ActGripper(1,1)             # activate bus index 1
    MoveGripper(1,<pct>)        # 0 = closed .. 100 = open
    GetGripperCurPosition(1)    # -> "res,fault,position_pct"
    GetGripperCurCurrent(1)     # -> "res,fault,current_pct"

Usage (bringup must be running with sim:=false — the service does not exist
on mock hardware):

  a2_gripper.py --activate            # config + activate (also implied by moves)
  a2_gripper.py --open                # MoveGripper 100
  a2_gripper.py --close               # MoveGripper 0
  a2_gripper.py --pos 40              # arbitrary percent
  a2_gripper.py --status              # one-shot position / current / fault
  a2_gripper.py --watch               # stream status at 2 Hz
  a2_gripper.py --stroke-test         # open -> close -> open with caliper prompts

The stroke test is how A2's numbers get measured: at each end it pauses so
you can caliper the jaw opening and the finger pads, then tells you exactly
what to record in config/gripper.yaml (consumed later by C2's width gate).
"""
import argparse
import math
import sys
import time
from dataclasses import dataclass

import rclpy
from rclpy.node import Node

from fairino_msgs.msg import RobotNonrtState
from fairino_msgs.srv import RemoteCmdInterface

GRIPPER_IDX = 1                 # Fairino gripper bus index (production value)
GRIPPER_VENDOR_CFG = "SetGripperConfig(4,0,0,0)"   # DH/Dahuan PGC/PGI series
SERVICE = "fairino_remote_command_service"
RPC_TIMEOUT_S = 5.0
# The Fairino bridge passes a 30 s maximum time to the vendor MoveGripper RPC.
# Its ROS service reply can therefore arrive well after an otherwise healthy
# physical move.  Wait beyond that server-side bound so a slow reply is not
# mislabeled as a rejection while the request continues running remotely.
MOVE_RPC_TIMEOUT_S = 35.0
MOVE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class GripperMotionResult:
    """Measured result of one gripper position command."""

    target_pct: int
    completed: bool
    final_position_pct: float | None
    peak_current_pct: float | None
    sample_count: int


class GripperNode(Node):
    def __init__(self):
        super().__init__('a2_gripper')
        self.cli = self.create_client(RemoteCmdInterface, SERVICE)
        self.nonrt = None
        self.create_subscription(RobotNonrtState, 'nonrt_state_data',
                                 self._on_nonrt, 10)

    def _on_nonrt(self, msg):
        self.nonrt = msg

    # ---- raw command channel ----------------------------------------------
    def call(self, cmd, timeout=RPC_TIMEOUT_S):
        """Send one command string; return raw cmd_res or None on failure."""
        if not self.cli.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f'{SERVICE} unavailable — is bringup running with sim:=false? '
                '(mock hardware has no gripper service)')
            return None
        req = RemoteCmdInterface.Request()
        req.cmd_str = cmd
        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        res = fut.result()
        if res is None:
            self.get_logger().error(
                f'timed out after {timeout:g} s: {cmd}')
            return None
        return res.cmd_res

    def call_ok(self, cmd, timeout=RPC_TIMEOUT_S):
        res = self.call(cmd, timeout)
        ok = res == '0'
        if not ok and res is not None:
            self.get_logger().error(f'{cmd} -> {res}')
        return ok

    def query_csv(self, cmd):
        """Query commands reply 'res,fault,value'; return value or None."""
        raw = self.call(cmd)
        if raw is None:
            return None
        parts = raw.split(',')
        if len(parts) != 3:
            self.get_logger().warn(f'{cmd}: unexpected reply {raw!r}')
            return None
        res, fault, value = parts
        if res != '0':
            self.get_logger().warn(f'{cmd} failed (code={res})')
            return None
        if fault != '0':
            self.get_logger().warn(f'{cmd} reports gripper fault={fault}')
            return None
        try:
            parsed = float(value)
        except ValueError:
            self.get_logger().warn(f'{cmd}: bad value {value!r}')
            return None
        if not math.isfinite(parsed):
            self.get_logger().warn(f'{cmd}: non-finite value {value!r}')
            return None
        return parsed

    # ---- gripper ops --------------------------------------------------------
    def activate(self):
        ok_cfg = self.call_ok(GRIPPER_VENDOR_CFG)
        print(f'SetGripperConfig -> {"ok" if ok_cfg else "FAILED"}')
        ok_act = self.call_ok(f'ActGripper({GRIPPER_IDX},1)')
        print(f'ActGripper       -> {"ok" if ok_act else "FAILED"}')
        return ok_cfg and ok_act

    def position_pct(self):
        return self.query_csv(f'GetGripperCurPosition({GRIPPER_IDX})')

    def current_pct(self):
        return self.query_csv(f'GetGripperCurCurrent({GRIPPER_IDX})')

    def _send_move(self, pct):
        """Send one move exactly once.

        A missing service response is an ambiguous outcome: the controller may
        still be processing the command after this client stops waiting.  Do
        not configure, activate, or resend automatically in that state.
        """
        pct = int(max(0, min(100, pct)))
        cmd = f'MoveGripper({GRIPPER_IDX},{pct})'
        response = self.call(cmd, timeout=MOVE_RPC_TIMEOUT_S)
        if response is None:
            print(
                f'{cmd} was not confirmed; command outcome is unknown. '
                'No automatic retry or gripper reactivation will be sent.',
                file=sys.stderr)
            return False
        if response != '0':
            self.get_logger().error(f'{cmd} -> {response}')
            print(
                f'{cmd} returned an error. No automatic retry or gripper '
                'reactivation will be sent.',
                file=sys.stderr)
            return False
        return True

    def move(self, pct, wait=True):
        """Command a jaw position; return whether the motion completed."""
        pct = int(max(0, min(100, pct)))
        if not self._send_move(pct):
            return False
        if not wait:
            return True
        return self._wait_motion_result(pct).completed

    def move_measured(self, pct):
        """Command a jaw position and retain position/current observations."""
        pct = int(max(0, min(100, pct)))
        if not self._send_move(pct):
            return GripperMotionResult(pct, False, None, None, 0)
        return self._wait_motion_result(pct)

    def _wait_motion_result(self, target_pct):
        """Prefer grip_motion_done from nonrt_state_data; fall back to
        position stability. Capture actual position and peak motor current."""
        grace = time.monotonic() + 0.5   # let grip_motion_done drop post-command
        deadline = time.monotonic() + MOVE_TIMEOUT_S
        last_pos, peak_current, stable, sample_count = None, None, 0, 0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.10)
            pos = self.position_pct()
            current = self.current_pct()
            sample_count += 1
            if current is not None:
                peak_current = (current if peak_current is None else
                                max(peak_current, current))
            if pos is not None:
                if last_pos is not None and abs(pos - last_pos) < 0.5:
                    stable += 1
                else:
                    stable = 0
                last_pos = pos

            motion_done = (
                time.monotonic() > grace and self.nonrt is not None
                and bool(self.nonrt.grip_motion_done))
            stable_done = (
                last_pos is not None
                and stable >= (3 if self.nonrt is None else 10))
            if motion_done or stable_done:
                mode = 'motion done' if motion_done else 'position stable'
                pos_text = '?' if last_pos is None else f'{last_pos:.1f}'
                current_text = ('?' if peak_current is None else
                                f'{peak_current:.1f}')
                print(f'done ({mode}): position={pos_text}% '
                      f'(commanded {target_pct}%), '
                      f'peak current={current_text}%')
                return GripperMotionResult(
                    target_pct, True, last_pos, peak_current, sample_count)
        print(f'TIMED OUT after {MOVE_TIMEOUT_S}s '
              f'(last position: {last_pos}, peak current: {peak_current})',
              file=sys.stderr)
        return GripperMotionResult(
            target_pct, False, last_pos, peak_current, sample_count)

    def print_status(self):
        pos = self.position_pct()
        cur = self.current_pct()
        fault = self.nonrt.gripperfaultnum if self.nonrt is not None else None
        done = bool(self.nonrt.grip_motion_done) if self.nonrt is not None else None
        print(f'position={pos}%  current={cur}%  fault={fault}  motion_done={done}')


def stroke_test(node):
    """Open -> close -> open, pausing for caliper measurements at each end."""
    print('\n=== A2 stroke test ===')
    print('Measures the numbers C2/F5/G2 need. Have calipers ready.\n')
    if not node.activate():
        return 1

    if not node.move(100):
        return 1
    print('\nJaws OPEN. Measure with calipers:')
    print('  1. jaw-to-jaw opening (inner faces of the pads)  -> max_jaw_stroke_m')
    print('  2. finger pad width and length                   -> finger_pad_*_m')
    input('Press Enter to CLOSE the jaws (keep fingers clear!)... ')

    if not node.move(0):
        return 1
    print('\nJaws CLOSED. Measure:')
    print('  3. jaw-to-jaw gap when closed (0 if pads touch)')
    input('Press Enter to re-open... ')

    if not node.move(100):
        return 1
    print('\nRecord the measurements in src/fr5_bringup/config/gripper.yaml:')
    print('  max_jaw_stroke_m   = open gap - closed gap (usable stroke)')
    print('  finger_pad_width_m / finger_pad_length_m')
    print('Sanity: boxes are only graspable if their short side < max_jaw_stroke_m.')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--activate', action='store_true')
    g.add_argument('--open', action='store_true')
    g.add_argument('--close', action='store_true')
    g.add_argument('--pos', type=int, metavar='PCT', help='0=closed .. 100=open')
    g.add_argument('--status', action='store_true')
    g.add_argument('--watch', action='store_true')
    g.add_argument('--stroke-test', action='store_true')
    ap.add_argument('--no-wait', action='store_true',
                    help='send the move and exit without waiting for motion done')
    args = ap.parse_args()

    rclpy.init()
    node = GripperNode()
    rc = 0
    try:
        # let one nonrt_state_data sample arrive so status fields are live
        t0 = time.monotonic()
        while node.nonrt is None and time.monotonic() - t0 < 1.0:
            rclpy.spin_once(node, timeout_sec=0.1)

        if args.activate:
            rc = 0 if node.activate() else 1
        elif args.open:
            rc = 0 if node.move(100, wait=not args.no_wait) else 1
        elif args.close:
            rc = 0 if node.move(0, wait=not args.no_wait) else 1
        elif args.pos is not None:
            rc = 0 if node.move(args.pos, wait=not args.no_wait) else 1
        elif args.status:
            node.print_status()
        elif args.watch:
            print('streaming gripper status at 2 Hz (Ctrl-C to stop)')
            while rclpy.ok():
                node.print_status()
                for _ in range(5):
                    rclpy.spin_once(node, timeout_sec=0.1)
        elif args.stroke_test:
            rc = stroke_test(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return rc


if __name__ == '__main__':
    sys.exit(main())
