#!/usr/bin/env python3
"""Task A2 (Second_plan.md): solve + verify the fingertip TCP offset.

Pivot calibration: touch ONE fixed physical point (a sharp corner, a bolt
head) with the closed fingertip from several DIFFERENT wrist orientations.
For each touch i the flange pose (R_i, p_i) is read from TF
`base_link -> wrist3_link`. The fingertip offset d (constant in the flange
frame) and the touched point c (constant in base) satisfy

    R_i @ d + p_i = c        for every touch

which stacks into the linear system [R_i  -I] [d; c] = -p_i, solved by least
squares. Residuals tell you how good the touches were.

Usage (bringup running, sim:=false; jog with the pendant between touches):

  a2_tcp_calibrate.py                     # capture 4 touches, solve, print d
  a2_tcp_calibrate.py --samples 6 --write # solve and update tcp_offset.yaml
  a2_tcp_calibrate.py --verify            # after relaunch: 2-orientation touch
                                          # test against tcp_link (<= 6 mm)

After --write: relaunch bringup so the URDF picks up the new offset, then run
--verify. (A non-symlink install must be rebuilt first.) Done-when (plan): the
two-orientation verify agrees within 6 mm.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

BASE_FRAME = 'base_link'
FLANGE_FRAME = 'wrist3_link'
TCP_FRAME = 'tcp_link'
VERIFY_TOLERANCE_M = 0.006
VERIFY_TOLERANCE_MM = VERIFY_TOLERANCE_M * 1000.0
# default yaml target: source tree (share/ is a symlink to it under
# --symlink-install, but writing to src is unambiguous)
DEFAULT_YAML = Path(__file__).resolve().parents[1] / 'config' / 'tcp_offset.yaml'


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class TouchNode(Node):
    def __init__(self):
        super().__init__('a2_tcp_calibrate')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def read_pose(self, child_frame, settle_s=0.5, timeout_s=5.0):
        """Return (R, p) of base_link -> child_frame, or None."""
        # spin briefly so TF is fresh (arm should be stationary during a touch)
        end = self.get_clock().now().nanoseconds / 1e9 + settle_s
        while self.get_clock().now().nanoseconds / 1e9 < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        deadline = self.get_clock().now().nanoseconds / 1e9 + timeout_s
        while self.get_clock().now().nanoseconds / 1e9 < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(BASE_FRAME, child_frame,
                                                     rclpy.time.Time())
                t, q = tf.transform.translation, tf.transform.rotation
                return (quat_to_matrix(q.x, q.y, q.z, q.w),
                        np.array([t.x, t.y, t.z]))
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
        return None


def rotation_diversity_deg(rotations):
    """Largest pairwise rotation angle — small values make the fit degenerate."""
    worst = 0.0
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            rel = rotations[i].T @ rotations[j]
            angle = math.degrees(math.acos(
                max(-1.0, min(1.0, (np.trace(rel) - 1.0) / 2.0))))
            worst = max(worst, angle)
    return worst


def solve_pivot(rotations, positions):
    """Least-squares d (flange-frame tip offset) and c (fixed point, base)."""
    n = len(rotations)
    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i, (R, p) in enumerate(zip(rotations, positions)):
        A[3 * i:3 * i + 3, 0:3] = R
        A[3 * i:3 * i + 3, 3:6] = -np.eye(3)
        b[3 * i:3 * i + 3] = -p
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    d, c = x[:3], x[3:]
    residuals = np.array([np.linalg.norm(R @ d + p - c)
                          for R, p in zip(rotations, positions)])
    return d, c, residuals


def capture(node, n_samples, frame):
    rotations, positions = [], []
    print(f'\nTouch the SAME fixed point {n_samples} times, each from a '
          f'clearly different wrist orientation.\n'
          f'Reading TF {BASE_FRAME} -> {frame}. Ctrl-C to abort.\n')
    for i in range(n_samples):
        input(f'[{i + 1}/{n_samples}] Jog the fingertip onto the point, '
              f'let the arm settle, then press Enter... ')
        pose = node.read_pose(frame)
        if pose is None:
            print(f'  no TF {BASE_FRAME} -> {frame} — is bringup running '
                  f'(and, for {TCP_FRAME}, was the URDF rebuilt)?',
                  file=sys.stderr)
            return None, None
        R, p = pose
        rotations.append(R)
        positions.append(p)
        print(f'  recorded p = [{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}]')
    return rotations, positions


def write_yaml(path, d, residuals):
    path = Path(path)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    path.write_text(
        '# Fingertip-center TCP offset in the wrist3_link (flange) frame, metres.\n'
        '# Solved by a2_tcp_calibrate.py (pivot calibration). Relaunch bringup\n'
        '# after editing; rebuild first only for a non-symlink install.\n'
        f'# project TCP verification tolerance: {VERIFY_TOLERANCE_M:.3f} m\n'
        f'# fit: {len(residuals)} touches, RMS residual {rms * 1000:.1f} mm\n'
        f'x: {d[0]:.4f}\n'
        f'y: {d[1]:.4f}\n'
        f'z: {d[2]:.4f}\n'
    )
    print(f'wrote {path}')


def run_calibrate(node, args):
    rotations, positions = capture(node, args.samples, FLANGE_FRAME)
    if rotations is None:
        return 1

    # Degeneracy guard 1: the FLANGE must physically travel between touches.
    # If the fingertip stays on the point while the wrist tilts, the flange
    # swings by several cm; near-zero spread means the arm never really moved
    # (or only spun in place) and the fit would be garbage with a perfect-
    # looking residual.
    pts = np.array(positions)
    span = max(np.linalg.norm(pts[i] - pts[j])
               for i in range(len(pts)) for j in range(i + 1, len(pts)))
    if span < 0.03:
        print(f'\nERROR: the wrist moved only {span * 1000:.1f} mm across all '
              'touches — the poses are effectively identical, so the offset '
              'cannot be solved.\nTilt the whole arm between touches: the '
              'fingertip stays on the point, but the wrist should land '
              '5-10 cm away from its previous spot each time. Nothing written.')
        return 1

    diversity = rotation_diversity_deg(rotations)
    if diversity < 20.0:
        print(f'\nWARNING: wrist orientations only span {diversity:.0f} deg — '
              'the fit is near-degenerate. Redo with more varied orientations.')

    d, c, residuals = solve_pivot(rotations, positions)

    # Degeneracy guard 2: the fingertip is physically IN FRONT of the flange
    # (a closed PGC140 puts it very roughly 10-20 cm out along flange +z).
    # An implausible z means degenerate touches, not a real measurement.
    if not (0.03 < d[2] < 0.35):
        print(f'\nERROR: solved z offset {d[2] * 1000:+.1f} mm is physically '
              'implausible (expected roughly +100..+200 mm in front of the '
              'flange). The touch set is degenerate — redo with clearly '
              'different wrist tilts. Nothing written.')
        return 1
    print('\n=== pivot calibration result ===')
    print(f'TCP offset d (flange frame, m): '
          f'[{d[0]:+.4f} {d[1]:+.4f} {d[2]:+.4f}]  |d| = {np.linalg.norm(d):.4f}')
    print(f'touched point c (base frame, m): '
          f'[{c[0]:+.4f} {c[1]:+.4f} {c[2]:+.4f}]')
    print(f'residuals per touch (mm): '
          + ' '.join(f'{r * 1000:.1f}' for r in residuals))
    rms = np.sqrt(np.mean(residuals ** 2)) * 1000
    print(f'RMS residual: {rms:.1f} mm  '
          f'({"OK" if rms <= VERIFY_TOLERANCE_MM else "high — redo the worst touches"})')

    if args.write:
        write_yaml(args.yaml, d, residuals)
        print('Now relaunch bringup, then run --verify. Rebuild first only for '
              'a non-symlink install.')
    else:
        print('(dry run — pass --write to update tcp_offset.yaml)')
    return 0


def run_verify(node, args):
    n = max(2, args.samples if args.samples != 4 else 2)
    rotations, positions = capture(node, n, TCP_FRAME)
    if rotations is None:
        return 1
    pts = np.array(positions)
    centroid = pts.mean(axis=0)
    spread = np.linalg.norm(pts - centroid, axis=1)
    worst = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            worst = max(worst, float(np.linalg.norm(pts[i] - pts[j])))
    print('\n=== TCP verify ===')
    for i, p in enumerate(pts):
        print(f'touch {i + 1}: [{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}]  '
              f'({spread[i] * 1000:.1f} mm from centroid)')
    print(f'max pairwise disagreement: {worst * 1000:.1f} mm  '
          f'-> {"PASS (<= 6 mm)" if worst <= VERIFY_TOLERANCE_M else "FAIL (> 6 mm) — refit"}')
    return 0 if worst <= VERIFY_TOLERANCE_M else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--samples', type=int, default=4,
                    help='touches to capture (default 4; >= 3 required)')
    ap.add_argument('--verify', action='store_true',
                    help=f'read {TCP_FRAME} instead and check touch agreement')
    ap.add_argument('--write', action='store_true',
                    help='write the solved offset into tcp_offset.yaml')
    ap.add_argument('--yaml', default=str(DEFAULT_YAML))
    args = ap.parse_args()

    if not args.verify and args.samples < 3:
        print('need >= 3 touches to solve the offset', file=sys.stderr)
        return 2

    rclpy.init()
    node = TouchNode()
    try:
        return run_verify(node, args) if args.verify else run_calibrate(node, args)
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
