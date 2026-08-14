#!/usr/bin/env python3
"""Task B1 (Second_plan.md, Milestone B): touch-point capture for hand-eye calibration.

Collects pixel-clicked camera-frame XYZ points paired with robot fingertip
(TCP) touches of the same physical point, and appends them to
``calib_points.json`` for the B2 rigid-transform fit.

Per pair, the flow is CLICK -> TOUCH -> SPACE:
  1. CLICK the target point in the live ZED left view (do this while the arm
     is clear of the point). The camera XYZ is the per-axis median of a 5x5
     patch of MEASURE.XYZ around the clicked pixel.
  2. Jog the arm (pendant) until the fingertip touches the same physical spot.
  3. Press SPACE in the viewer window: the TCP position (TF base_link ->
     tcp_link, median of ~12 samples) is recorded and the pair is saved.

Keys:  SPACE record TCP + save pair   r cancel pending click
       u undo last saved pair         q / Esc quit (prints summary)

Requires: bringup running (for TF), ZED free (stop the mask daemon / any other
ZED process first - the camera is single-open). This tool never moves the
robot; all motion is pendant jogging.

Collect 8-12 pairs spread across the whole workspace, INCLUDING different
heights (put a block under some touches - coplanar points make the B2 fit
degenerate in Z). The summary warns if the height spread looks too flat.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Same interpreter-path shuffle as ZED_2i_Depth/zed_depth_viewer.py: the user
# site has pyzed + NumPy 2.x, but the system OpenCV expects system NumPy 1.x.
for package_path in ("/usr/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages"):
    if os.path.isdir(package_path):
        if package_path in sys.path:
            sys.path.remove(package_path)
        sys.path.insert(0, package_path)

import cv2
import numpy as np
import pyzed.sl as sl

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

BASE_FRAME = 'base_link'
TCP_FRAME = 'tcp_link'   # A2 fingertip frame; see Next_time.md/Second_plan.md.
WINDOW = 'B1 touch-point capture'
PATCH = 5                # pixels; plan calls for a 5x5 median patch
MIN_VALID_IN_PATCH = 8   # of 25; below this the click is rejected as bad depth
TCP_SAMPLES = 12
DEFAULT_OUT = Path.home() / 'VLA_Model_Work' / 'robot_ws' / 'calib' / 'calib_points.json'

DEPTH_MODES = {
    'quality': sl.DEPTH_MODE.QUALITY,
    'ultra': sl.DEPTH_MODE.ULTRA,
    'neural': sl.DEPTH_MODE.NEURAL,
    'neural_light': sl.DEPTH_MODE.NEURAL_LIGHT,
    'neural_plus': sl.DEPTH_MODE.NEURAL_PLUS,
}


def patch_median_xyz(xyz_map, u, v):
    """Median camera-frame XYZ over a PATCH x PATCH window at pixel (u, v).

    Returns (median[3], std[3], n_valid) or (None, None, n_valid) when too few
    finite depth pixels survive.
    """
    xyz = np.asarray(xyz_map)[..., :3]
    r = PATCH // 2
    y0, y1 = max(0, v - r), min(xyz.shape[0], v + r + 1)
    x0, x1 = max(0, u - r), min(xyz.shape[1], u + r + 1)
    patch = xyz[y0:y1, x0:x1].reshape(-1, 3)
    valid = patch[np.isfinite(patch).all(axis=1) & (patch[:, 2] > 0.0)]
    if valid.shape[0] < MIN_VALID_IN_PATCH:
        return None, None, valid.shape[0]
    return (np.median(valid, axis=0).tolist(),
            valid.std(axis=0).tolist(),
            valid.shape[0])


class TcpReader(Node):
    def __init__(self):
        super().__init__('b1_capture_points')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def spin_some(self, seconds=0.0):
        rclpy.spin_once(self, timeout_sec=seconds)

    def tcp_now(self):
        try:
            tf = self.tf_buffer.lookup_transform(BASE_FRAME, TCP_FRAME, rclpy.time.Time())
        except Exception:
            return None
        t = tf.transform.translation
        return [t.x, t.y, t.z]

    def tcp_median(self, n=TCP_SAMPLES, period=0.05):
        """Median TCP position over n TF samples (~n*period seconds)."""
        samples = []
        for _ in range(n):
            self.spin_some(period)
            p = self.tcp_now()
            if p is not None:
                samples.append(p)
        if len(samples) < max(3, n // 2):
            return None, len(samples)
        arr = np.asarray(samples)
        return np.median(arr, axis=0).tolist(), len(samples)


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def load_or_new(path: Path, header):
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or 'pairs' not in data:
            raise SystemExit(f'{path} exists but is not a B1 capture file')
        print(f'Appending to existing {path} ({len(data["pairs"])} pairs)')
        return data
    return {**header, 'pairs': []}


def spread_summary(pairs):
    if len(pairs) < 2:
        return []
    lines = []
    for label, key in (('camera', 'cam_xyz'), ('base  ', 'base_xyz')):
        pts = np.asarray([p[key] for p in pairs])
        rng = pts.max(axis=0) - pts.min(axis=0)
        lines.append(f'{label} spread  x {rng[0]*1000:6.0f} mm   '
                     f'y {rng[1]*1000:6.0f} mm   z {rng[2]*1000:6.0f} mm')
    base_z = np.asarray([p['base_xyz'][2] for p in pairs])
    if base_z.max() - base_z.min() < 0.05:
        lines.append('WARNING: base-frame height spread < 50 mm - near-coplanar '
                     'points make the B2 fit degenerate in Z. Add touches on a block.')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT,
                    help=f'output JSON (default {DEFAULT_OUT})')
    ap.add_argument('--depth-mode', choices=sorted(DEPTH_MODES), default='neural',
                    help='static scene, so favor quality (default neural)')
    ap.add_argument('--resolution', default='hd720', choices=['hd720', 'hd1080'],
                    help='must match the resolution the pipeline runs at (default hd720)')
    args = ap.parse_args()

    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print('No display - this tool needs a GUI window to click pixels.', file=sys.stderr)
        return 1

    # --- camera ------------------------------------------------------------
    init = sl.InitParameters()
    init.camera_resolution = {'hd720': sl.RESOLUTION.HD720,
                              'hd1080': sl.RESOLUTION.HD1080}[args.resolution]
    init.depth_mode = DEPTH_MODES[args.depth_mode]
    init.coordinate_units = sl.UNIT.METER
    # IMAGE = X right, Y down, Z out of the lens - the optical-frame convention
    # every camera-frame number in this project already uses (Second_plan.md).
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f'Could not open ZED: {repr(status)} - is the mask daemon (or another '
              'process) holding the camera?', file=sys.stderr)
        return 1

    cam_info = zed.get_camera_information()
    lc = cam_info.camera_configuration.calibration_parameters.left_cam
    header = {
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'camera_frame': 'zed_left_optical (IMAGE coords: X right, Y down, Z out)',
        'robot_frame': BASE_FRAME,
        'tcp_frame': TCP_FRAME,
        'resolution': args.resolution,
        'depth_mode': args.depth_mode,
        'intrinsics': {'fx': lc.fx, 'fy': lc.fy, 'cx': lc.cx, 'cy': lc.cy},
    }

    # --- ROS / TF ----------------------------------------------------------
    rclpy.init()
    node = TcpReader()

    data = load_or_new(args.out, header)
    pending = None            # {'pixel': [u, v], 'cam_xyz': ..., 'cam_xyz_std': ...}
    mouse = {'pos': None, 'click': None}

    def on_mouse(event, x, y, flags, _param):
        mouse['pos'] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse['click'] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)

    runtime = sl.RuntimeParameters()
    left_mat, xyz_mat = sl.Mat(), sl.Mat()
    status_msg = 'Click the target point (arm clear of it).'

    print(__doc__)
    try:
        while True:
            node.spin_some(0.0)
            if zed.grab(runtime) > sl.ERROR_CODE.SUCCESS:
                time.sleep(0.05)
                continue
            zed.retrieve_image(left_mat, sl.VIEW.LEFT_BGR)
            zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
            frame = left_mat.get_data(sl.MEM.CPU)[..., :3].copy()
            xyz_map = xyz_mat.get_data(sl.MEM.CPU)

            # -- handle a new click ------------------------------------------
            if mouse['click'] is not None and pending is None:
                u, v = mouse['click']
                med, std, n = patch_median_xyz(xyz_map, u, v)
                if med is None:
                    status_msg = f'Bad depth at ({u},{v}): only {n}/25 valid px - click again.'
                else:
                    pending = {'pixel': [u, v], 'cam_xyz': med, 'cam_xyz_std': std}
                    status_msg = (f'Point locked: cam XYZ [{med[0]:+.3f} {med[1]:+.3f} '
                                  f'{med[2]:+.3f}] m. Jog fingertip to it, press SPACE.')
                    print(status_msg)
            mouse['click'] = None

            # -- overlay ------------------------------------------------------
            if pending is not None:
                cv2.drawMarker(frame, tuple(pending['pixel']), (0, 0, 255),
                               cv2.MARKER_CROSS, 30, 2)
            if mouse['pos'] is not None:
                u, v = mouse['pos']
                med, _, _ = patch_median_xyz(xyz_map, u, v)
                hover = (f'({u},{v}) ' + (f'{med[2]:.3f} m' if med else 'no depth'))
                cv2.putText(frame, hover, (u + 12, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            tcp = node.tcp_now()
            tcp_txt = (f'TCP [{tcp[0]:+.3f} {tcp[1]:+.3f} {tcp[2]:+.3f}]' if tcp
                       else f'TCP: no TF {BASE_FRAME}->{TCP_FRAME} (bringup running?)')
            lines = [f'pairs saved: {len(data["pairs"])}   {tcp_txt}',
                     status_msg,
                     'SPACE save   r cancel click   u undo last   q quit']
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (16, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (16, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)

            # -- keys ----------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r') and pending is not None:
                pending = None
                status_msg = 'Click cancelled. Click the next target point.'
            elif key == ord('u') and data['pairs']:
                dropped = data['pairs'].pop()
                atomic_write_json(args.out, data)
                status_msg = f'Undid pair at pixel {dropped["pixel"]}.'
                print(status_msg)
            elif key == ord(' ') and pending is not None:
                status_msg = 'Sampling TCP...'
                base_xyz, n = node.tcp_median()
                if base_xyz is None:
                    status_msg = (f'Only {n}/{TCP_SAMPLES} TF samples - is the bringup '
                                  'running and tcp_link published? Pair NOT saved.')
                    print(status_msg)
                    continue
                pair = {**pending, 'base_xyz': base_xyz,
                        'stamp': datetime.now(timezone.utc).isoformat(timespec='seconds')}
                data['pairs'].append(pair)
                atomic_write_json(args.out, data)
                pending = None
                status_msg = (f'Pair {len(data["pairs"])} saved (TCP [{base_xyz[0]:+.3f} '
                              f'{base_xyz[1]:+.3f} {base_xyz[2]:+.3f}]). MOVE THE ARM '
                              'CLEAR, then click the next point.')
                print(status_msg)
    except KeyboardInterrupt:
        pass
    finally:
        zed.close()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

    print(f'\n{len(data["pairs"])} pairs in {args.out}')
    for line in spread_summary(data['pairs']):
        print('  ' + line)
    if len(data['pairs']) >= 8:
        print('B1 done-when satisfied (>= 8 pairs) - next: B2 fit.')
    else:
        print(f'Need >= 8 well-spread pairs for B2; have {len(data["pairs"])}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
