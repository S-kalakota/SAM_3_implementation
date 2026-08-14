#!/usr/bin/env python3
"""Task B3, step 1: click one ZED point and print a copyable robot target.

This tool NEVER moves the robot.  It opens the ZED directly, takes the median
XYZ of a 5x5 patch, applies the accepted B2 transform, and prints both the
surface point and the 100 mm hover point in ``base_link``.

Flow:
  1. Keep the arm clear of the camera view and click a marked point on a flat,
     matte surface.
  2. Inspect the locked camera/base coordinates and depth variation.
  3. Press SPACE to accept, or ``r`` to reject and click again.
  4. Copy either the printed hover-validation command or the experimental
     point-grab command into a sourced terminal.

Keys: left click = lock point, SPACE = accept, r = reject, q/Esc = quit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

# Match B1's interpreter-path handling: pyzed is in the user site while the
# system OpenCV build requires the system NumPy ABI.
for package_path in ('/usr/lib/python3.12/dist-packages',
                     '/usr/lib/python3/dist-packages'):
    if os.path.isdir(package_path):
        if package_path in sys.path:
            sys.path.remove(package_path)
        sys.path.insert(0, package_path)

import cv2
import numpy as np
import pyzed.sl as sl


DEFAULT_CALIB_DIR = Path.home() / 'VLA_Model_Work' / 'robot_ws' / 'calib'
DEFAULT_CALIBRATION = DEFAULT_CALIB_DIR / 'T_base_cam.json'
DEFAULT_OUT = Path('/tmp/fr5_b3_target.json')
WINDOW = 'B3 pick validation point (NO ROBOT MOTION)'
PATCH = 5
MIN_VALID_PIXELS = 20
MAX_DEPTH_STD_MM = 10.0
CALIBRATION_MARGIN_M = 0.075
HOVER_M = 0.100

DEPTH_MODES = {
    'quality': sl.DEPTH_MODE.QUALITY,
    'ultra': sl.DEPTH_MODE.ULTRA,
    'neural': sl.DEPTH_MODE.NEURAL,
    'neural_light': sl.DEPTH_MODE.NEURAL_LIGHT,
    'neural_plus': sl.DEPTH_MODE.NEURAL_PLUS,
}


class PickError(RuntimeError):
    pass


def load_accepted_calibration(path: Path):
    try:
        calibration = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise PickError(f'cannot load B2 calibration {path}: {exc}') from exc
    if calibration.get('schema_version') != 1:
        raise PickError(f'unsupported B2 schema in {path}')
    if not calibration.get('quality', {}).get('passed', False):
        raise PickError(f'B2 calibration did not pass quality gates: {path}')
    if calibration.get('parent_frame') != 'base_link':
        raise PickError('B2 parent frame is not base_link')
    if calibration.get('child_frame') != 'zed_left_optical':
        raise PickError('B2 child frame is not zed_left_optical')

    rotation = np.asarray(calibration.get('R'), dtype=float)
    translation = np.asarray(calibration.get('t_m'), dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise PickError('B2 R/t have invalid dimensions')
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise PickError('B2 R/t contain non-finite values')

    source_path = Path(calibration.get('source_capture', '')).expanduser()
    if not source_path.is_file():
        sibling = path.with_name('calib_points.json')
        source_path = sibling if sibling.is_file() else source_path
    try:
        source_raw = source_path.read_bytes()
        capture = json.loads(source_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PickError(f'cannot load B1 source capture {source_path}: {exc}') from exc
    expected_hash = calibration.get('source_sha256')
    actual_hash = hashlib.sha256(source_raw).hexdigest()
    if not expected_hash or actual_hash != expected_hash:
        raise PickError(
            'B1 points changed after B2 was solved; rerun b2_fit_transform.py --write')
    pairs = capture.get('pairs')
    if not isinstance(pairs, list) or len(pairs) < 8:
        raise PickError('B1 source does not contain at least eight pairs')

    camera_points = np.asarray([pair['cam_xyz'] for pair in pairs], dtype=float)
    base_points = np.asarray([pair['base_xyz'] for pair in pairs], dtype=float)
    return calibration, capture, rotation, translation, camera_points, base_points


def patch_median_xyz(xyz_map, u, v):
    xyz = np.asarray(xyz_map)[..., :3]
    radius = PATCH // 2
    y0, y1 = max(0, v - radius), min(xyz.shape[0], v + radius + 1)
    x0, x1 = max(0, u - radius), min(xyz.shape[1], u + radius + 1)
    patch = xyz[y0:y1, x0:x1].reshape(-1, 3)
    valid = patch[np.isfinite(patch).all(axis=1) & (patch[:, 2] > 0.0)]
    if not len(valid):
        return None, None, 0
    return np.median(valid, axis=0), valid.std(axis=0), len(valid)


def envelope_error(point, reference_points, label):
    lower = reference_points.min(axis=0) - CALIBRATION_MARGIN_M
    upper = reference_points.max(axis=0) + CALIBRATION_MARGIN_M
    if np.any(point < lower) or np.any(point > upper):
        return (f'{label} is outside the calibrated envelope (+/- '
                f'{CALIBRATION_MARGIN_M * 1000:.0f} mm margin): '
                f'allowed {np.round(lower, 3).tolist()} .. '
                f'{np.round(upper, 3).tolist()}')
    return None


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def accepted_record(calibration_path, calibration, pixel, camera_xyz,
                    camera_std, valid_pixels, base_xyz):
    hover = base_xyz + np.asarray([0.0, 0.0, HOVER_M])
    return {
        'schema_version': 1,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'purpose': 'B3 100 mm hover validation target',
        'calibration_file': str(calibration_path.resolve()),
        'calibration_created': calibration.get('created'),
        'calibration_source_sha256': calibration.get('source_sha256'),
        'camera_frame': calibration['child_frame'],
        'base_frame': calibration['parent_frame'],
        'pixel': list(pixel),
        'camera_xyz_m': camera_xyz.tolist(),
        'camera_patch_std_m': camera_std.tolist(),
        'valid_patch_pixels': int(valid_pixels),
        'base_surface_xyz_m': base_xyz.tolist(),
        'hover_offset_m': HOVER_M,
        'base_hover_xyz_m': hover.tolist(),
    }


def print_commands(record, out_path):
    target = ','.join(f'{value:+.6f}' for value in record['base_surface_xyz_m'])
    target_file = shlex.quote(str(out_path))
    print('\n=== B3 POINT ACCEPTED (NO ROBOT MOTION OCCURRED) ===')
    print(f"pixel:       {record['pixel']}")
    print('camera XYZ:  ' + ' '.join(
        f'{value:+.6f}' for value in record['camera_xyz_m']) + ' m')
    print('base surface:' + ' '.join(
        f' {value:+.6f}' for value in record['base_surface_xyz_m']) + ' m')
    print('base hover:  ' + ' '.join(
        f'{value:+.6f}' for value in record['base_hover_xyz_m']) + ' m')
    print(f'saved target: {out_path}')
    print('\n1) COPY/RUN THIS FIRST (PLAN ONLY - CANNOT MOVE):')
    print(f'ros2 run fr5_bringup b3_hover.py --target-file={target_file}')
    print('\n2) ONLY AFTER THE PLAN SUCCEEDS, ARM IS CLEAR, AND E-STOP IS READY:')
    print(f'ros2 run fr5_bringup b3_hover.py --target-file={target_file} --execute')
    print('\nRaw copyable base surface coordinate:')
    print(target)
    print('Equivalent plan-only command using the coordinate directly:')
    print(f'ros2 run fr5_bringup b3_hover.py --target={target}')
    print('\nEXPERIMENTAL POINT GRAB (skips the C1/C2/C3 safety gate):')
    print('1) Plan the complete grab/retreat/standby sequence:')
    print(f'ros2 run fr5_bringup d0_point_grab.py --target-file={target_file}')
    print('2) Only after reviewing that plan and clearing the full path:')
    print('ros2 run fr5_bringup d0_point_grab.py '
          f'--target-file={target_file} --execute --confirm-ungated-grab')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION,
                        help=f'accepted B2 JSON (default: {DEFAULT_CALIBRATION})')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help=f'accepted target JSON (default: {DEFAULT_OUT})')
    parser.add_argument('--max-depth-std-mm', type=float,
                        default=MAX_DEPTH_STD_MM,
                        help=f'reject noisier depth patches (default: {MAX_DEPTH_STD_MM})')
    args = parser.parse_args(argv)
    if args.max_depth_std_mm <= 0.0 or args.max_depth_std_mm > 15.0:
        parser.error('--max-depth-std-mm must be in (0, 15]')
    return args


def main(argv=None):
    args = parse_args(argv)
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        print('B3 ERROR: a GUI display is required to click a point.', file=sys.stderr)
        return 2
    try:
        loaded = load_accepted_calibration(args.calibration.expanduser())
    except PickError as exc:
        print(f'B3 ERROR: {exc}', file=sys.stderr)
        return 2
    calibration, capture, rotation, translation, camera_points, base_points = loaded

    resolution = capture.get('resolution', 'hd720')
    depth_mode = capture.get('depth_mode', 'neural')
    if resolution not in ('hd720', 'hd1080') or depth_mode not in DEPTH_MODES:
        print('B3 ERROR: unsupported resolution/depth mode in B1 source', file=sys.stderr)
        return 2

    init = sl.InitParameters()
    init.camera_resolution = {
        'hd720': sl.RESOLUTION.HD720,
        'hd1080': sl.RESOLUTION.HD1080,
    }[resolution]
    init.depth_mode = DEPTH_MODES[depth_mode]
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(f'B3 ERROR: could not open ZED ({status!r}); another process may own it.',
              file=sys.stderr)
        return 2

    runtime = sl.RuntimeParameters()
    left_mat, xyz_mat = sl.Mat(), sl.Mat()
    mouse = {'position': None, 'click': None}
    pending = None
    accepted = None
    status_message = 'Arm clear; click a marked point on a flat surface.'

    def on_mouse(event, x, y, _flags, _parameter):
        mouse['position'] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse['click'] = (x, y)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print('B3 picker started. This process has no robot-motion interface.')
    print(f'Using B2 RMS {calibration["quality"]["rms_residual_mm"]:.3f} mm, '
          f'{resolution}, depth mode {depth_mode}.')

    try:
        while True:
            if zed.grab(runtime) > sl.ERROR_CODE.SUCCESS:
                continue
            zed.retrieve_image(left_mat, sl.VIEW.LEFT_BGR)
            zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ, sl.MEM.CPU)
            frame = left_mat.get_data(sl.MEM.CPU)[..., :3].copy()
            xyz_map = xyz_mat.get_data(sl.MEM.CPU)

            if mouse['click'] is not None and pending is None:
                pixel = mouse['click']
                camera_xyz, camera_std, valid_pixels = patch_median_xyz(
                    xyz_map, *pixel)
                if camera_xyz is None or valid_pixels < MIN_VALID_PIXELS:
                    status_message = (
                        f'Rejected: only {valid_pixels}/25 valid depth pixels.')
                elif camera_std[2] * 1000.0 > args.max_depth_std_mm:
                    status_message = (
                        f'Rejected: depth std {camera_std[2] * 1000:.1f} mm > '
                        f'{args.max_depth_std_mm:.1f} mm. Avoid edges.')
                elif envelope_error(camera_xyz, camera_points, 'Camera point'):
                    status_message = envelope_error(
                        camera_xyz, camera_points, 'Camera point')
                else:
                    base_xyz = rotation @ camera_xyz + translation
                    error = envelope_error(base_xyz, base_points, 'Base point')
                    if error:
                        status_message = error
                    else:
                        pending = accepted_record(
                            args.calibration.expanduser(), calibration, pixel,
                            camera_xyz, camera_std, valid_pixels, base_xyz)
                        status_message = (
                            f'Locked base [{base_xyz[0]:+.3f} {base_xyz[1]:+.3f} '
                            f'{base_xyz[2]:+.3f}] m, depth std '
                            f'{camera_std[2] * 1000:.1f} mm. SPACE accepts; r rejects.')
                        print(status_message)
                mouse['click'] = None

            if pending is not None:
                cv2.drawMarker(frame, tuple(pending['pixel']), (0, 0, 255),
                               cv2.MARKER_CROSS, 30, 2)
            if mouse['position'] is not None:
                hover_xyz, _, valid = patch_median_xyz(
                    xyz_map, *mouse['position'])
                u, v = mouse['position']
                hover_text = (f'{hover_xyz[2]:.3f} m' if hover_xyz is not None
                              and valid >= MIN_VALID_PIXELS else 'no clean depth')
                cv2.putText(frame, hover_text, (u + 12, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                            cv2.LINE_AA)

            lines = [
                'B3 PICKER ONLY - THIS WINDOW CANNOT MOVE THE ROBOT',
                status_message,
                'click lock   SPACE accept+exit   r reject   q quit',
            ]
            for index, text in enumerate(lines):
                y = 28 + index * 27
                cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('r') and pending is not None:
                pending = None
                status_message = 'Rejected. Arm clear; click another flat point.'
            elif key == ord(' ') and pending is not None:
                accepted = pending
                atomic_write_json(args.out.expanduser(), accepted)
                break
    except KeyboardInterrupt:
        pass
    finally:
        zed.close()
        cv2.destroyAllWindows()

    if accepted is None:
        print('No B3 point accepted; no target file was written.')
        return 1
    print_commands(accepted, args.out.expanduser())
    return 0


if __name__ == '__main__':
    sys.exit(main())
