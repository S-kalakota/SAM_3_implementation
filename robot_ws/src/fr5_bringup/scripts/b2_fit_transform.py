#!/usr/bin/env python3
"""Solve Task B2's fixed ZED-camera-to-FR5-base rigid transform.

The B1 capture file contains coordinates of the same physical points in two
frames.  This tool fits the no-scale Kabsch transform

    p_base = R_base_cam @ p_cam + t_base_cam

reports every residual, and writes ``T_base_cam.json`` only when the fit meets
the configured quality gates.  The output also contains the quaternion needed
by ``tf2_ros/static_transform_publisher``.

Examples:
  b2_fit_transform.py                         # fit/report only
  b2_fit_transform.py --write                 # validate and save the transform
  b2_fit_transform.py --input points.json --out T_base_cam.json --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_CALIB_DIR = Path(
    os.environ.get(
        'FR5_CALIB_DIR',
        Path.home() / 'VLA_Model_Work' / 'robot_ws' / 'calib',
    )
).expanduser()
DEFAULT_INPUT = DEFAULT_CALIB_DIR / 'calib_points.json'
DEFAULT_OUTPUT = DEFAULT_CALIB_DIR / 'T_base_cam.json'
MIN_POINTS = 8
DEFAULT_RMS_THRESHOLD_MM = 8.0
# The plan calls for "no single wild outlier".  Fifteen millimetres makes that
# check explicit and matches the subsequent B3 maximum hover-miss tolerance.
DEFAULT_MAX_RESIDUAL_MM = 15.0


class CalibrationError(RuntimeError):
    """Raised when the B1 input cannot produce a trustworthy B2 fit."""


def _as_points(values, label):
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise CalibrationError(f'{label} must be an Nx3 array; got {points.shape}')
    if not np.isfinite(points).all():
        raise CalibrationError(f'{label} contains NaN or infinite coordinates')
    return points


def load_capture(path: Path):
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CalibrationError(f'cannot read B1 capture file {path}: {exc}') from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CalibrationError(f'{path} is not valid JSON: {exc}') from exc

    pairs = data.get('pairs') if isinstance(data, dict) else None
    if not isinstance(pairs, list):
        raise CalibrationError(f'{path} does not contain a pairs array')
    if len(pairs) < MIN_POINTS:
        raise CalibrationError(
            f'B2 requires at least {MIN_POINTS} pairs; {path} contains {len(pairs)}')

    try:
        cam = _as_points([pair['cam_xyz'] for pair in pairs], 'camera points')
        base = _as_points([pair['base_xyz'] for pair in pairs], 'base points')
    except (KeyError, TypeError) as exc:
        raise CalibrationError(f'every pair needs cam_xyz and base_xyz: {exc}') from exc

    robot_frame = data.get('robot_frame')
    tcp_frame = data.get('tcp_frame')
    camera_description = data.get('camera_frame')
    if not isinstance(robot_frame, str) or not robot_frame:
        raise CalibrationError('capture is missing robot_frame')
    if not isinstance(tcp_frame, str) or not tcp_frame:
        raise CalibrationError('capture is missing tcp_frame')
    if not isinstance(camera_description, str) or not camera_description:
        raise CalibrationError('capture is missing camera_frame')

    # B1 records a human-readable suffix after the TF frame name.
    camera_frame = camera_description.split(maxsplit=1)[0]
    return data, pairs, cam, base, robot_frame, tcp_frame, camera_frame, raw


def fit_rigid_transform(cam_pts, base_pts):
    """Return R, t, and singular values for p_base = R @ p_cam + t."""
    cam_center = cam_pts.mean(axis=0)
    base_center = base_pts.mean(axis=0)
    cam_zero = cam_pts - cam_center
    base_zero = base_pts - base_center
    covariance = cam_zero.T @ base_zero
    U, singular_values, Vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    rotation = Vt.T @ correction @ U.T
    translation = base_center - rotation @ cam_center
    return rotation, translation, singular_values


def rotation_matrix_to_quaternion_xyzw(rotation):
    """Convert a proper 3x3 rotation matrix to a normalized ROS quaternion."""
    r = rotation
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s

    quat = np.asarray([qx, qy, qz, qw], dtype=float)
    quat /= np.linalg.norm(quat)
    # q and -q encode the same orientation; keep output deterministic.
    if quat[3] < 0.0:
        quat *= -1.0
    return quat


def build_result(path, capture, pairs, cam, base, robot_frame, tcp_frame,
                 camera_frame, source_raw, rms_threshold_mm,
                 max_residual_threshold_mm):
    rotation, translation, singular_values = fit_rigid_transform(cam, base)
    predicted = (rotation @ cam.T).T + translation
    error_vectors = predicted - base
    residuals_m = np.linalg.norm(error_vectors, axis=1)

    det_rotation = float(np.linalg.det(rotation))
    orthogonality_error = float(np.linalg.norm(rotation.T @ rotation - np.eye(3)))
    if not np.isclose(det_rotation, 1.0, atol=1e-9):
        raise CalibrationError(f'fit produced det(R)={det_rotation}, expected +1')
    if orthogonality_error > 1e-9:
        raise CalibrationError(
            f'fit produced a non-orthonormal rotation (error {orthogonality_error})')
    if singular_values[-1] <= np.finfo(float).eps * singular_values[0] * len(cam):
        raise CalibrationError(
            'point geometry is rank-deficient; add calibration points at varied heights')

    rms_mm = float(np.sqrt(np.mean(residuals_m ** 2)) * 1000.0)
    mean_mm = float(np.mean(residuals_m) * 1000.0)
    max_mm = float(np.max(residuals_m) * 1000.0)
    rms_pass = rms_mm <= rms_threshold_mm
    max_pass = max_mm <= max_residual_threshold_mm
    passed = rms_pass and max_pass
    quaternion = rotation_matrix_to_quaternion_xyzw(rotation)

    residual_records = []
    for index, (pair, prediction, vector, residual) in enumerate(
            zip(pairs, predicted, error_vectors, residuals_m), start=1):
        record = {
            'pair': index,
            'residual_m': float(residual),
            'residual_mm': float(residual * 1000.0),
            'error_base_xyz_m': vector.tolist(),
            'predicted_base_xyz_m': prediction.tolist(),
            'camera_xyz_m': cam[index - 1].tolist(),
            'measured_base_xyz_m': base[index - 1].tolist(),
        }
        for key in ('pixel', 'cam_xyz_std', 'stamp'):
            if key in pair:
                record[key] = pair[key]
        residual_records.append(record)

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    spread_cam = np.ptp(cam, axis=0)
    spread_base = np.ptp(base, axis=0)

    return {
        'schema_version': 1,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source_capture': str(path.resolve()),
        'source_capture_created': capture.get('created'),
        'source_sha256': hashlib.sha256(source_raw).hexdigest(),
        'method': 'Kabsch rigid transform (no scale)',
        'convention': 'p_base = R @ p_camera + t',
        'parent_frame': robot_frame,
        'child_frame': camera_frame,
        'touch_frame': tcp_frame,
        'R': rotation.tolist(),
        't_m': translation.tolist(),
        'T_base_camera': transform.tolist(),
        'quaternion_xyzw': quaternion.tolist(),
        'quality': {
            'passed': passed,
            'point_count': len(pairs),
            'rms_residual_mm': rms_mm,
            'mean_residual_mm': mean_mm,
            'max_residual_mm': max_mm,
            'rms_threshold_mm': rms_threshold_mm,
            'max_residual_threshold_mm': max_residual_threshold_mm,
            'rms_passed': rms_pass,
            'max_residual_passed': max_pass,
            'camera_spread_m': spread_cam.tolist(),
            'base_spread_m': spread_base.tolist(),
            'covariance_singular_values': singular_values.tolist(),
            'smallest_to_largest_singular_value': float(
                singular_values[-1] / singular_values[0]),
            'rotation_determinant': det_rotation,
            'rotation_orthogonality_error': orthogonality_error,
        },
        'residuals': residual_records,
    }


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def print_report(result):
    quality = result['quality']
    print('\n=== B2 camera-to-base rigid fit ===')
    print(f"points:  {quality['point_count']}")
    print(f"frames:  {result['parent_frame']} -> {result['child_frame']}")
    print(f"RMS:     {quality['rms_residual_mm']:.3f} mm "
          f"(limit {quality['rms_threshold_mm']:.3f} mm)")
    print(f"mean:    {quality['mean_residual_mm']:.3f} mm")
    print(f"maximum: {quality['max_residual_mm']:.3f} mm "
          f"(limit {quality['max_residual_threshold_mm']:.3f} mm)")
    print('\nper-point residuals:')
    for record in result['residuals']:
        print(f"  pair {record['pair']:2d}: {record['residual_mm']:7.3f} mm")
    print('\nR_base_camera:')
    for row in result['R']:
        print('  ' + ' '.join(f'{value:+.9f}' for value in row))
    print('t_base_camera [m]:  ' +
          ' '.join(f'{value:+.9f}' for value in result['t_m']))
    print('quaternion [x y z w]: ' +
          ' '.join(f'{value:+.9f}' for value in result['quaternion_xyzw']))
    print('\nB2 QUALITY: ' + ('PASS' if quality['passed'] else 'FAIL'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help=f'B1 capture JSON (default: {DEFAULT_INPUT})')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT,
                        help=f'B2 transform JSON (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--rms-threshold-mm', type=float,
                        default=DEFAULT_RMS_THRESHOLD_MM)
    parser.add_argument('--max-residual-mm', type=float,
                        default=DEFAULT_MAX_RESIDUAL_MM)
    parser.add_argument('--write', action='store_true',
                        help='atomically write --out, but only when quality passes')
    args = parser.parse_args(argv)
    if args.rms_threshold_mm <= 0.0 or args.max_residual_mm <= 0.0:
        parser.error('quality thresholds must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        loaded = load_capture(args.input.expanduser())
        result = build_result(
            args.input.expanduser(), *loaded[:-1], loaded[-1],
            args.rms_threshold_mm, args.max_residual_mm)
    except CalibrationError as exc:
        print(f'B2 ERROR: {exc}', file=sys.stderr)
        return 2

    print_report(result)
    if args.write:
        if not result['quality']['passed']:
            print(f'REFUSED to write {args.out}: quality gates failed.', file=sys.stderr)
            return 2
        atomic_write_json(args.out.expanduser(), result)
        print(f'Wrote accepted transform: {args.out.expanduser()}')
    else:
        print('(dry run - pass --write to save T_base_cam.json)')
    return 0 if result['quality']['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
