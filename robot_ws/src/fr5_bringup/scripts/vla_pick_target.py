#!/usr/bin/env python3
"""Convert a language-selected DINO/SAM mask into a checked FR5 surface target.

This process has no robot-motion interface.  It parses a constrained pick
request with ``VLA_project``, asks the resident Grounding DINO/SAM 3.1 service
for one fresh segmentation, back-projects the selected bounding-box center
with ZED depth, and writes the schema-1 target consumed by ``b3_hover.py`` and
``d0_point_grab.py``.

The DINO service must be running on localhost from the monorepo root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from repo_paths import calibration_dir, find_repo_root


DEFAULT_PROJECT_ROOT = find_repo_root(Path(__file__))
DEFAULT_VLA_PROJECT = DEFAULT_PROJECT_ROOT / 'VLA_project'
DEFAULT_SAM_PROJECT = DEFAULT_PROJECT_ROOT
DEFAULT_CALIBRATION = calibration_dir(Path(__file__)) / 'T_base_cam.json'
DEFAULT_TARGET = Path('/tmp/fr5_vla_target.json')
DEFAULT_AUDIT = Path('/tmp/fr5_vla_target_audit.png')
DEFAULT_SERVICE_HOST = '127.0.0.1:8765'

BASE_FRAME = 'base_link'
CAMERA_FRAME = 'zed_left_optical'
HOVER_M = 0.100
CALIBRATION_MARGIN_M = 0.075
DEFAULT_MIN_SCORE = 0.10
DEFAULT_MIN_VALID_DEPTH_FRACTION = 0.80
DEFAULT_MIN_VALID_DEPTH_PIXELS = 20
DEFAULT_MAX_DEPTH_SPREAD_MM = 75.0
DEFAULT_MAX_XYZ_DISAGREEMENT_MM = 60.0
DEFAULT_MAX_FRAME_AGE_S = 30.0
DEFAULT_MIN_OBJECT_EXTENT_MM = 5.0
DEFAULT_MAX_OBJECT_EXTENT_MM = 600.0
DEFAULT_SURFACE_Z_MARGIN_MM = 75.0


class IntegrationError(RuntimeError):
    """Refusal raised when perception cannot produce a safe target."""

    pass


@dataclass(frozen=True)
class PickIntent:
    """Constrained language intent allowed to reach perception."""

    transcript: str
    transcript_source: str
    object_name: str
    qualifier: str | None
    destination: str | None
    source_phrase: str
    grounding_intent: dict[str, Any]
    intent_hash: str

    @property
    def segmentation_request(self) -> str:
        """Return the deterministic category/qualifier SAM request."""

        return self.source_phrase


@dataclass(frozen=True)
class Calibration:
    """Validated camera-to-base calibration and its source evidence."""

    path: Path
    raw: dict[str, Any]
    rotation: np.ndarray
    translation: np.ndarray
    intrinsics: dict[str, float]
    camera_points: np.ndarray
    base_points: np.ndarray
    resolution: str


QUALIFIER_PATTERNS = (
    ('rightmost', r'\bright[ -]?most\b|\bon the (?:far )?right\b'),
    ('leftmost', r'\bleft[ -]?most\b|\bon the (?:far )?left\b'),
    ('topmost', r'\btop[ -]?most\b|\bat the top\b'),
    ('bottommost', r'\bbottom[ -]?most\b|\bat the bottom\b'),
    ('nearest', r'\bnearest\b|\bclosest\b'),
    ('farthest', r'\bfarthest\b|\bfurthest\b'),
    ('largest', r'\blargest\b|\bbiggest\b'),
    ('smallest', r'\bsmallest\b'),
)


def split_spatial_qualifier(object_phrase: str) -> tuple[str, str | None]:
    """Remove one supported qualifier from the parsed object phrase."""
    normalized = re.sub(r'\s+', ' ', object_phrase.strip().lower())
    found = [
        (name, pattern) for name, pattern in QUALIFIER_PATTERNS
        if re.search(pattern, normalized)
    ]
    canonical = sorted({name for name, _ in found})
    if len(canonical) > 1:
        raise IntegrationError(
            'request contains multiple spatial qualifiers: ' +
            ', '.join(canonical))
    if not found:
        if not normalized:
            raise IntegrationError('parsed request has no target object')
        return normalized, None

    qualifier, pattern = found[0]
    object_name = re.sub(pattern, ' ', normalized)
    object_name = re.sub(r'\s+', ' ', object_name).strip(' ,.-')
    if not object_name:
        raise IntegrationError(
            f'qualifier {qualifier!r} was provided without an object')
    return object_name, qualifier


def load_grounding_contract(sam_project: Path):
    """Import the one shared structured-intent implementation."""

    scripts_dir = sam_project.expanduser().resolve() / 'scripts'
    if not (scripts_dir / 'grounding_intent.py').is_file():
        raise IntegrationError(
            f'grounding intent contract not found under {scripts_dir}')
    scripts_text = str(scripts_dir)
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)
    try:
        import grounding_intent
    except ImportError as exc:
        raise IntegrationError(
            f'cannot import structured grounding contract: {exc}') from exc
    return grounding_intent


def load_pick_intent(
    *,
    text: str | None,
    voice: bool,
    voice_duration: float,
    whisper_model: str | None,
    vla_project: Path,
    sam_project: Path,
) -> PickIntent:
    """Get text or speech from VLA_project and constrain it to one pick."""

    source_dir = vla_project.expanduser().resolve() / 'src'
    if not (source_dir / 'co_bot_vlm').is_dir():
        raise IntegrationError(
            f'VLA_project package not found under {source_dir}')
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

    try:
        from co_bot_vlm.command import parse_transcript_command
        from co_bot_vlm.transcript import get_transcript
    except ImportError as exc:
        raise IntegrationError(f'cannot import VLA_project: {exc}') from exc

    try:
        transcript = get_transcript(
            text=text,
            voice=voice,
            voice_duration_seconds=voice_duration,
            whisper_model=whisper_model,
        )
        command = parse_transcript_command(transcript.text)
    except Exception as exc:
        code = getattr(exc, 'code', type(exc).__name__)
        message = getattr(exc, 'message', str(exc))
        raise IntegrationError(f'VLA intent refused ({code}): {message}') from exc

    if command.action != 'pick_and_place' or not command.object:
        raise IntegrationError(
            f'only an object pick request is supported, not {command.action!r}')
    object_name, qualifier = split_spatial_qualifier(command.object)
    source_phrase = (
        f'{qualifier} {object_name}' if qualifier else object_name
    )
    if command.source:
        source_phrase = f'{source_phrase} from the {command.source}'
    grounding = load_grounding_contract(sam_project)
    try:
        structured = grounding.parse_grounding_intent(source_phrase)
        structured_hash = grounding.intent_hash(structured)
    except Exception as exc:
        code = getattr(exc, 'code', type(exc).__name__)
        raise IntegrationError(
            f'visual grounding intent refused ({code}): {exc}') from exc
    return PickIntent(
        transcript=transcript.text,
        transcript_source=transcript.source,
        object_name=structured['category'],
        qualifier=structured['selector'],
        destination=command.destination,
        source_phrase=structured['source_phrase'],
        grounding_intent=structured,
        intent_hash=structured_hash,
    )


def load_calibration(path: Path) -> Calibration:
    """Load the accepted B2 transform, intrinsics, and source envelopes."""

    path = path.expanduser().resolve()
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationError(f'cannot load calibration {path}: {exc}') from exc
    if not isinstance(raw, dict):
        raise IntegrationError(f'calibration {path} must contain one object')
    if raw.get('schema_version') != 1:
        raise IntegrationError(f'unsupported calibration schema in {path}')
    if not raw.get('quality', {}).get('passed', False):
        raise IntegrationError(f'calibration did not pass quality gates: {path}')
    if raw.get('parent_frame') != BASE_FRAME:
        raise IntegrationError(f'calibration parent must be {BASE_FRAME}')
    if raw.get('child_frame') != CAMERA_FRAME:
        raise IntegrationError(f'calibration child must be {CAMERA_FRAME}')

    try:
        rotation = np.asarray(raw.get('R'), dtype=float)
        translation = np.asarray(raw.get('t_m'), dtype=float)
    except (TypeError, ValueError) as exc:
        raise IntegrationError('calibration R/t must be numeric') from exc
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise IntegrationError('calibration R/t have invalid dimensions')
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise IntegrationError('calibration R/t contain non-finite values')

    source_capture = raw.get('source_capture')
    capture_path = (Path(source_capture).expanduser()
                    if isinstance(source_capture, str) else Path())
    if not capture_path.is_file():
        sibling = path.with_name('calib_points.json')
        capture_path = sibling if sibling.is_file() else capture_path
    try:
        capture_bytes = capture_path.read_bytes()
        capture = json.loads(capture_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationError(
            f'cannot load calibration source {capture_path}: {exc}') from exc
    actual_hash = hashlib.sha256(capture_bytes).hexdigest()
    if not raw.get('source_sha256') or actual_hash != raw['source_sha256']:
        raise IntegrationError(
            'calibration source points changed; rerun B2 before targeting')

    try:
        pairs = capture['pairs']
        camera_points = np.asarray(
            [pair['cam_xyz'] for pair in pairs], dtype=float)
        base_points = np.asarray(
            [pair['base_xyz'] for pair in pairs], dtype=float)
        intrinsics = {
            key: float(capture['intrinsics'][key])
            for key in ('fx', 'fy', 'cx', 'cy')
        }
        resolution = str(capture['resolution'])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrationError('invalid calibration source data') from exc
    if camera_points.shape[0] < 8 or camera_points.shape[1:] != (3,):
        raise IntegrationError('calibration source needs at least eight camera points')
    if base_points.shape != camera_points.shape:
        raise IntegrationError('camera/base calibration point counts differ')
    if not np.isfinite(camera_points).all() or not np.isfinite(base_points).all():
        raise IntegrationError('calibration source contains non-finite points')
    if any(not math.isfinite(value) or value <= 0.0
           for key, value in intrinsics.items() if key in ('fx', 'fy')):
        raise IntegrationError('calibration focal lengths are invalid')
    return Calibration(
        path=path,
        raw=raw,
        rotation=rotation,
        translation=translation,
        intrinsics=intrinsics,
        camera_points=camera_points,
        base_points=base_points,
        resolution=resolution,
    )


def request_segmentation(
    intent: PickIntent,
    *,
    host: str,
    use_agent_fallback: bool,
    timeout: float,
) -> dict[str, Any]:
    """Request one fresh segmentation from the localhost SAM service."""

    base_url = host.rstrip('/')
    if not re.match(r'^https?://', base_url):
        base_url = 'http://' + base_url
    request_payload = {
        'schema_version': 1,
        'source_phrase': intent.source_phrase,
        'grounding_intent': intent.grounding_intent,
        'intent_hash': intent.intent_hash,
        'use_agent_fallback': bool(use_agent_fallback),
    }
    http_request = urllib.request.Request(
        f'{base_url}/v1/segment',
        data=json.dumps(request_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST')
    try:
        with urllib.request.urlopen(http_request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode('utf-8', errors='replace')
        except Exception:
            detail = str(exc)
        raise IntegrationError(
            f'SAM service returned HTTP {exc.code}: {detail}') from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise IntegrationError(
            f'cannot reach SAM service at {base_url}: {exc}') from exc
    except json.JSONDecodeError as exc:
        raise IntegrationError('SAM service returned invalid JSON') from exc
    if not isinstance(payload, dict):
        raise IntegrationError('SAM service response must be a JSON object')
    return payload


def read_response_file(path: Path) -> dict[str, Any]:
    """Load a recorded service response for offline testing."""

    try:
        value = json.loads(path.expanduser().read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationError(f'cannot read service response {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise IntegrationError('saved service response must be a JSON object')
    return value


def validate_response_identity(response: dict[str, Any], intent: PickIntent) -> None:
    """Ensure the exact versioned intent survived the service round trip."""

    if response.get('schema_version') != 1:
        raise IntegrationError('SAM response has no supported schema version')
    returned_intent = response.get('grounding_intent')
    if not isinstance(returned_intent, dict):
        raise IntegrationError('SAM response has no structured grounding intent')
    if returned_intent != intent.grounding_intent:
        raise IntegrationError(
            'SAM returned a changed structured grounding intent')
    source_phrase = returned_intent.get('source_phrase')
    if source_phrase != intent.source_phrase:
        raise IntegrationError(
            f'SAM source phrase changed: {source_phrase!r} != '
            f'{intent.source_phrase!r}')
    canonical_hash = hashlib.sha256(json.dumps(
        returned_intent,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    ).encode('utf-8')).hexdigest()
    returned_hash = response.get('intent_hash')
    if returned_hash != intent.intent_hash or returned_hash != canonical_hash:
        raise IntegrationError(
            'SAM structured intent hash does not match the request')
    selection = response.get('selection')
    if selection is not None:
        if not isinstance(selection, dict):
            raise IntegrationError('SAM spatial selection metadata is invalid')
        if selection.get('selector') != intent.qualifier:
            raise IntegrationError(
                'SAM spatial selection metadata is inconsistent')


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise IntegrationError(f'{label} must be a finite number')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise IntegrationError(f'{label} must be a finite number') from exc
    if not math.isfinite(number):
        raise IntegrationError(f'{label} must be a finite number')
    return number


def frame_capture_time(
    response: dict[str, Any],
    *,
    max_age_s: float,
    allow_stale: bool,
) -> tuple[Path, datetime, float]:
    """Resolve the captured frame time and refuse stale perception."""

    frame_value = response.get('frame')
    if not isinstance(frame_value, str) or not frame_value:
        raise IntegrationError('SAM response has no frame path')
    frame_path = Path(frame_value).expanduser().resolve()
    if not frame_path.is_file():
        raise IntegrationError(f'SAM frame does not exist: {frame_path}')
    captured = datetime.fromtimestamp(frame_path.stat().st_mtime, timezone.utc)
    age_s = (datetime.now(timezone.utc) - captured).total_seconds()
    if age_s < -5.0:
        raise IntegrationError(
            f'SAM frame timestamp is {-age_s:.1f} seconds in the future')
    if age_s > max_age_s and not allow_stale:
        raise IntegrationError(
            f'SAM frame is {age_s:.1f} seconds old (limit {max_age_s:.1f}); '
            'request a fresh segmentation')
    return frame_path, captured, max(0.0, age_s)


def selected_box(
    response: dict[str, Any],
) -> tuple[list[int], list[int], list[int], int, int]:
    """Return local bbox, local center, full-frame center, image width/height."""
    if response.get('num_kept') != 1:
        raise IntegrationError(
            f'exactly one mask is required; SAM kept {response.get("num_kept")!r}')
    selected_mask = response.get('selected_mask')
    if isinstance(selected_mask, dict):
        zed_frame = response.get('zed_frame')
        crop = zed_frame.get('crop') if isinstance(zed_frame, dict) else None
        if not isinstance(crop, dict):
            raise IntegrationError('SAM selected mask lacks crop metadata')
        try:
            image_width = int(crop['output_width'])
            image_height = int(crop['output_height'])
            x, y, width, height = [
                int(value) for value in selected_mask['bbox_xywh_crop_pixels']
            ]
            local_u, local_v = [
                int(round(float(value)))
                for value in selected_mask['center_xy_crop_pixels']
            ]
            full_u, full_v = [
                int(round(float(value)))
                for value in selected_mask['center_xy_full_pixels']
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrationError('SAM selected mask coordinates are invalid') from exc
        if (
            image_width <= 0 or image_height <= 0 or x < 0 or y < 0
            or width <= 0 or height <= 0 or x + width > image_width
            or y + height > image_height
        ):
            raise IntegrationError('SAM selected mask box exceeds the crop')
        return (
            [x, y, x + width, y + height],
            [local_u, local_v],
            [full_u, full_v],
            image_width,
            image_height,
        )

    gate = response.get('presence_gate')
    if not isinstance(gate, dict):
        raise IntegrationError('SAM response has no presence gate')
    kept_indices = gate.get('kept_indices')
    if not isinstance(kept_indices, list) or len(kept_indices) != 1:
        raise IntegrationError('SAM response does not identify one kept candidate')
    candidate_index = kept_indices[0]
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int):
        raise IntegrationError('SAM kept candidate index is invalid')

    sam_json_value = response.get('sam_json')
    if not isinstance(sam_json_value, str) or not sam_json_value:
        raise IntegrationError('SAM response has no detection JSON path')
    sam_json_path = Path(sam_json_value).expanduser().resolve()
    try:
        sam_output = json.loads(sam_json_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrationError(
            f'cannot read SAM detections {sam_json_path}: {exc}') from exc
    try:
        image_width = int(sam_output['orig_img_w'])
        image_height = int(sam_output['orig_img_h'])
        raw_box = sam_output['pred_boxes'][candidate_index]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise IntegrationError('selected SAM bounding box is missing') from exc
    if image_width <= 0 or image_height <= 0:
        raise IntegrationError('SAM detection image dimensions are invalid')
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        raise IntegrationError('SAM pred_boxes entry must be normalized xywh')
    x, y, width, height = (
        _finite_float(value, f'pred_boxes[{candidate_index}]')
        for value in raw_box)
    if (x < 0.0 or y < 0.0 or width <= 0.0 or height <= 0.0 or
            x + width > 1.001 or y + height > 1.001):
        raise IntegrationError(
            'selected SAM box must be normalized [x, y, width, height]')

    x1 = max(0, int(math.floor(x * image_width)))
    y1 = max(0, int(math.floor(y * image_height)))
    x2 = min(image_width, int(math.ceil((x + width) * image_width)))
    y2 = min(image_height, int(math.ceil((y + height) * image_height)))
    local_u = int(round((x + width / 2.0) * image_width))
    local_v = int(round((y + height / 2.0) * image_height))
    local_u = min(max(local_u, 0), image_width - 1)
    local_v = min(max(local_v, 0), image_height - 1)

    zed_frame = response.get('zed_frame')
    if not isinstance(zed_frame, dict):
        raise IntegrationError('SAM response has no ZED frame metadata')
    crop = zed_frame.get('crop', {})
    if not isinstance(crop, dict):
        raise IntegrationError('SAM crop metadata is invalid')
    if crop.get('enabled', False):
        applied = crop.get('applied_xyxy')
        if (not isinstance(applied, list) or len(applied) != 4 or
                any(isinstance(value, bool) or not isinstance(value, int)
                    for value in applied)):
            raise IntegrationError('SAM crop lacks applied_xyxy metadata')
        offset_x, offset_y = applied[:2]
    else:
        offset_x = offset_y = 0
    full_center = [local_u + offset_x, local_v + offset_y]
    return [x1, y1, x2, y2], [local_u, local_v], full_center, image_width, image_height


def depth_evidence(
    response: dict[str, Any],
    *,
    min_score: float,
    min_valid_fraction: float,
    min_valid_pixels: int,
    max_spread_m: float,
) -> tuple[float, float, dict[str, Any], np.ndarray]:
    """Validate the selected mask's score, depth coverage, and spread."""

    report = response.get('object_depth')
    objects = report.get('objects') if isinstance(report, dict) else None
    if not isinstance(objects, list) or len(objects) != 1:
        raise IntegrationError('SAM response needs depth for exactly one object')
    obj = objects[0]
    if not isinstance(obj, dict):
        raise IntegrationError('SAM object depth entry is invalid')
    score = _finite_float(obj.get('score'), 'segmentation score')
    if score < min_score:
        raise IntegrationError(
            f'segmentation score {score:.3f} is below {min_score:.3f}')
    stats = obj.get('depth_stats_m')
    if not isinstance(stats, dict):
        raise IntegrationError('SAM object has no depth statistics')
    valid_fraction = _finite_float(
        stats.get('valid_fraction'), 'valid depth fraction')
    if valid_fraction < min_valid_fraction:
        raise IntegrationError(
            f'valid depth fraction {valid_fraction:.3f} is below '
            f'{min_valid_fraction:.3f}')
    valid_pixels = stats.get('valid_depth_pixels')
    if (isinstance(valid_pixels, bool) or not isinstance(valid_pixels, int) or
            valid_pixels < min_valid_pixels):
        raise IntegrationError(
            f'only {valid_pixels!r} valid masked depth pixels; '
            f'need at least {min_valid_pixels}')
    median_depth = _finite_float(stats.get('median'), 'median depth')
    if median_depth <= 0.0:
        raise IntegrationError('median depth must be positive')
    p10 = _finite_float(stats.get('p10'), 'depth p10')
    p90 = _finite_float(stats.get('p90'), 'depth p90')
    spread = p90 - p10
    if spread < 0.0 or spread > max_spread_m:
        raise IntegrationError(
            f'masked depth p90-p10 spread is {spread * 1000:.1f} mm '
            f'(limit {max_spread_m * 1000:.1f} mm)')
    try:
        service_xyz = np.asarray(obj.get('xyz_centroid_m'), dtype=float)
    except (TypeError, ValueError) as exc:
        raise IntegrationError(
            'SAM object has no valid camera XYZ reference') from exc
    if service_xyz.shape != (3,) or not np.isfinite(service_xyz).all():
        raise IntegrationError('SAM object has no valid camera XYZ reference')
    return score, median_depth, stats, service_xyz


def check_service_frame(response: dict[str, Any], calibration: Calibration) -> None:
    """Require the calibrated ZED view, resolution, and XYZ measure."""

    zed = response.get('zed_frame')
    if not isinstance(zed, dict):
        raise IntegrationError('SAM response has no ZED frame metadata')
    if zed.get('view') != 'LEFT':
        raise IntegrationError('targeting requires the calibrated LEFT ZED view')
    resolution = str(zed.get('resolution', ''))
    if resolution.lower() != calibration.resolution.lower():
        raise IntegrationError(
            f'ZED resolution {resolution!r} does not match calibration '
            f'{calibration.resolution!r}')
    if zed.get('xyz_measure') != 'XYZ':
        raise IntegrationError('targeting requires ZED MEASURE.XYZ')


def backproject_pixel(
    pixel: list[int], depth_m: float, intrinsics: dict[str, float]
) -> np.ndarray:
    """Back-project a full-frame image pixel into ZED image coordinates."""

    u, v = pixel
    return np.asarray([
        (float(u) - intrinsics['cx']) * depth_m / intrinsics['fx'],
        (float(v) - intrinsics['cy']) * depth_m / intrinsics['fy'],
        depth_m,
    ], dtype=float)


def envelope_error(point: np.ndarray, references: np.ndarray, label: str) -> str | None:
    """Describe a point outside the B1 calibrated workspace, if any."""

    lower = references.min(axis=0) - CALIBRATION_MARGIN_M
    upper = references.max(axis=0) + CALIBRATION_MARGIN_M
    if np.any(point < lower) or np.any(point > upper):
        return (
            f'{label} {np.round(point, 4).tolist()} is outside calibrated '
            f'envelope {np.round(lower, 4).tolist()} .. '
            f'{np.round(upper, 4).tolist()}')
    return None


def build_target_record(
    *,
    intent: PickIntent,
    response: dict[str, Any],
    calibration: Calibration,
    captured: datetime,
    frame_age_s: float,
    frame_path: Path,
    bbox_local: list[int],
    center_local: list[int],
    center_full: list[int],
    score: float,
    depth_m: float,
    stats: dict[str, Any],
    service_xyz: np.ndarray,
    max_xyz_disagreement_m: float,
    min_object_extent_m: float,
    max_object_extent_m: float,
    surface_z_margin_m: float,
) -> dict[str, Any]:
    """Transform checked camera evidence into the existing B3 contract."""

    bbox_width_pixels = bbox_local[2] - bbox_local[0]
    bbox_height_pixels = bbox_local[3] - bbox_local[1]
    approximate_width_m = bbox_width_pixels * depth_m / calibration.intrinsics['fx']
    approximate_height_m = bbox_height_pixels * depth_m / calibration.intrinsics['fy']
    largest_extent_m = max(approximate_width_m, approximate_height_m)
    if largest_extent_m < min_object_extent_m:
        raise IntegrationError(
            f'projected object extent is {largest_extent_m * 1000:.1f} mm '
            f'(minimum {min_object_extent_m * 1000:.1f} mm)')
    if largest_extent_m > max_object_extent_m:
        raise IntegrationError(
            f'projected object extent is {largest_extent_m * 1000:.1f} mm '
            f'(maximum {max_object_extent_m * 1000:.1f} mm)')

    camera_xyz = backproject_pixel(
        center_full, depth_m, calibration.intrinsics)
    disagreement = float(np.linalg.norm(camera_xyz - service_xyz))
    if disagreement > max_xyz_disagreement_m:
        raise IntegrationError(
            'box-center back-projection disagrees with SAM mask XYZ by '
            f'{disagreement * 1000:.1f} mm '
            f'(limit {max_xyz_disagreement_m * 1000:.1f} mm)')
    error = envelope_error(
        camera_xyz, calibration.camera_points, 'camera target')
    if error:
        raise IntegrationError(error)
    base_xyz = calibration.rotation @ camera_xyz + calibration.translation
    error = envelope_error(base_xyz, calibration.base_points, 'base target')
    if error:
        raise IntegrationError(error)
    surface_z_min = float(calibration.base_points[:, 2].min() - surface_z_margin_m)
    surface_z_max = float(calibration.base_points[:, 2].max() + surface_z_margin_m)
    if not surface_z_min <= float(base_xyz[2]) <= surface_z_max:
        raise IntegrationError(
            f'target surface height {base_xyz[2]:.4f} m is outside calibrated '
            f'range {surface_z_min:.4f}..{surface_z_max:.4f} m')
    hover = base_xyz + np.asarray([0.0, 0.0, HOVER_M])

    return {
        'schema_version': 1,
        'created': captured.isoformat(timespec='seconds'),
        'purpose': 'VLA-selected SAM bounding-box center surface target',
        'producer': 'fr5_bringup/vla_pick_target.py',
        'calibration_file': str(calibration.path),
        'calibration_created': calibration.raw.get('created'),
        'calibration_source_sha256': calibration.raw.get('source_sha256'),
        'camera_frame': CAMERA_FRAME,
        'base_frame': BASE_FRAME,
        'transcript': {
            'text': intent.transcript,
            'source': intent.transcript_source,
        },
        'intent': {
            'action': 'pick_and_place',
            'object': intent.object_name,
            'qualifier': intent.qualifier,
            'destination': intent.destination,
            'grounding': intent.grounding_intent,
            'grounding_hash': intent.intent_hash,
        },
        'segmentation': {
            'request': intent.segmentation_request,
            'path': response.get('path'),
            'sam_prompt': response.get('sam_prompt'),
            'sam_prompts': response.get('sam_prompts'),
            'score': score,
            'bbox_xyxy_in_segmentation_image': bbox_local,
            'bbox_center_pixel_in_segmentation_image': center_local,
            'selection': response.get('selection'),
            'result_json': response.get('result_json'),
            'sam_json': response.get('sam_json'),
            'output_dir': response.get('output_dir'),
            'selected_mask': response.get('selected_mask'),
            'candidate_generation': response.get('candidate_generation'),
            'verification': response.get('verification'),
            'geometry_gate': response.get('geometry_gate'),
        },
        'frame': {
            'path': str(frame_path),
            'captured': captured.isoformat(timespec='seconds'),
            'age_at_target_creation_s': round(frame_age_s, 3),
            'zed': response.get('zed_frame'),
        },
        'pixel': center_full,
        'camera_xyz_method': (
            'SAM bbox center back-projected with masked median ZED depth'),
        'camera_xyz_m': camera_xyz.tolist(),
        'camera_xyz_reference_m': service_xyz.tolist(),
        'camera_xyz_reference_method': 'per-axis median XYZ under selected mask',
        'camera_xyz_disagreement_m': disagreement,
        'depth_evidence': {
            **stats,
            'selected_depth_m': depth_m,
        },
        'physical_size_evidence': {
            'method': 'pinhole projection of selected mask bounding box',
            'bbox_width_pixels': bbox_width_pixels,
            'bbox_height_pixels': bbox_height_pixels,
            'approximate_width_m': approximate_width_m,
            'approximate_height_m': approximate_height_m,
            'largest_extent_m': largest_extent_m,
            'allowed_extent_m': [min_object_extent_m, max_object_extent_m],
        },
        'surface_height_evidence': {
            'base_surface_z_m': float(base_xyz[2]),
            'calibrated_allowed_z_m': [surface_z_min, surface_z_max],
            'margin_m': surface_z_margin_m,
        },
        'base_surface_xyz_m': base_xyz.tolist(),
        'hover_offset_m': HOVER_M,
        'base_hover_xyz_m': hover.tolist(),
    }


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace a JSON target only after its full content is written."""

    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def render_audit(
    *,
    frame_path: Path,
    output_path: Path,
    bbox: list[int],
    center: list[int],
    intent: PickIntent,
    record: dict[str, Any],
) -> Path:
    """Render the selected box, center, depth, and robot-space target."""

    try:
        import cv2
    except ImportError as exc:
        raise IntegrationError(
            'OpenCV is required to render the targeting audit overlay') from exc
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise IntegrationError(f'cannot read SAM frame for audit: {frame_path}')
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    if x2 > width or y2 > height:
        raise IntegrationError('SAM bounding box exceeds the saved frame')
    cv2.rectangle(image, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)),
                  (0, 255, 0), 2)
    cv2.drawMarker(image, tuple(center), (0, 0, 255),
                   cv2.MARKER_CROSS, 30, 2)
    base = record['base_surface_xyz_m']
    lines = [
        f'NO ROBOT MOTION | {intent.transcript[:100]}',
        f'target: {intent.segmentation_request} | score '
        f'{record["segmentation"]["score"]:.3f}',
        f'box center: full pixel {record["pixel"]} | depth '
        f'{record["depth_evidence"]["selected_depth_m"]:.3f} m',
        'base XYZ m: ' + ' '.join(f'{value:+.4f}' for value in base),
    ]
    for index, line in enumerate(lines):
        y = 28 + 27 * index
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 1, cv2.LINE_AA)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        output_path.stem + '.tmp' + output_path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise IntegrationError(f'cannot write audit overlay {temporary}')
    temporary.replace(output_path)
    return output_path


def print_handoff(record: dict[str, Any], target_path: Path, audit_path: Path) -> None:
    """Print auditable plan-only commands for the existing robot tools."""

    target_path = target_path.expanduser().resolve()
    target_arg = shlex.quote(str(target_path))
    print('\n=== VLA TARGET ACCEPTED (NO ROBOT MOTION OCCURRED) ===')
    print(f'transcript:  {record["transcript"]["text"]}')
    print(f'intent:      {record["intent"]}')
    print(f'box center:  {record["pixel"]}')
    print('camera XYZ:  ' + ' '.join(
        f'{value:+.6f}' for value in record['camera_xyz_m']) + ' m')
    print('base surface:' + ' '.join(
        f' {value:+.6f}' for value in record['base_surface_xyz_m']) + ' m')
    print(f'audit:       {audit_path}')
    print(f'target:      {target_path}')
    print('\nReview the audit image first. Then run a PLAN-ONLY hover:')
    print(f'ros2 run fr5_bringup b3_hover.py --target-file={target_arg}')
    print('\nAfter hover validation, plan the existing D0 sequence (still no motion):')
    print(f'ros2 run fr5_bringup d0_point_grab.py --target-file={target_arg}')
    print('\nThis integration intentionally provides no live-execution flag.')


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and range-check the no-motion integration CLI."""

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--text', help='typed pick command')
    source.add_argument('--voice', action='store_true',
                        help='record and transcribe one push-to-talk command')
    parser.add_argument('--voice-duration', type=float, default=5.0)
    parser.add_argument('--whisper-model')
    parser.add_argument('--vla-project', type=Path, default=DEFAULT_VLA_PROJECT)
    parser.add_argument('--sam-project', type=Path, default=DEFAULT_SAM_PROJECT)
    parser.add_argument('--service-host', default=DEFAULT_SERVICE_HOST)
    parser.add_argument('--service-timeout-sec', type=float, default=300.0)
    fallback = parser.add_mutually_exclusive_group()
    fallback.add_argument(
        '--agent-fallback', action='store_true',
        help=('allow the unbounded SAM/Qwen agent only after the bounded DINO '
              'pipeline produces no candidate; disabled by default'))
    fallback.add_argument(
        '--no-agent-fallback', action='store_true',
        help=('deprecated compatibility flag; bounded DINO mode already disables '
              'the agent fallback by default'))
    parser.add_argument(
        '--service-response', type=Path,
        help='use a saved /v1/segment response for an offline/stale-data test')
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--out', type=Path, default=DEFAULT_TARGET)
    parser.add_argument('--audit-output', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--min-score', type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument('--min-valid-depth-fraction', type=float,
                        default=DEFAULT_MIN_VALID_DEPTH_FRACTION)
    parser.add_argument('--min-valid-depth-pixels', type=int,
                        default=DEFAULT_MIN_VALID_DEPTH_PIXELS)
    parser.add_argument('--max-depth-spread-mm', type=float,
                        default=DEFAULT_MAX_DEPTH_SPREAD_MM)
    parser.add_argument('--max-xyz-disagreement-mm', type=float,
                        default=DEFAULT_MAX_XYZ_DISAGREEMENT_MM)
    parser.add_argument('--max-frame-age-sec', type=float,
                        default=DEFAULT_MAX_FRAME_AGE_S)
    parser.add_argument('--min-object-extent-mm', type=float,
                        default=DEFAULT_MIN_OBJECT_EXTENT_MM)
    parser.add_argument('--max-object-extent-mm', type=float,
                        default=DEFAULT_MAX_OBJECT_EXTENT_MM)
    parser.add_argument('--surface-z-margin-mm', type=float,
                        default=DEFAULT_SURFACE_Z_MARGIN_MM)
    parser.add_argument(
        '--allow-stale-frame', action='store_true',
        help='allow offline perception inspection; D0 still sees the old timestamp')
    args = parser.parse_args(argv)
    if args.voice_duration <= 0.0:
        parser.error('--voice-duration must be positive')
    if args.service_timeout_sec <= 0.0:
        parser.error('--service-timeout-sec must be positive')
    if not 0.0 <= args.min_score <= 1.0:
        parser.error('--min-score must be in [0, 1]')
    if not 0.0 <= args.min_valid_depth_fraction <= 1.0:
        parser.error('--min-valid-depth-fraction must be in [0, 1]')
    if args.min_valid_depth_pixels < 1:
        parser.error('--min-valid-depth-pixels must be positive')
    if args.max_depth_spread_mm <= 0.0:
        parser.error('--max-depth-spread-mm must be positive')
    if args.max_xyz_disagreement_mm <= 0.0:
        parser.error('--max-xyz-disagreement-mm must be positive')
    if args.max_frame_age_sec <= 0.0:
        parser.error('--max-frame-age-sec must be positive')
    if args.min_object_extent_mm <= 0.0:
        parser.error('--min-object-extent-mm must be positive')
    if args.max_object_extent_mm <= args.min_object_extent_mm:
        parser.error('--max-object-extent-mm must exceed --min-object-extent-mm')
    if args.surface_z_margin_mm < 0.0:
        parser.error('--surface-z-margin-mm must not be negative')
    return args


def main(argv: list[str] | None = None) -> int:
    """Run language-to-target integration without commanding robot motion."""

    args = parse_args(argv)
    try:
        intent = load_pick_intent(
            text=args.text,
            voice=args.voice,
            voice_duration=args.voice_duration,
            whisper_model=args.whisper_model,
            vla_project=args.vla_project,
            sam_project=args.sam_project,
        )
        print('=== INTERPRETED REQUEST (NO ROBOT MOTION) ===', flush=True)
        print(f'transcript: {intent.transcript}')
        print(f'object: {intent.object_name}')
        print(f'qualifier: {intent.qualifier or "none"}')
        print(f'visual intent: {intent.grounding_intent}')
        print(f'intent hash: {intent.intent_hash}')
        print(f'destination: {intent.destination or "none"}', flush=True)

        calibration = load_calibration(args.calibration)
        if args.service_response:
            response = read_response_file(args.service_response)
        else:
            response = request_segmentation(
                intent,
                host=args.service_host,
                use_agent_fallback=args.agent_fallback,
                timeout=args.service_timeout_sec,
            )
        validate_response_identity(response, intent)
        check_service_frame(response, calibration)
        frame_path, captured, frame_age_s = frame_capture_time(
            response,
            max_age_s=args.max_frame_age_sec,
            allow_stale=args.allow_stale_frame,
        )
        bbox, center_local, center_full, _width, _height = selected_box(response)
        score, depth_m, stats, service_xyz = depth_evidence(
            response,
            min_score=args.min_score,
            min_valid_fraction=args.min_valid_depth_fraction,
            min_valid_pixels=args.min_valid_depth_pixels,
            max_spread_m=args.max_depth_spread_mm / 1000.0,
        )
        record = build_target_record(
            intent=intent,
            response=response,
            calibration=calibration,
            captured=captured,
            frame_age_s=frame_age_s,
            frame_path=frame_path,
            bbox_local=bbox,
            center_local=center_local,
            center_full=center_full,
            score=score,
            depth_m=depth_m,
            stats=stats,
            service_xyz=service_xyz,
            max_xyz_disagreement_m=args.max_xyz_disagreement_mm / 1000.0,
            min_object_extent_m=args.min_object_extent_mm / 1000.0,
            max_object_extent_m=args.max_object_extent_mm / 1000.0,
            surface_z_margin_m=args.surface_z_margin_mm / 1000.0,
        )
        audit_path = render_audit(
            frame_path=frame_path,
            output_path=args.audit_output,
            bbox=bbox,
            center=center_local,
            intent=intent,
            record=record,
        )
        record['audit_overlay'] = str(audit_path)
        atomic_write_json(args.out, record)
        print_handoff(record, args.out, audit_path)
        return 0
    except IntegrationError as exc:
        print(f'VLA TARGET REFUSED: {exc}', file=sys.stderr)
        print('No new target was written and no robot motion was requested.',
              file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('VLA targeting interrupted; no robot motion was requested.',
              file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f'VLA TARGET REFUSED: unexpected {type(exc).__name__}: {exc}',
            file=sys.stderr,
        )
        print('No new target was written and no robot motion was requested.',
              file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
