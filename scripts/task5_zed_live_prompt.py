#!/usr/bin/env python3
"""Task 5: grab a live ZED RGB frame and run the SAM 3.1 overlay pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import mask_depth
import stereo_mask_warp as stereo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/sam3.1/sam3.1_multiplex.pt"
DEFAULT_FRAME_OUTPUT = PROJECT_ROOT / "outputs/task5_live_frame.png"
DEFAULT_STEREO_FRAME_OUTPUT = PROJECT_ROOT / "outputs/task5_live_frame_stereo.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/task5_zed_live_prompt.json"
DEFAULT_OVERLAY_OUTPUT = PROJECT_ROOT / "outputs/result_live.png"
DEFAULT_STEREO_OVERLAY_OUTPUT = PROJECT_ROOT / "outputs/result_live_stereo.png"
DEFAULT_CROP = (448, 360, 384, 360)


RESOLUTION_NAMES = ("HD2K", "HD1200", "HD1080", "HD720", "SVGA", "VGA", "AUTO")
VIEW_NAMES = ("LEFT", "RIGHT")


def parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = value.replace(",", " ").split()
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,w,h")
    try:
        x, y, width, height = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop values must be integers") from exc
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("crop must use x>=0, y>=0, w>0, h>0")
    return x, y, width, height


def add_crop_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared ZED crop controls to a command-line parser."""

    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop",
        type=parse_crop,
        metavar="X,Y,W,H",
        help=(
            "Override the standard HD720 pixel crop as x,y,w,h. "
            f"Default: {','.join(str(value) for value in DEFAULT_CROP)}."
        ),
    )
    crop_group.add_argument(
        "--no-crop",
        dest="crop",
        action="store_const",
        const=None,
        help="Disable the standard crop and process the complete camera frame.",
    )
    parser.set_defaults(crop=DEFAULT_CROP)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grab a live ZED RGB frame, optionally crop it, then run the "
            "Task 4 SAM overlay pipeline."
        )
    )
    parser.add_argument(
        "--prompt",
        default="green object",
        help="Text prompt to send to SAM 3.1 after grabbing the live frame.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Only save live RGB frames for camera/crop tuning. Do not run SAM.",
    )
    parser.add_argument(
        "--preview-count",
        default=1,
        type=int,
        help="Preview frames to save. Use 0 to refresh until Ctrl-C.",
    )
    parser.add_argument(
        "--interval",
        default=0.5,
        type=float,
        help="Seconds between preview frame refreshes.",
    )
    add_crop_arguments(parser)
    parser.add_argument(
        "--frame-output",
        default=DEFAULT_FRAME_OUTPUT,
        type=Path,
        help="Where to save the latest live RGB frame before SAM.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT,
        type=Path,
        help="Where to write the live-frame SAM result summary.",
    )
    parser.add_argument(
        "--overlay-output",
        default=DEFAULT_OVERLAY_OUTPUT,
        type=Path,
        help="Where to write the Task 4 overlay for the live frame.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        type=Path,
        help="Path to sam3.1_multiplex.pt.",
    )
    parser.add_argument(
        "--threshold",
        default=0.05,
        type=float,
        help="SAM output probability threshold.",
    )
    parser.add_argument(
        "--presence-conf-threshold",
        default=0.25,
        type=float,
        help="Presence gate: drop masks with scores at or below this value.",
    )
    parser.add_argument(
        "--min-area",
        default=250,
        type=int,
        help="Presence gate: drop masks with pixel area at or below this value.",
    )
    parser.add_argument(
        "--det-threshold",
        default=None,
        type=float,
        help="Optional internal detector threshold override for probing weak prompts.",
    )
    parser.add_argument(
        "--resolution",
        default="HD720",
        choices=RESOLUTION_NAMES,
        help="ZED camera resolution preset.",
    )
    parser.add_argument(
        "--camera-fps",
        default=30,
        type=int,
        help="ZED camera FPS. Use 0 to leave the SDK default.",
    )
    parser.add_argument(
        "--view",
        default="LEFT",
        choices=VIEW_NAMES,
        help="ZED image view to retrieve and run SAM on.",
    )
    parser.add_argument(
        "--stereo-warp-mask",
        action="store_true",
        help=(
            "Retrieve the opposite ZED view and write a second overlay by "
            "warping the kept masks with ZED disparity instead of rerunning SAM."
        ),
    )
    parser.add_argument(
        "--stereo-frame-output",
        default=DEFAULT_STEREO_FRAME_OUTPUT,
        type=Path,
        help="Where to save the opposite-view RGB frame when --stereo-warp-mask is set.",
    )
    parser.add_argument(
        "--stereo-overlay-output",
        default=DEFAULT_STEREO_OVERLAY_OUTPUT,
        type=Path,
        help="Where to write the opposite-view warped-mask overlay.",
    )
    parser.add_argument(
        "--stereo-cleanup-kernel",
        default=3,
        type=int,
        help="Odd morphology kernel used to close small holes in warped stereo masks. Use 1 to disable.",
    )
    parser.add_argument(
        "--stereo-disparity-sign",
        default=None,
        type=int,
        choices=(-1, 1),
        help="Override disparity shift sign if the warped mask moves the wrong way.",
    )
    parser.add_argument(
        "--warmup-frames",
        default=5,
        type=int,
        help="Frames to grab before saving, so exposure has a moment to settle.",
    )
    parser.add_argument(
        "--grab-timeout",
        default=5.0,
        type=float,
        help="Seconds to wait for each successful ZED grab.",
    )
    parser.add_argument(
        "--use-fa3",
        action="store_true",
        help="Enable FlashAttention 3 for the SAM run.",
    )
    parser.add_argument(
        "--verbose-load",
        action="store_true",
        help="Print the full checkpoint load log from SAM 3.1.",
    )
    return parser.parse_args()


def import_zed() -> object:
    try:
        import pyzed.sl as sl
    except ImportError as exc:
        raise RuntimeError(
            "pyzed is not importable. Install the ZED SDK Python bindings first."
        ) from exc
    return sl


def open_zed(args: argparse.Namespace):
    sl = import_zed()
    init_params = sl.InitParameters()
    init_params.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    # Depth/point-cloud measures default to millimeters; keep everything metric.
    init_params.coordinate_units = sl.UNIT.METER
    if args.camera_fps > 0:
        init_params.camera_fps = args.camera_fps
    if getattr(args, "view", "LEFT") == "RIGHT":
        # MEASURE.DISPARITY_RIGHT (and other *_RIGHT measures) fail with
        # INVALID FUNCTION PARAMETERS unless right-side measures are enabled.
        init_params.enable_right_side_measure = True

    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {err}")
    return sl, zed


def wait_for_grab(sl: object, zed: object, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error = None
    while time.monotonic() < deadline:
        err = zed.grab()
        if err == sl.ERROR_CODE.SUCCESS:
            return
        last_error = err
        time.sleep(0.02)
    raise RuntimeError(f"ZED grab failed before timeout: {last_error}")


def retrieve_zed_rgb(sl: object, zed: object, view_name: str) -> np.ndarray:
    rgb_mat = sl.Mat()
    zed.retrieve_image(rgb_mat, getattr(sl.VIEW, view_name))
    bgra = rgb_mat.get_data()
    if bgra.ndim != 3 or bgra.shape[2] < 3:
        raise RuntimeError(f"Unexpected ZED image shape: {bgra.shape}")
    return bgra[:, :, :3][:, :, ::-1].copy()


def retrieve_zed_disparity(
    sl: object,
    zed: object,
    source_view_name: str,
) -> tuple[np.ndarray, str]:
    measure_name = stereo.disparity_measure_name(source_view_name)
    disparity_mat = sl.Mat()
    err = zed.retrieve_measure(disparity_mat, getattr(sl.MEASURE, measure_name))
    if err is not None and err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED retrieve_measure({measure_name}) failed: {err}")
    disparity = disparity_mat.get_data()
    if disparity.ndim == 3:
        disparity = disparity[:, :, 0]
    if disparity.ndim != 2:
        raise RuntimeError(f"Unexpected ZED disparity shape: {disparity.shape}")
    return disparity.astype(np.float32, copy=True), measure_name


def depth_measure_names(view_name: str) -> tuple[str, str]:
    if view_name == "LEFT":
        return "DEPTH", "XYZ"
    if view_name == "RIGHT":
        return "DEPTH_RIGHT", "XYZ_RIGHT"
    raise ValueError(f"Unsupported stereo view: {view_name}")


def retrieve_zed_depth_and_xyz(
    sl: object,
    zed: object,
    view_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Depth map and point cloud for the last grab, registered to `view_name`.

    Must be called while the camera is open, after the grab that produced the
    RGB frame, so all measures describe the same instant.
    """
    depth_name, xyz_name = depth_measure_names(view_name)

    depth_mat = sl.Mat()
    err = zed.retrieve_measure(depth_mat, getattr(sl.MEASURE, depth_name))
    if err is not None and err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED retrieve_measure({depth_name}) failed: {err}")
    depth = depth_mat.get_data()
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise RuntimeError(f"Unexpected ZED depth shape: {depth.shape}")

    xyz_mat = sl.Mat()
    err = zed.retrieve_measure(xyz_mat, getattr(sl.MEASURE, xyz_name))
    if err is not None and err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED retrieve_measure({xyz_name}) failed: {err}")
    xyz = xyz_mat.get_data()
    if xyz.ndim != 3 or xyz.shape[2] < 3:
        raise RuntimeError(f"Unexpected ZED XYZ shape: {xyz.shape}")

    return (
        depth.astype(np.float32, copy=True),
        xyz.astype(np.float32, copy=True),
        {"depth_measure": depth_name, "xyz_measure": xyz_name},
    )


def object_depth_report(
    kept: list[tuple[np.ndarray, float]],
    *,
    depth_np: np.ndarray,
    xyz_np: np.ndarray,
    view_name: str,
    depth_measure: str,
    xyz_measure: str,
) -> dict:
    objects = []
    for index, (mask, score) in enumerate(kept):
        objects.append(
            {
                "index": index,
                "score": float(score),
                "depth_stats_m": mask_depth.mask_depth_stats(mask, depth_np),
                "xyz_centroid_m": mask_depth.mask_xyz_centroid(mask, xyz_np),
            }
        )
    return {
        "view": view_name,
        "depth_measure": depth_measure,
        "xyz_measure": xyz_measure,
        "units": "meters",
        "objects": objects,
    }


def print_object_depths(report: dict) -> None:
    if not report["objects"]:
        print("object_depth: no masks kept, nothing to measure")
        return
    for entry in report["objects"]:
        stats = entry["depth_stats_m"]
        median = stats["median"]
        centroid = entry["xyz_centroid_m"]
        print(
            f"object[{entry['index']}]"
            f" depth_m={median if median is None else round(median, 3)}"
            f" xyz_centroid_m={centroid if centroid is None else [round(v, 3) for v in centroid]}"
            f" valid_fraction={round(stats['valid_fraction'], 3)}"
        )


def grab_zed_rgb(
    sl: object,
    zed: object,
    view_name: str,
    warmup_frames: int,
    grab_timeout: float,
) -> np.ndarray:
    for _ in range(max(warmup_frames, 0)):
        wait_for_grab(sl, zed, grab_timeout)

    wait_for_grab(sl, zed, grab_timeout)
    return retrieve_zed_rgb(sl, zed, view_name)


def grab_zed_stereo_rgb_and_disparity(
    sl: object,
    zed: object,
    source_view_name: str,
    warmup_frames: int,
    grab_timeout: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    for _ in range(max(warmup_frames, 0)):
        wait_for_grab(sl, zed, grab_timeout)

    wait_for_grab(sl, zed, grab_timeout)
    target_view_name = stereo.opposite_view(source_view_name)
    source_rgb = retrieve_zed_rgb(sl, zed, source_view_name)
    target_rgb = retrieve_zed_rgb(sl, zed, target_view_name)
    disparity, measure_name = retrieve_zed_disparity(sl, zed, source_view_name)
    return source_rgb, target_rgb, disparity, target_view_name, measure_name


def apply_crop(
    rgb_np: np.ndarray,
    crop: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, dict]:
    full_height, full_width = rgb_np.shape[:2]
    if crop is None:
        return rgb_np, {
            "enabled": False,
            "full_width": int(full_width),
            "full_height": int(full_height),
            "output_width": int(full_width),
            "output_height": int(full_height),
        }

    x, y, width, height = crop
    if x >= full_width or y >= full_height:
        raise ValueError(
            f"Crop origin ({x}, {y}) is outside frame {full_width}x{full_height}"
        )
    x2 = min(x + width, full_width)
    y2 = min(y + height, full_height)
    cropped = rgb_np[y:y2, x:x2].copy()
    return cropped, {
        "enabled": True,
        "full_width": int(full_width),
        "full_height": int(full_height),
        "requested_xywh": [int(x), int(y), int(width), int(height)],
        "applied_xyxy": [int(x), int(y), int(x2), int(y2)],
        "output_width": int(cropped.shape[1]),
        "output_height": int(cropped.shape[0]),
    }


def write_rgb_image(output_path: Path, rgb_np: np.ndarray) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    if not cv2.imwrite(str(tmp_path), rgb_np[:, :, ::-1]):
        raise OSError(f"Failed to write RGB image: {tmp_path}")
    tmp_path.replace(output_path)


def capture_processed_frame(
    sl: object,
    zed: object,
    args: argparse.Namespace,
    warmup_frames: int,
) -> tuple[np.ndarray, dict]:
    full_rgb = grab_zed_rgb(
        sl,
        zed,
        view_name=args.view,
        warmup_frames=warmup_frames,
        grab_timeout=args.grab_timeout,
    )
    rgb_np, crop_info = apply_crop(full_rgb, args.crop)
    frame_info = {
        "view": args.view,
        "resolution": args.resolution,
        "camera_fps": int(args.camera_fps),
        "crop": crop_info,
    }
    return rgb_np, frame_info


def capture_processed_stereo_frame(
    sl: object,
    zed: object,
    args: argparse.Namespace,
    warmup_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    source_full, target_full, disparity_full, target_view, measure_name = (
        grab_zed_stereo_rgb_and_disparity(
            sl,
            zed,
            source_view_name=args.view,
            warmup_frames=warmup_frames,
            grab_timeout=args.grab_timeout,
        )
    )
    source_rgb, crop_info = apply_crop(source_full, args.crop)
    target_rgb, _ = apply_crop(target_full, args.crop)
    disparity_np, _ = apply_crop(disparity_full, args.crop)
    frame_info = {
        "view": args.view,
        "target_view": target_view,
        "resolution": args.resolution,
        "camera_fps": int(args.camera_fps),
        "crop": crop_info,
        "stereo_warp": {
            "enabled": True,
            "source_view": args.view,
            "target_view": target_view,
            "disparity_measure": measure_name,
            "disparity_image": stereo.disparity_image_stats(disparity_np),
        },
    }
    return source_rgb, target_rgb, disparity_np, frame_info


def write_stereo_overlay(
    *,
    target_rgb_np: np.ndarray,
    source_kept: list[tuple[np.ndarray, float]],
    disparity_np: np.ndarray,
    label: str,
    output_path: Path,
    source_view_name: str,
    cleanup_kernel: int,
    configured_shift_sign: int | None,
) -> dict:
    import task2_sam31_image_prompt as task2

    shift_sign = stereo.resolve_shift_sign(source_view_name, configured_shift_sign)
    warped_kept, warp_stats = stereo.warp_kept_masks(
        source_kept,
        disparity_np,
        shift_sign=shift_sign,
        cleanup_kernel=cleanup_kernel,
    )
    overlay = task2.overlay_masks(
        target_rgb_np,
        warped_kept,
        label,
        output_path.expanduser().resolve(),
    )
    return {
        **overlay,
        "source_view": source_view_name,
        "target_view": stereo.opposite_view(source_view_name),
        "disparity_measure": stereo.disparity_measure_name(source_view_name),
        "shift_sign": int(shift_sign),
        "cleanup_kernel": int(cleanup_kernel),
        "mask_warp_stats": warp_stats,
    }


def run_preview(args: argparse.Namespace) -> None:
    sl, zed = open_zed(args)
    count = 0
    first = True
    frame_output = args.frame_output.expanduser().resolve()
    try:
        while args.preview_count == 0 or count < args.preview_count:
            if args.stereo_warp_mask:
                rgb_np, stereo_rgb, _, frame_info = capture_processed_stereo_frame(
                    sl,
                    zed,
                    args,
                    warmup_frames=args.warmup_frames if first else 0,
                )
                stereo_frame_output = args.stereo_frame_output.expanduser().resolve()
                write_rgb_image(stereo_frame_output, stereo_rgb)
            else:
                rgb_np, frame_info = capture_processed_frame(
                    sl,
                    zed,
                    args,
                    warmup_frames=args.warmup_frames if first else 0,
                )
                stereo_frame_output = None
            first = False
            count += 1
            write_rgb_image(frame_output, rgb_np)
            print(
                f"preview_frame={count} wrote={frame_output} "
                f"size={rgb_np.shape[1]}x{rgb_np.shape[0]} crop={frame_info['crop']}"
            )
            if stereo_frame_output is not None:
                print(f"preview_stereo_frame={count} wrote={stereo_frame_output}")
            if args.preview_count == 0 or count < args.preview_count:
                time.sleep(max(args.interval, 0.0))
    except KeyboardInterrupt:
        print("preview_stopped=keyboard_interrupt")
    finally:
        zed.close()


def run_segment(args: argparse.Namespace) -> dict:
    sl, zed = open_zed(args)
    frame_output = args.frame_output.expanduser().resolve()
    stereo_rgb = None
    disparity_np = None
    stereo_frame_output = args.stereo_frame_output.expanduser().resolve()
    try:
        if args.stereo_warp_mask:
            rgb_np, stereo_rgb, disparity_np, frame_info = capture_processed_stereo_frame(
                sl,
                zed,
                args,
                warmup_frames=args.warmup_frames,
            )
        else:
            rgb_np, frame_info = capture_processed_frame(
                sl,
                zed,
                args,
                warmup_frames=args.warmup_frames,
            )
        depth_np, xyz_np, depth_info = retrieve_zed_depth_and_xyz(sl, zed, args.view)
    finally:
        zed.close()

    depth_np, _ = apply_crop(depth_np, args.crop)
    xyz_np, _ = apply_crop(xyz_np, args.crop)

    write_rgb_image(frame_output, rgb_np)
    print(f"wrote_live_frame={frame_output}")
    if stereo_rgb is not None:
        write_rgb_image(stereo_frame_output, stereo_rgb)
        print(f"wrote_stereo_frame={stereo_frame_output}")

    import task2_sam31_image_prompt as task2

    sam_args = argparse.Namespace(
        image=frame_output,
        prompt=args.prompt,
        checkpoint=args.checkpoint,
        threshold=args.threshold,
        output_json=args.output_json,
        presence_conf_threshold=args.presence_conf_threshold,
        min_area=args.min_area,
        overlay_output=args.overlay_output,
        det_threshold=args.det_threshold,
        use_fa3=args.use_fa3,
        verbose_load=args.verbose_load,
    )
    result = task2.run_once(sam_args, include_kept_masks=True)
    kept = result.pop("_kept_masks", [])
    result["object_depth"] = object_depth_report(
        kept,
        depth_np=depth_np,
        xyz_np=xyz_np,
        view_name=args.view,
        **depth_info,
    )
    if args.stereo_warp_mask:
        result["stereo_overlay"] = write_stereo_overlay(
            target_rgb_np=stereo_rgb,
            source_kept=kept,
            disparity_np=disparity_np,
            label=args.prompt,
            output_path=args.stereo_overlay_output,
            source_view_name=args.view,
            cleanup_kernel=args.stereo_cleanup_kernel,
            configured_shift_sign=args.stereo_disparity_sign,
        )
    result["zed_frame"] = {
        **frame_info,
        "saved_frame": str(frame_output),
    }
    if args.stereo_warp_mask:
        result["zed_frame"]["stereo_saved_frame"] = str(stereo_frame_output)

    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={output_path}")
    if result.get("overlay") is not None:
        print(f"wrote_overlay={result['overlay']['output']}")
    if result.get("stereo_overlay") is not None:
        print(f"wrote_stereo_overlay={result['stereo_overlay']['output']}")
    print_object_depths(result["object_depth"])
    return result


def main() -> None:
    args = parse_args()
    if args.preview:
        run_preview(args)
    else:
        result = run_segment(args)
        gate_summary = result["presence_gate"]
        print(f"kept={gate_summary['num_kept']}")
        print(f"kept_scores={gate_summary['kept_scores']}")
        print(f"kept_areas_pixels={gate_summary['kept_areas_pixels']}")


if __name__ == "__main__":
    main()
