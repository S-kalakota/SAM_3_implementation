#!/usr/bin/env python3
"""Task 7 smoke test: grab a live ZED frame and run the Qwen-backed SAM agent."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import local_qwen
import stereo_mask_warp as stereo
import task2_sam31_image_prompt as task2
import task5_zed_live_prompt as task5
import task6_sam31_agent as task6


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_OUTPUT = PROJECT_ROOT / "outputs/task7_live_agent_frame.png"
DEFAULT_STEREO_FRAME_OUTPUT = PROJECT_ROOT / "outputs/task7_live_agent_frame_stereo.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/task7_zed_live_agent.json"
DEFAULT_OVERLAY_OUTPUT = PROJECT_ROOT / "outputs/result_live_agent.png"
DEFAULT_STEREO_OVERLAY_OUTPUT = PROJECT_ROOT / "outputs/result_live_agent_stereo.png"
DEFAULT_AGENT_RENDER_OUTPUT = PROJECT_ROOT / "outputs/result_live_agent_meta.png"
DEFAULT_AGENT_OUTPUT_DIR = PROJECT_ROOT / "outputs/task7_live_agent_workspace"
DEFAULT_LOOP_OUTPUT_ROOT = PROJECT_ROOT / "outputs/task7_live_agent_loop"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be 1 or greater")
    return parsed


def apply_view_output_defaults(args: argparse.Namespace) -> argparse.Namespace:
    source_view = args.view.lower()
    target_view = stereo.opposite_view(args.view).lower()
    output_dir = PROJECT_ROOT / "outputs"
    defaults = {
        "frame_output": output_dir / f"{source_view}_view.png",
        "stereo_frame_output": output_dir / f"{target_view}_view.png",
        "overlay_output": output_dir / f"{source_view}_view_overlay.png",
        "stereo_overlay_output": output_dir / f"{target_view}_view_overlay.png",
    }
    for name, path in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, path)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grab one live ZED RGB frame, then run Meta's SAM 3.1 agent with "
            "local Qwen-VL on that frame."
        )
    )
    parser.add_argument(
        "--request",
        "--prompt",
        dest="request",
        required=True,
        help=(
            "Natural-language request for the agent. --prompt is accepted as "
            "an alias for compatibility with the Task 5 command."
        ),
    )
    parser.add_argument(
        "--frame-output",
        default=None,
        type=Path,
        help="Where to save the live RGB frame before the agent runs.",
    )
    parser.add_argument(
        "--stereo-frame-output",
        default=None,
        type=Path,
        help="Where to save the opposite-view RGB frame when --stereo-warp-mask is set.",
    )
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--overlay-output", default=None, type=Path)
    parser.add_argument("--stereo-overlay-output", default=None, type=Path)
    parser.add_argument("--agent-render-output", default=DEFAULT_AGENT_RENDER_OUTPUT, type=Path)
    parser.add_argument("--agent-output-dir", default=DEFAULT_AGENT_OUTPUT_DIR, type=Path)
    parser.add_argument("--checkpoint", default=task6.DEFAULT_CHECKPOINT, type=Path)
    parser.add_argument("--threshold", default=0.05, type=float)
    parser.add_argument("--presence-conf-threshold", default=task2.DEFAULT_PRESENCE_CONF_THRESH, type=float)
    parser.add_argument("--min-area", default=task2.DEFAULT_MIN_AREA, type=int)
    parser.add_argument("--det-threshold", default=None, type=float)
    parser.add_argument("--max-generations", default=8, type=int)
    parser.add_argument("--qwen-model", default=local_qwen.DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-max-new-tokens", default=local_qwen.DEFAULT_MAX_NEW_TOKENS, type=int)
    parser.add_argument("--qwen-device-map", default=local_qwen.DEFAULT_DEVICE_MAP)
    parser.add_argument(
        "--allow-qwen-downloads",
        action="store_true",
        help="Allow Transformers to fetch missing Qwen files. Default is local cache only.",
    )
    task5.add_crop_arguments(parser)
    parser.add_argument("--resolution", default="HD720", choices=task5.RESOLUTION_NAMES)
    parser.add_argument("--camera-fps", default=30, type=int)
    parser.add_argument("--view", default="LEFT", choices=task5.VIEW_NAMES)
    parser.add_argument(
        "--stereo-warp-mask",
        action="store_true",
        help=(
            "Retrieve the opposite ZED view and write a second overlay by "
            "warping the kept masks with ZED disparity instead of rerunning SAM."
        ),
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
    parser.add_argument("--warmup-frames", default=5, type=int)
    parser.add_argument("--grab-timeout", default=5.0, type=float)
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--verbose-load", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Keep running live-agent captures until Ctrl-C.",
    )
    parser.add_argument(
        "--loop-output-root",
        default=DEFAULT_LOOP_OUTPUT_ROOT,
        type=Path,
        help="Run folder where round_### loop outputs are written.",
    )
    parser.add_argument(
        "--loop-start-index",
        default=1,
        type=positive_int,
        help="Starting index for loop output folders.",
    )
    parser.add_argument(
        "--loop-interval",
        default=0.0,
        type=float,
        help="Seconds to wait between loop runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In loop mode, record errors and continue instead of stopping.",
    )
    return apply_view_output_defaults(parser.parse_args())


def run_live_agent(args: argparse.Namespace) -> dict:
    sl, zed = task5.open_zed(args)
    frame_output = args.frame_output.expanduser().resolve()
    stereo_rgb = None
    disparity_np = None
    stereo_frame_output = args.stereo_frame_output.expanduser().resolve()
    try:
        if args.stereo_warp_mask:
            rgb_np, stereo_rgb, disparity_np, frame_info = task5.capture_processed_stereo_frame(
                sl,
                zed,
                args,
                warmup_frames=args.warmup_frames,
            )
        else:
            rgb_np, frame_info = task5.capture_processed_frame(
                sl,
                zed,
                args,
                warmup_frames=args.warmup_frames,
            )
        depth_np, xyz_np, depth_info = task5.retrieve_zed_depth_and_xyz(
            sl, zed, args.view
        )
    finally:
        zed.close()

    depth_np, _ = task5.apply_crop(depth_np, args.crop)
    xyz_np, _ = task5.apply_crop(xyz_np, args.crop)

    task5.write_rgb_image(frame_output, rgb_np)
    print(f"wrote_live_frame={frame_output}")
    if stereo_rgb is not None:
        task5.write_rgb_image(stereo_frame_output, stereo_rgb)
        print(f"wrote_stereo_frame={stereo_frame_output}")

    agent_args = argparse.Namespace(
        image=frame_output,
        request=args.request,
        checkpoint=args.checkpoint,
        output_json=args.output_json,
        overlay_output=args.overlay_output,
        agent_render_output=args.agent_render_output,
        agent_output_dir=args.agent_output_dir,
        qwen_model=args.qwen_model,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
        allow_qwen_downloads=args.allow_qwen_downloads,
        qwen_device_map=args.qwen_device_map,
        threshold=args.threshold,
        presence_conf_threshold=args.presence_conf_threshold,
        min_area=args.min_area,
        det_threshold=args.det_threshold,
        max_generations=args.max_generations,
        use_fa3=args.use_fa3,
        verbose_load=args.verbose_load,
        debug=args.debug,
    )
    result = task6.run_agent(agent_args)
    masks = task6.decode_agent_masks(result["agent_final_outputs"])
    scores = np.asarray(
        result["agent_final_outputs"].get("pred_scores", []),
        dtype=np.float32,
    )
    kept, _ = task2.gate_masks(
        masks,
        scores,
        conf_thresh=args.presence_conf_threshold,
        min_area=args.min_area,
    )
    result["object_depth"] = task5.object_depth_report(
        kept,
        depth_np=depth_np,
        xyz_np=xyz_np,
        view_name=args.view,
        **depth_info,
    )
    task5.print_object_depths(result["object_depth"])
    if args.stereo_warp_mask:
        result["stereo_overlay"] = task5.write_stereo_overlay(
            target_rgb_np=stereo_rgb,
            source_kept=kept,
            disparity_np=disparity_np,
            label=args.request,
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
    return result


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_loop_summary(output_root: Path, summary: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def loop_run_args(
    args: argparse.Namespace,
    *,
    round_dir: Path,
    agent_output_dir: Path,
) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    source_view = args.view.lower()
    target_view = stereo.opposite_view(args.view).lower()
    run_args.frame_output = round_dir / f"{source_view}_view.png"
    run_args.stereo_frame_output = round_dir / f"{target_view}_view.png"
    run_args.output_json = round_dir / "result.json"
    run_args.overlay_output = round_dir / f"{source_view}_view_overlay.png"
    run_args.stereo_overlay_output = round_dir / f"{target_view}_view_overlay.png"
    run_args.agent_render_output = round_dir / "agent_render.png"
    run_args.agent_output_dir = agent_output_dir
    return run_args


def loop_record(*, index: int, round_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    gate = result.get("presence_gate", {})
    zed_frame = result.get("zed_frame", {})
    source_view = zed_frame.get("view")
    target_view = zed_frame.get("target_view")

    frame_by_view = {}
    overlay_by_view = {}
    if source_view is not None:
        frame_by_view[source_view] = zed_frame.get("saved_frame")
        overlay_by_view[source_view] = result.get("overlay", {}).get("output")
    if target_view is not None:
        frame_by_view[target_view] = zed_frame.get("stereo_saved_frame")
        overlay_by_view[target_view] = result.get("stereo_overlay", {}).get("output")

    return {
        "round": index,
        "status": "ok",
        "round_dir": str(round_dir),
        "result_json": str(round_dir / "result.json"),
        "source_view": source_view,
        "target_view": target_view,
        "left_view_frame": frame_by_view.get("LEFT"),
        "right_view_frame": frame_by_view.get("RIGHT"),
        "left_view_overlay": overlay_by_view.get("LEFT"),
        "right_view_overlay": overlay_by_view.get("RIGHT"),
        "frame": zed_frame.get("saved_frame"),
        "overlay": result.get("overlay", {}).get("output"),
        "stereo_overlay": result.get("stereo_overlay", {}).get("output"),
        "agent_render_output": result.get("agent_render_output"),
        "num_candidates": gate.get("num_candidates"),
        "num_kept": gate.get("num_kept"),
        "kept_scores": gate.get("kept_scores"),
        "kept_areas_pixels": gate.get("kept_areas_pixels"),
        "object_median_depths_m": [
            obj["depth_stats_m"]["median"]
            for obj in (result.get("object_depth") or {}).get("objects", [])
        ],
        "object_xyz_centroids_m": [
            obj["xyz_centroid_m"]
            for obj in (result.get("object_depth") or {}).get("objects", [])
        ],
    }


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.loop_output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "started_at": timestamp(),
        "finished_at": None,
        "request": args.request,
        "crop": list(args.crop) if args.crop is not None else None,
        "output_root": str(output_root),
        "runs": [],
    }
    write_loop_summary(output_root, summary)

    index = args.loop_start_index
    print(f"loop_output_root={output_root}")
    print("loop_status=running; press Ctrl-C to stop")
    try:
        while True:
            round_dir = output_root / f"round_{index:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            print("=" * 72)
            print(f"LOOP ROUND {index} START")
            print(f"loop_round={index}")
            print(f"loop_round_output={round_dir}")
            print("Internal agent reasoning rounds may print as 'Round 1', 'Round 2', etc.")
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"task7_round_{index:03d}_agent_"
                ) as agent_dir:
                    print(f"INTERNAL AGENT REASONING START for loop_round={index}")
                    result = run_live_agent(
                        loop_run_args(
                            args,
                            round_dir=round_dir,
                            agent_output_dir=Path(agent_dir),
                        )
                    )
                    print(f"INTERNAL AGENT REASONING END for loop_round={index}")
            except Exception as exc:
                record = {
                    "round": index,
                    "status": "error",
                    "round_dir": str(round_dir),
                    "error": repr(exc),
                }
                summary["runs"].append(record)
                write_loop_summary(output_root, summary)
                print(f"LOOP ROUND {index} ERROR: {exc!r}")
                if not args.continue_on_error:
                    raise
            else:
                record = loop_record(index=index, round_dir=round_dir, result=result)
                summary["runs"].append(record)
                write_loop_summary(output_root, summary)
                print(
                    f"LOOP ROUND {index} COMPLETE "
                    f"kept={record['num_kept']} "
                    f"candidates={record['num_candidates']} "
                    f"result_json={record['result_json']}"
                )

            index += 1
            time.sleep(max(args.loop_interval, 0.0))
    except KeyboardInterrupt:
        print("loop_status=stopped_by_keyboard_interrupt")
    finally:
        summary["finished_at"] = timestamp()
        write_loop_summary(output_root, summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.loop:
        summary = run_loop(args)
        print(f"wrote_summary={Path(summary['output_root']) / 'summary.json'}")
        return

    result = run_live_agent(args)
    gate = result["presence_gate"]
    print(f"request={result['request']!r}")
    print(f"qwen_model={result['qwen_model']}")
    print(f"kept={gate['num_kept']}")
    print(f"kept_scores={gate['kept_scores']}")
    print(f"kept_areas_pixels={gate['kept_areas_pixels']}")
    print(f"wrote={args.output_json.expanduser().resolve()}")
    print(f"wrote_agent_render={result['agent_render_output']}")
    print(f"wrote_overlay={result['overlay']['output']}")
    if result.get("stereo_overlay") is not None:
        print(f"wrote_stereo_overlay={result['stereo_overlay']['output']}")


if __name__ == "__main__":
    main()
