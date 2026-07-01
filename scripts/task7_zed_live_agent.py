#!/usr/bin/env python3
"""Task 7 smoke test: grab a live ZED frame and run the Qwen-backed SAM agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import local_qwen
import task2_sam31_image_prompt as task2
import task5_zed_live_prompt as task5
import task6_sam31_agent as task6


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_OUTPUT = PROJECT_ROOT / "outputs/task7_live_agent_frame.png"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/task7_zed_live_agent.json"
DEFAULT_OVERLAY_OUTPUT = PROJECT_ROOT / "outputs/result_live_agent.png"
DEFAULT_AGENT_RENDER_OUTPUT = PROJECT_ROOT / "outputs/result_live_agent_meta.png"
DEFAULT_AGENT_OUTPUT_DIR = PROJECT_ROOT / "outputs/task7_live_agent_workspace"


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
        default=DEFAULT_FRAME_OUTPUT,
        type=Path,
        help="Where to save the live RGB frame before the agent runs.",
    )
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--overlay-output", default=DEFAULT_OVERLAY_OUTPUT, type=Path)
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
    parser.add_argument("--crop", type=task5.parse_crop)
    parser.add_argument("--resolution", default="HD720", choices=task5.RESOLUTION_NAMES)
    parser.add_argument("--camera-fps", default=30, type=int)
    parser.add_argument("--view", default="LEFT", choices=task5.VIEW_NAMES)
    parser.add_argument("--warmup-frames", default=5, type=int)
    parser.add_argument("--grab-timeout", default=5.0, type=float)
    parser.add_argument("--use-fa3", action="store_true")
    parser.add_argument("--verbose-load", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def run_live_agent(args: argparse.Namespace) -> dict:
    sl, zed = task5.open_zed(args)
    frame_output = args.frame_output.expanduser().resolve()
    try:
        rgb_np, frame_info = task5.capture_processed_frame(
            sl,
            zed,
            args,
            warmup_frames=args.warmup_frames,
        )
    finally:
        zed.close()

    task5.write_rgb_image(frame_output, rgb_np)
    print(f"wrote_live_frame={frame_output}")

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
    result["zed_frame"] = {
        **frame_info,
        "saved_frame": str(frame_output),
    }

    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()
