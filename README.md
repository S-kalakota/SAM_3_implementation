# Co-Bot VLM

Dependency-free skeleton for a VLM-based object existence verification pipeline.

This phase verifies that a requested object is visually present in a single RGB
image. It does not produce robot poses, joint angles, gripper commands,
trajectories, or motion commands.

## Run

```bash
python -m co_bot_vlm.cli --text "pick up the red cup to the drop zone" --image-file path/to/frame.jpg
```

The default `mock` VLM backend is deterministic and does not require camera
hardware, model weights, OpenCV, Transformers, or the ZED SDK.

```bash
python -m co_bot_vlm.cli --help
python -m co_bot_vlm.cli --vlm-backend qwen --text "pick up the red cup" --image-file path/to/frame.jpg
```

The `qwen`, camera, audio, and voice paths are intentionally clear stubs for
later agents.
