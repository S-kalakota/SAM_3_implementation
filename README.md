# Co-Bot VLM

Lightweight skeleton for a VLM-based object existence verification pipeline.

This phase verifies that a requested object is visually present in a single RGB
image. It does not produce robot poses, joint angles, gripper commands,
trajectories, or motion commands.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

```bash
.venv/bin/python -m co_bot_vlm.cli --text "pick up the red cup to the drop zone" --image-file path/to/frame.jpg
```

The default `mock` VLM backend is deterministic. With `--image-file`, it does
not require camera hardware, model weights, Transformers, or the ZED SDK.

```bash
.venv/bin/python -m co_bot_vlm.cli --help
.venv/bin/python -m co_bot_vlm.cli --vlm-backend qwen --text "pick up the red cup" --image-file path/to/frame.jpg
```

For a live camera snapshot, install the project dependencies and pass a generic
camera index:

```bash
.venv/bin/python -m co_bot_vlm.cli --text "pick up the red cup to the drop zone" --camera-index 0
```

The `--camera-index` path captures one warmed-up RGB frame through OpenCV and
saves a temporary snapshot for downstream VLM backends. A ZED 2i can be used
only if the OS exposes it as a normal RGB video device; this phase does not
require the ZED SDK.

The `qwen`, audio, and voice paths are intentionally clear stubs for later
agents.
