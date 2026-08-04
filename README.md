# Co-Bot VLM

Lightweight skeleton for a voice/text to visual-grounding verification pipeline.

This phase transcribes or accepts a command, parses a structured intent, then
asks the VLM only to ground the requested target object in a single RGB image.
It does not produce robot poses, joint angles, gripper commands, trajectories,
or motion commands.

## Current Flow

1. `Whisper` or `--text` produces a transcript string.
2. The instruction parser extracts an open-vocabulary target object phrase and
   intent:

   ```json
   {
     "action": "pick_and_place",
     "object": "blue box",
     "source": "left bin",
     "destination": "right bin"
   }
   ```

3. The Qwen/mock VLM receives only the parsed target object and RGB image. It
   answers whether that object is visible in the frame.

4. The visual gate approves only when the requested object is present in the
   frame.

## Current Visual Approval Guidelines

Approval means the requested object was visually verified in the current frame.
It does not mean the robot is allowed to move yet.

The target object is open vocabulary: `green water bottle`, `blue block`,
`orange object`, or another noun phrase can be parsed and sent to the VLM.

Visual approval requires only that the object is marked visible in the current
frame. Qwen is prompted to answer exactly `YES` or `NO`; the adapter converts
that to the internal result.

The VLM grounding schema is `object-grounding-v1`. Qwen output with
`action`, `source`, `destination`, robot poses, trajectories, gripper commands,
or other motion fields is rejected because those belong to later robot-control
stages. The VLM is prompted to answer only whether the requested object exists
in the frame.

## Run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

```bash
.venv/bin/python -m co_bot_vlm.cli --text "pick up the green water bottle to the drop zone" --image-file path/to/frame.jpg
```

The default `mock` VLM backend is deterministic. With `--image-file`, it does
not require camera hardware, model weights, Transformers, or the ZED SDK.

```bash
.venv/bin/python -m co_bot_vlm.cli --help
.venv/bin/python -m co_bot_vlm.cli --vlm-backend qwen --text "pick up the green water bottle" --image-file path/to/frame.jpg
```

Qwen setup is optional and heavier than the mock backend:

```bash
.venv/bin/python -m pip install -e ".[qwen]"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", repo_type="model")
PY
```

The default local Qwen model is `Qwen/Qwen2.5-VL-7B-Instruct`. Override it with
`--qwen-model` or `CO_BOT_VLM_QWEN_MODEL_ID`. On macOS, the Qwen backend defaults
to `CO_BOT_VLM_QWEN_DEVICE_MAP=cpu` because the Apple MPS backend can exceed
Metal temporary tensor limits for this model. CPU inference is slower but avoids
hard crashes.

For a live camera snapshot, install the project dependencies and pass a generic
camera index:

```bash
.venv/bin/python -m co_bot_vlm.cli --text "pick up the green water bottle to the drop zone" --camera-index 0
```

The `--camera-index` path captures one warmed-up RGB frame through OpenCV and
saves a temporary snapshot for downstream VLM backends. A ZED 2i can be used
only if the OS exposes it as a normal RGB video device; this phase does not
require the ZED SDK.

For a continuous camera check, add `--live`. Use `--max-frames` for a bounded
test or omit it and stop with Ctrl-C:

```bash
.venv/bin/co-bot-vlm --live --camera-index 0 --max-frames 5 --text "pick up the green water bottle to the drop zone" --plain
.venv/bin/co-bot-vlm --live --camera-index 0 --stop-on-approved --text "pick up the blue block to the bin" --plain
```

For spoken commands, install the voice extras, download the default Whisper
model, and make sure SoX's `rec` command is available:

```bash
.venv/bin/python -m pip install -e ".[voice]"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("openai/whisper-tiny.en", repo_type="model")
PY
```

Then run with `--voice`. The command records for `--voice-duration` seconds,
transcribes the microphone audio with Whisper, parses the intent, and sends only
the requested target object into the same image/VLM grounding pipeline:

```bash
.venv/bin/co-bot-vlm --voice --voice-duration 5 --vlm-backend qwen --image-file path/to/frame.jpg --pretty
```

Override the Whisper model with `--whisper-model` or
`CO_BOT_VLM_WHISPER_MODEL_ID`. On macOS, grant microphone permission to the
terminal app running the command.
