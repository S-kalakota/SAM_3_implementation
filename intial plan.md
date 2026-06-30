# Command to Segmentation — SAM 3.1 Version (use once access is granted)

> This is the SAM 3.1 path. The 3.1 checkpoints are gated behind Hugging Face access. Until that access lands, build with the **SAM 2 version** (`pipeline_sam2.md`), which is the same pipeline using Grounding DINO + SAM 2 and is fully open. When 3.1 access comes through, switch to this file: it collapses the separate detector and segmenter into one model, so the agent's segment tool becomes a single call.

Goal of this version: type a natural-language request, get the correct masks drawn on the image. One Qwen, used only as the brain of the SAM 3.1 Agent. The request goes straight into the agent; Qwen cleans up the wording and reasons about the image as part of proposing noun phrases to SAM 3.1. No voice yet (no mic on the Thor), no selection logic, no depth. When several instances match, we overlay all of them.

Built as ordered tasks across three milestones. Each ends with a concrete "done when" check.

---

## Data flow

```
 typed request ─┐
                ▼
 ZED 2i / saved image ─► RGB ─► SAM 3.1 Agent  (Qwen reasons, calls SAM 3.1, iterates)
                                          │
                                   masks + scores
                                          │
                               Presence gate (score + area)
                                          │
                          overlay ALL kept masks ─► result.png (verify)
```

Milestone 1 validates the SAM 3.1 tool directly with a typed phrase. That same tool is what the agent calls internally, so prove it works clean before wrapping Qwen around it.

---

# Milestone 1: masks on a still image (typed phrase)

This is the hard, important part, and it is the exact tool the agent will drive. Get it solid first.

## Task 1: Install SAM 3.1 and vision deps
**Do:**
- Request access to the `facebook/sam3.1` checkpoints on Hugging Face, then `hf auth login`.
- Clone the official repo and install (3.1 ships only as a gated checkpoint, no Transformers integration, so the code comes from the repo):

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
pip install einops ninja
pip install flash-attn-3 --no-deps --index-url https://download.pytorch.org/whl/cu128
pip install torch torchvision numpy opencv-python pillow
# download the SAM 3.1 checkpoint (multiplex fp16 variant) from facebook/sam3.1
# see RELEASE_SAM3p1.md for exact checkpoint names
```

**Done when:** `python -c "from sam3.model_builder import build_sam3_image_model"` runs with no error and the checkpoint file is on disk.

---

## Task 2: Get masks from SAM 3.1 on a saved image
**Do:**
- Drop a test photo at `test.jpg` (a couple of objects in a box, like your real scene).
- Load the model once, set the image, prompt with a hard-coded phrase, read back masks and scores.

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(checkpoint_path="sam3.1_multiplex_fp16.safetensors",
                               load_from_HF=False)
model = model.half().cuda()              # FP16, ~4 GB VRAM
processor = Sam3Processor(model, device="cuda")

phrase = "red cup"                        # typed for now
state = processor.set_image(Image.open("test.jpg"))
out = processor.set_text_prompt(state=state, prompt=phrase)
masks, scores = out["masks"], out["scores"]
print(len(masks), [float(s) for s in scores])
```

**Done when:** a phrase that IS in the image returns one or more masks with reasonable scores, and a phrase that is NOT in the image returns nothing (or only very low scores).

---

## Task 3: Add the presence gate
**Do:**
- Convert each mask to a boolean numpy array, then drop anything below a score threshold or a minimum pixel area. Tune the two constants against `test.jpg`.

```python
import numpy as np
CONF_THRESH, MIN_AREA = 0.5, 500          # tune these on your image

def gate(masks, scores):
    kept = []
    for m, s in zip(masks, scores):
        mb = (m > 0.5).squeeze().cpu().numpy().astype(bool)   # HxW bool
        if float(s) > CONF_THRESH and mb.sum() > MIN_AREA:
            kept.append((mb, float(s)))
    return kept

kept = gate(masks, scores)
print(len(kept), "kept")
```

**Done when:** tiny/low-confidence false positives are filtered out, and a phrase not in the image yields `kept == []`.

---

## Task 4: Overlay all kept masks and verify
**Do:**
- Paint every kept mask onto the image, label each with its score, write `result.png`.

```python
import cv2
def overlay_masks(rgb_np, kept, label):
    out = rgb_np.copy()
    for mb, s in kept:
        out[mb] = (0.5 * out[mb] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        ys, xs = np.where(mb)
        cv2.putText(out, f"{label} {s:.2f}",
                    (int(xs.min()), max(int(ys.min()) - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite("result.png", out[:, :, ::-1])            # RGB -> BGR for cv2

rgb_np = np.array(Image.open("test.jpg"))[:, :, :3]
overlay_masks(rgb_np, kept, phrase)
```

**Done when:** `result.png` highlights the right object(s) for several typed phrases. `result.png` is now your tuning signal for `CONF_THRESH`, `MIN_AREA`, and phrasing.

**Milestone 1 complete:** the SAM 3.1 tool produces correct masks from a clean phrase.

---

# Milestone 2: live camera

## Task 5: Swap the saved image for a ZED RGB frame
**Do:**
- Install the ZED SDK and `pyzed`. Grab the left RGB and feed it where `test.jpg` was. Depth is not used.

```python
import pyzed.sl as sl
zed = sl.Camera()
zed.open(sl.InitParameters())             # defaults fine for RGB
rgb = sl.Mat()
if zed.grab() == sl.ERROR_CODE.SUCCESS:
    zed.retrieve_image(rgb, sl.VIEW.LEFT)             # HxWx4 BGRA
rgb_np = rgb.get_data()[:, :, :3][:, :, ::-1].copy()  # HxWx3 RGB
```

**Done when:** a live grab plus a typed phrase produces a correct `result.png`.

---

# Milestone 3: natural language in, through the agent

## Task 6: SAM 3.1 Agent with your Qwen
**Do:**
- Wrap your already-running Qwen in one function. It takes the image so it can see what it is choosing among. (Qwen3-VL-8B recommended for the agent role; it must be a VL model.)

```python
# local_qwen.py — adapter over your local Qwen-VL (Qwen3-VL-8B recommended)
def qwen_generate(messages, images=None) -> str:
    ...                                   # your existing Qwen load + generate
    return text                           # pass images for the agent's vision calls
```

- Use the repo's `examples/sam3_agent` as the scaffold. It already drives SAM 3.1 (the same `processor` from Task 2) and renders masks back to an MLLM. Replace its MLLM hook with `qwen_generate`, passing the current image with masks drawn on it.

```python
from local_qwen import qwen_generate
# inside the agent's MLLM hook:
reply = qwen_generate(messages, images=[pil_image_with_rendered_masks])
```

- Drive the agent with the image and a typed natural-language request. Test a simple one ("the red cup") and a relational one ("the cup you could drink from"). The agent returns final masks and scores, same shape Task 3 expects.

```python
masks, scores = run_agent(image=Image.fromarray(rgb_np), request="the red cup")
# run_agent stands in for the agent entry point; follow examples/sam3_agent for the exact call
kept = gate(masks, scores)
overlay_masks(rgb_np, kept, "the red cup")
```

**Done when:** a typed natural-language request returns correct masks for both a simple and a relational phrasing, with no hand-cleaning of the wording.

**Note on cost:** every request goes through the agent, including easy ones, so each pays the agent's multi-call latency. For a few objects on a demo that is fine. If it drags later, add a fast direct-`set_text_prompt` path for clean phrases and reserve the agent for the hard ones.

---

## Task 7: Wire the loop
**Do:**
- Chain it: read a typed request -> grab a ZED frame -> feed the request to the agent -> gate -> overlay. Load Qwen and SAM 3.1 once at startup and keep them resident.

```python
# per command:
request = input("command> ")                     # typed for now (no mic on the Thor yet)
rgb_np = grab_zed_rgb()                           # Task 5
masks, scores = run_agent(image=Image.fromarray(rgb_np), request=request)
kept = gate(masks, scores)                        # Task 3
overlay_masks(rgb_np, kept, request)              # Task 4
```

**Done when:** you type a command and `result.png` shows the right masks.

---

# Later (not now)

- **Voice input.** Add Whisper (`faster-whisper`) plus `sounddevice` back once a mic is connected to the Thor. It just produces the same request string the agent already takes, so it slots in ahead of Task 7 with nothing else changing.
- **TensorRT deployment.** Swap the prototype SAM 3.1 load for an exported FP16 TensorRT engine once masks are correct, since the plain load is slow on edge. The agent benefits most, since it calls SAM 3.1 several times per request.
- **Depth and grasping.** Re-add the ZED XYZ measure to get 3D points per mask, then hand off to AnyGrasp.
