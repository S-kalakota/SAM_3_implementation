# Daemon plan — a resident masking service

Companion to `Second_plan.md`. Solves one problem: in one-shot use, ~95 % of wall-clock is loading SAM 3.1 (3.3 GB) + Qwen before the first mask. Fix: one long-lived process loads the models **once** and holds them warm on the GPU; every "one run, one round" request afterwards is served in seconds. Request semantics don't change — each call still grabs a fresh frame and does a single round.

This service also becomes the natural masking entry point for the Fairino executive later (Second_plan, Milestone D): the executive calls `/segment` and gets masks + depth back as JSON.

## Architecture

```
                     ┌───────────────────────────────────────────┐
  starts once        │  mask_service.py  (one process, GPU-warm) │
  (~1 min load)      │                                           │
                     │  SAM 3.1 resident ── Qwen resident (lazy) │
                     │  ZED 2i held open                         │
                     │  FastAPI on 127.0.0.1:8765                │
                     └───────────▲────────────────┬──────────────┘
                                 │ POST /segment  │ JSON reply (~1–2 s)
                                 │ {"request":    │ masks, scores, depth,
                                 │  "yellow box"} │ xyz centroids, file paths
                     ┌───────────┴────────────────▼──────────────┐
                     │ callers: curl / mask_client.py / executive │
                     └────────────────────────────────────────────┘
```

Design decisions (and why):

- **HTTP on localhost, FastAPI + uvicorn.** Ten lines, testable with `curl`, and the executive can be any language. Bound to `127.0.0.1` only — nothing off the Thor can reach it.
- **Camera held open by the service.** Opening the ZED costs ~2 s and exposure needs settling; holding it open makes frames instant. Consequence: task5/task7 scripts can't open the camera while the service runs — stop the service first (`systemctl stop mask-service`) when running them manually.
- **Fast path first, agent as fallback.** Direct SAM call with the request text (one forward pass); the multi-turn Qwen agent runs only if the direct call gates to zero masks or the caller forces it. Most requests ("yellow box") never touch Qwen.
- **One request at a time.** A global lock serializes requests — one GPU, one camera, one robot. A second caller just waits.
- **Reuse, don't rewrite.** The service imports the pieces that already work: `MultiplexSam3AgentService` (task6) for SAM, `local_qwen` (already `@lru_cache`d) for the fallback brain, `task5` capture/depth/report helpers, `task2.gate_masks`.

---

## Task 1: dependencies

**Do:**
```bash
cd ~/VLA_Model_Work/SAM_3_implementation
.venv/bin/pip install fastapi uvicorn
```
**Done when:** `.venv/bin/python -c "import fastapi, uvicorn"` is silent.

---

## Task 2: service skeleton — load once, prove it's warm

**Do:** create `scripts/mask_service.py`. Startup loads SAM and opens the camera; `/health` proves both without doing any work.

```python
#!/usr/bin/env python3
"""Resident masking service: SAM 3.1 + ZED held warm, served over localhost."""
import argparse, threading, time
from pathlib import Path

from fastapi import FastAPI, HTTPException
import uvicorn

import task2_sam31_image_prompt as task2
import task5_zed_live_prompt as task5
import task6_sam31_agent as task6

app = FastAPI()
LOCK = threading.Lock()          # one GPU, one camera: serialize requests
STATE: dict = {}                 # sam, sl, zed, args — filled at startup

RUN_DIR = Path(__file__).resolve().parents[1] / "outputs" / "service"

def startup(args):
    print("loading SAM 3.1 (once)...")
    STATE["sam"] = task6.MultiplexSam3AgentService(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        det_threshold=None,
        use_fa3=args.use_fa3,
        verbose_load=False,
    )
    print("opening ZED (held open)...")
    STATE["sl"], STATE["zed"] = task5.open_zed(args)
    STATE["args"] = args
    print("service ready")

@app.get("/health")
def health():
    return {
        "sam_loaded": "sam" in STATE,
        "camera_open": "zed" in STATE,
        "uptime_s": round(time.monotonic() - STATE.get("t0", 0), 1),
    }
```

**Done when:** `uvicorn` prints `service ready` once at startup, and
`curl -s localhost:8765/health` returns `{"sam_loaded": true, "camera_open": true, ...}` instantly.

---

## Task 3: `/segment` — fresh frame, fast path, agent fallback

**Do:** add the endpoint. Flow per request: flush stale frames → grab RGB + depth + XYZ (same instant) → direct SAM with the request text → gate → only if empty (and allowed), run the full Qwen agent → depth report → JSON.

```python
@app.post("/segment")
def segment(request: str, use_agent_fallback: bool = True):
    with LOCK:
        t0 = time.monotonic()
        args, sl, zed = STATE["args"], STATE["sl"], STATE["zed"]

        # camera has been streaming; flush buffered frames so this is "now"
        rgb_np, frame_info = task5.capture_processed_frame(
            sl, zed, args, warmup_frames=2)
        depth_np, xyz_np, depth_info = task5.retrieve_zed_depth_and_xyz(
            sl, zed, args.view)
        depth_np, _ = task5.apply_crop(depth_np, args.crop)
        xyz_np, _ = task5.apply_crop(xyz_np, args.crop)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        req_dir = RUN_DIR / stamp
        req_dir.mkdir(parents=True, exist_ok=True)
        frame_path = req_dir / "frame.png"
        task5.write_rgb_image(frame_path, rgb_np)

        # ---- fast path: one SAM forward, no LLM ----
        sam_json = STATE["sam"](image_path=str(frame_path),
                                text_prompt=request,
                                output_folder_path=str(req_dir))
        outputs = task6.load_json(sam_json)          # or json.load(open(...))
        masks = task6.decode_agent_masks(outputs)
        scores = task2.to_numpy_array(outputs["pred_scores"]).reshape(-1)
        kept, _ = task2.gate_masks(masks, scores,
                                   conf_thresh=args.presence_conf_threshold,
                                   min_area=args.min_area)
        path_used = "direct"

        # ---- fallback: full agent, only when direct found nothing ----
        if not kept and use_agent_fallback:
            path_used = "agent"
            kept = run_agent_fallback(request, frame_path, req_dir, args)

        report = task5.object_depth_report(
            kept, depth_np=depth_np, xyz_np=xyz_np,
            view_name=args.view, **depth_info)
        return {
            "request": request,
            "path": path_used,
            "num_kept": len(kept),
            "scores": [float(s) for _, s in kept],
            "object_depth": report,
            "frame": str(frame_path),
            "elapsed_s": round(time.monotonic() - t0, 2),
        }
```

`run_agent_fallback` wraps the existing `agent_inference` exactly the way `task6.run_agent` does, but passes the **already-loaded** `STATE["sam"]` as `call_sam_service` and the existing `build_qwen_sender(args)` — no reload of anything. Qwen loads lazily inside `local_qwen`'s `lru_cache` the first time the fallback fires, then stays warm too.

Main block:

```python
if __name__ == "__main__":
    args = build_args()          # reuse task5/task7 defaults: checkpoint,
    STATE["t0"] = time.monotonic()   # threshold, view, crop, resolution...
    startup(args)
    uvicorn.run(app, host="127.0.0.1", port=8765)
```

**Done when:**
- first `curl -s -X POST 'localhost:8765/segment?request=yellow%20box'` after startup answers in ~1–2 s with `"path": "direct"`, correct `num_kept`, and sane `object_depth`;
- a nonsense request (`"purple elephant"`) falls back (`"path": "agent"`) and still returns cleanly;
- a second request right after the first is just as fast (nothing reloaded).

---

## Task 4: client — one function, one CLI

**Do:** create `scripts/mask_client.py` so callers never think about HTTP:

```python
#!/usr/bin/env python3
import json, sys, urllib.parse, urllib.request

def segment(request: str, host: str = "127.0.0.1:8765") -> dict:
    q = urllib.parse.urlencode({"request": request})
    with urllib.request.urlopen(
            urllib.request.Request(f"http://{host}/segment?{q}", method="POST"),
            timeout=300) as r:
        return json.load(r)

if __name__ == "__main__":
    print(json.dumps(segment(" ".join(sys.argv[1:])), indent=2))
```

**Done when:** `.venv/bin/python scripts/mask_client.py yellow box` prints masks + depth JSON. This `segment()` function is exactly what the Fairino executive will import.

---

## Task 5: lifecycle — survives reboots and crashes

**Do:** for development, run it in tmux. For real use, a systemd unit at `/etc/systemd/system/mask-service.service`:

```ini
[Unit]
Description=SAM3 masking service (resident)
After=network.target

[Service]
User=team
WorkingDirectory=/home/team/VLA_Model_Work/SAM_3_implementation
ExecStart=/home/team/VLA_Model_Work/SAM_3_implementation/.venv/bin/python scripts/mask_service.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now mask-service
journalctl -u mask-service -f        # live logs
```

**Done when:** after a reboot, `curl localhost:8765/health` works with no manual steps; `kill -9` on the process and it's back within ~1 min + load time.

---

## Task 6 (polish, when needed)

- **Warm-up request at startup:** run one dummy `/segment` against a saved image so the first real request doesn't pay CUDA warm-up.
- **`compile=True, warm_up=True`** in the SAM build kwargs — now that the process is long-lived, one-time compile cost amortizes across every request. Time it A/B.
- **Metrics in every reply** (already sketched: `elapsed_s`, `path`) — log them; they're your regression alarm.
- **`/release_camera` + `/acquire_camera` endpoints** so you can run task5/task7 manually without stopping the whole service.

---

## Expectations & gotchas

- **Cold start:** unchanged (~1 min, paid once per boot). **Warm request:** ~1–2 s direct path; agent fallback stays slow (multiple Qwen generations) — that's fine, it should be rare.
- **GPU memory:** SAM + Qwen resident ≈ 10 GB, held permanently. Irrelevant on the 128 GB Thor.
- **Camera exclusivity:** while the service runs, nothing else can open the ZED. Stop the service (or use the Task 6 release endpoint) before running the standalone scripts.
- **Stale frames:** the ZED buffers; always grab a couple of throwaway frames per request (the `warmup_frames=2` above) so the mask describes *now*, not two minutes ago.
- **Crashed camera** (unplugged, etc.): catch grab errors and return HTTP 503 with a clear message; systemd's `Restart=on-failure` covers process death.
- **Security:** keep it on `127.0.0.1`. If it must be reachable from another machine later, that's the moment to think about auth, not before.
