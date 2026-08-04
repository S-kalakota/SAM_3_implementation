# SAM 3.1 ZED Segmentation

## Quick start

From this directory, submit an instruction with one command:

```bash
./sam3 "orange and white box"
```

`./sam3` checks whether the resident segmentation service is ready. If it is not running, the command starts it, waits for SAM and the ZED camera to load, and then submits the instruction. Later requests reuse the warm models and are much faster.

If a compatible service was already started manually from the local `.venv`, `./sam3` detects and reuses it instead of trying to start a conflicting Docker container. New managed starts use Docker for consistent lifecycle handling.

Quotes are optional when the instruction contains only ordinary words:

```bash
./sam3 get the rightmost pasta box
```

The standard live ZED crop is `448,360,384,360` at HD720.

## Commands worth remembering

```bash
./sam3 "your instruction"  # Normal use; starts automatically
./sam3 status              # Check health
./sam3 logs                # Follow logs; Ctrl-C stops following only
./sam3 stop                # Release the ZED camera
```

Less common maintenance commands:

```bash
./sam3 restart             # Apply service/configuration changes
./sam3 rebuild             # Only after Dockerfile/dependency changes
./sam3 --help              # Complete command summary
```

After pulling or editing service code, run `./sam3 restart` once so the resident process loads the changes. The status response then reports the active crop and Qwen model.

Use the direct SAM path without the Qwen fallback when deliberately testing it:

```bash
./sam3 ask --no-agent-fallback orange box
```

## Why Docker is present

Docker is the service runtime, not the normal user interface. It provides:

- The NVIDIA GPU runtime and required library paths
- ZED camera device and SDK access
- A consistent Python/model environment
- A long-lived process so SAM and Qwen stay loaded
- Automatic restart after a crash or reboot

You should not normally run `docker build`, `docker compose up`, `docker compose logs`, or `docker compose stop` yourself. The `./sam3` command wraps those operations.

Avoid running the Docker service and a standalone camera script simultaneously because only one process can own the ZED camera. Run `./sam3 stop` before manual Task 5 or Task 7 camera diagnostics.

## Output

Each request prints its JSON result and saves the frame, overlay, mask metadata, depth results, and request log beneath:

```text
outputs/service/<timestamp>/
```

The detailed architecture and development roadmap are documented separately in `Daemon_plan.md`, `Second_plan.md`, and `Segmentation_Identification_Improvement_Plan.md`.
