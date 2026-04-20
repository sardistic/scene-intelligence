# Scene Intelligence

Camera-based scene understanding for webcams, video files, and network streams.

`Scene Intelligence` watches a live or recorded video source and emits a portable scene-state model:

- focus position and confidence
- ambient and accent color suggestions
- scene energy
- face mood, glance, proximity, and speaking cues
- person / pose cues
- motion regions and tracked objects
- semantic object detections with **long-term instance memory**
- JSON output for automation, visualization, or downstream integrations

There is no device-control setup, no API key, and no lighting hardware required.

Download ZIP:

https://github.com/sardistic/scene-intelligence/archive/refs/heads/main.zip

## Requirements

- Python 3.10+ (including 3.13)
- A webcam, video file, or stream URL
- Windows, macOS, or Linux

## Fastest Install

### Windows

1. Install Python 3.11 from https://www.python.org/downloads/ and enable `Add Python to PATH` during install.
2. Clone this repo or download the ZIP:
   https://github.com/sardistic/scene-intelligence/archive/refs/heads/main.zip
3. Double-click `install.bat`.
4. Double-click `run-webcam.bat`.

That is the quickest out-of-box path.

If the window closes too quickly after an error, use `run-webcam-debug.bat` instead. It keeps the console open and runs with verbose Python output.

If you prefer PowerShell:

```powershell
.\install.ps1
.\run-webcam.ps1
```

### macOS / Linux

```bash
git clone https://github.com/sardistic/scene-intelligence.git
cd scene-intelligence
bash install.sh
bash run-webcam.sh
```

## Standard Install

```bash
git clone https://github.com/sardistic/scene-intelligence.git
cd scene-intelligence
python -m venv .venv
```

### Windows

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\scene-intelligence.exe --source 0
```

### macOS / Linux

```bash
./.venv/bin/python -m pip install --upgrade pip wheel
./.venv/bin/python -m pip install .
./.venv/bin/scene-intelligence --source 0
```

## Simple Usage

Open the default webcam with preview:

```bash
scene-intelligence --source 0
```

Analyze a local video file:

```bash
scene-intelligence --source ./demo.mp4
```

Write the latest scene state to a JSON file:

```bash
scene-intelligence --source 0 --state-path ./scene_state.json
```

Append newline-delimited scene events:

```bash
scene-intelligence --source 0 --jsonl-path ./scene_events.jsonl
```

Print JSON to stdout without the preview window:

```bash
scene-intelligence --source 0 --stdout-json --no-preview
```

## Object Detection

The default detector is **YOLO11m** via [ultralytics](https://github.com/ultralytics/ultralytics) — 51.5 mAP@50-95 across 80 COCO classes. Model weights (~40 MB) are downloaded automatically on first run to `~/.cache/ultralytics/`.

> **Note:** `ultralytics` requires PyTorch, which is a large download (~2 GB) on a fresh machine. It is a one-time install. If ultralytics is unavailable the app falls back to **EfficientDet-Lite2** automatically.

## Long-Term Object Memory

The app keeps a persistent memory file (`.scene_memory.json` next to the package) that tracks every object it sees across sessions.

- Each label (cup, person, laptop, …) maintains up to 8 distinct **instances** identified by a colour-histogram appearance fingerprint.
- The overlay shows instance labels like `cup#1`, `cup#2` when multiple distinct instances of the same class have been seen.
- Familiar objects get lower detection thresholds and a confidence boost, so things you see regularly are detected more reliably over time.
- Delete `.scene_memory.json` to reset all memory.

## What You Get

Each emitted payload includes fields like:

| Field | Description |
|---|---|
| `summary` | Human-readable scene description |
| `focus_x` / `focus_y` | Normalised attention point (0–1) |
| `focus_confidence` | How confident the focus estimate is |
| `ambient_rgb` / `accent_rgb` | Scene colour suggestions |
| `scene_energy` | Activity level 14–100 |
| `active_signals` | Which cues are driving focus (face, pose, motion, …) |
| `environment` | Brightness, warmth, contrast, edge density |
| `face` | Mood, glance direction, proximity, speaking flag |
| `pose` | Body lean, arms raised, activity score |
| `motion` | Dominant motion region |
| `detections` | Object detections with `display_label`, `instance_id`, `confidence` |
| `tracked_objects` | Tracked motion blobs with classified labels |

## Notes

- EfficientDet-Lite2 is downloaded on first run (~7 MB). The bundled Lite0 model is kept as a fallback.
- Face Mesh, Pose, and Hands all fall back gracefully — the app keeps running with reduced cues if any model fails to init.
- On first run with a fresh install, three MediaPipe task models are also downloaded (~42 MB total). Subsequent runs are instant.

## Troubleshooting

If the webcam does not open:

- make sure another app is not already using the camera
- try a different source index like `--source 1`
- confirm your OS camera permissions allow Python / terminal access

If Windows says `python` or `py` is missing:

- reinstall Python from python.org
- enable `Add Python to PATH`

If you want the full CLI options:

```bash
scene-intelligence --help
```
