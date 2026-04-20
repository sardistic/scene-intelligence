# Scene Intelligence

Camera-based scene understanding for webcams, video files, and network streams.

`Scene Intelligence` watches a live or recorded video source and emits a portable scene-state model:

- focus position and confidence
- ambient and accent color suggestions
- scene energy
- face mood, glance, proximity, and speaking cues
- person / pose cues
- motion regions and tracked objects
- optional semantic object detections
- JSON output for automation, visualization, or downstream integrations

There is no device-control setup, no API key, and no lighting hardware required.

Download ZIP:

https://github.com/sardistic/scene-intelligence/archive/refs/heads/main.zip

## Requirements

- Python 3.10 or 3.11 recommended
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

## What You Get

Each emitted payload includes fields like:

- `summary`
- `focus_x`
- `focus_y`
- `focus_confidence`
- `focus_spread`
- `ambient_rgb`
- `accent_rgb`
- `scene_energy`
- `active_signals`
- `environment`
- `face`
- `pose`
- `motion`
- `tracked_objects`
- `detections`

## Notes

- The bundled EfficientDet model is already included in this repo, so object detection works without a first-run model download.
- If Face Mesh fails to initialize on a machine, the app will keep running with face-driven cues disabled instead of crashing.
- The dependency pin for `protobuf<5` is intentional because newer protobuf versions can break MediaPipe Face Mesh on some systems.

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
