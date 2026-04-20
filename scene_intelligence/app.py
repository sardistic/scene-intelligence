from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TextIO

import cv2

if __package__ in (None, ""):
    from scene_engine import SceneIntelligenceEngine, SceneSettings, scene_result_to_payload
else:
    from .scene_engine import SceneIntelligenceEngine, SceneSettings, scene_result_to_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone scene-intelligence app for webcams, files, and network video sources."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="OpenCV video source: webcam index like 0, file path, or stream URL.",
    )
    parser.add_argument(
        "--emit-interval",
        type=float,
        default=0.5,
        help="Seconds between JSON/state emissions.",
    )
    parser.add_argument(
        "--stdout-json",
        action="store_true",
        help="Emit newline-delimited JSON scene payloads to stdout.",
    )
    parser.add_argument(
        "--jsonl-path",
        help="Append newline-delimited JSON payloads to this file.",
    )
    parser.add_argument(
        "--state-path",
        help="Overwrite this file with the latest JSON payload on each emit.",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run headless without the OpenCV preview window.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a human-readable scene summary at each emit interval.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame limit for smoke runs against a video file.",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Requested capture width for webcam sources.",
    )
    parser.add_argument(
        "--height",
        type=int,
        help="Requested capture height for webcam sources.",
    )
    parser.add_argument(
        "--disable-object-detection",
        action="store_true",
        help="Disable the MediaPipe object detector pass.",
    )
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable persistent cross-session object memory.",
    )
    parser.add_argument(
        "--disable-overlay",
        action="store_true",
        help="Disable detailed model overlays while keeping the preview window.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def resolve_source(value: str) -> int | str:
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    if stripped.startswith("-") and stripped[1:].isdigit():
        return int(stripped)
    return stripped


def open_capture(source: int | str, *, width: int | None, height: int | None) -> cv2.VideoCapture:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    if isinstance(source, int):
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return cap


def open_jsonl_writer(path: str | None) -> TextIO | None:
    if not path:
        return None
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open("a", encoding="utf-8")


def emit_payload(
    payload: dict,
    *,
    stdout_json: bool,
    summary_output: bool,
    jsonl_writer: TextIO | None,
    state_path: str | None,
) -> None:
    line = json.dumps(payload, separators=(",", ":"))
    if stdout_json:
        print(line, flush=True)
    if summary_output:
        print(payload.get("summary", ""), flush=True)
    if jsonl_writer is not None:
        jsonl_writer.write(line + "\n")
        jsonl_writer.flush()
    if state_path:
        state_file = Path(state_path)
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    source = resolve_source(args.source)
    settings = SceneSettings(
        scene_object_detection_enabled=not args.disable_object_detection,
        scene_memory_enabled=not args.disable_memory,
        scene_model_enabled=not args.disable_overlay,
    )

    engine = None
    cap = None
    try:
        engine = SceneIntelligenceEngine(settings)
        cap = open_capture(source, width=args.width, height=args.height)
    except Exception as exc:
        logging.error("%s", exc)
        if engine is not None:
            engine.close()
        return 2

    jsonl_writer = open_jsonl_writer(args.jsonl_path)
    summary_output = args.print_summary or (args.no_preview and not args.stdout_json and not args.jsonl_path and not args.state_path)
    window_title = "Scene Intelligence Standalone"
    last_emit = 0.0
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logging.info("Video source ended or failed to provide a frame.")
                break

            frame_count += 1
            result = engine.process_frame(frame)

            if (result["timestamp_monotonic"] - last_emit) >= max(0.05, float(args.emit_interval)):
                payload = scene_result_to_payload(result, source=str(source))
                emit_payload(
                    payload,
                    stdout_json=args.stdout_json,
                    summary_output=summary_output,
                    jsonl_writer=jsonl_writer,
                    state_path=args.state_path,
                )
                last_emit = float(result["timestamp_monotonic"])

            if not args.no_preview:
                preview = engine.annotate_frame(frame, result)
                cv2.imshow(window_title, preview)
                pressed = cv2.waitKey(1) & 0xFF
                if pressed in (27, ord("q")):
                    break

            if args.max_frames and frame_count >= int(args.max_frames):
                break
    finally:
        if cap is not None:
            cap.release()
        if jsonl_writer is not None:
            jsonl_writer.close()
        if engine is not None:
            engine.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
