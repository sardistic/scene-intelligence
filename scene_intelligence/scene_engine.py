from __future__ import annotations

import colorsys
import json
import logging
import math
import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Silence TensorFlow Lite / absl native noise before MediaPipe loads.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np


def _get_mp_solutions():
    """Return the mediapipe solutions module, trying multiple import paths."""
    try:
        return mp.solutions
    except AttributeError:
        pass
    try:
        import mediapipe.python.solutions as _sol
        return _sol
    except ImportError:
        pass
    return None


WHITE = (255, 255, 255)
CAMERA_IDLE_BASE = (6, 10, 24)
CAMERA_PALETTE_STOP_COUNT = 5
CAMERA_MOTION_MIN_AREA_RATIO = 0.004
CAMERA_MOTION_MAX_AREA_RATIO = 0.18
CAMERA_MOTION_DETECT_WIDTH = 320
CAMERA_MOTION_IDLE_TIMEOUT_SEC = 1.6
SCENE_OBJECT_MAX_COUNT = 4
SCENE_OBJECT_MAX_AGE_SEC = 1.6
SCENE_OBJECT_MATCH_PX = 120.0
SCENE_ENV_RESIZE = (160, 90)

FACE_LEFT_EYE_CORNERS = (33, 133)
FACE_RIGHT_EYE_CORNERS = (362, 263)
FACE_LEFT_EYE_LIDS = (159, 145)
FACE_RIGHT_EYE_LIDS = (386, 374)
FACE_LEFT_IRIS = (468, 469, 470, 471, 472)
FACE_RIGHT_IRIS = (473, 474, 475, 476, 477)
FACE_LEFT_BROW_INNER = 107
FACE_RIGHT_BROW_INNER = 336
FACE_LEFT_BROW_MID = (105, 66)
FACE_RIGHT_BROW_MID = (334, 293)
FACE_FOREHEAD = 10
FACE_CHIN = 152
FACE_NOSE_TIP = 1
FACE_MOUTH_TOP = 13
FACE_MOUTH_BOTTOM = 14
FACE_MOUTH_LEFT = 61
FACE_MOUTH_RIGHT = 291

SPEAKING_WINDOW_FRAMES = 16
SPEAKING_MEAN_THRESHOLD = 0.038
SPEAKING_VAR_THRESHOLD = 0.00035

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "efficientdet_lite0.tflite"
DEFAULT_MEMORY_PATH = APP_DIR / ".scene_memory.json"


@dataclass(slots=True)
class SceneSettings:
    camera_smoothing: float = 0.2
    scene_face_enabled: bool = True
    scene_glance_enabled: bool = True
    scene_mood_enabled: bool = True
    scene_person_enabled: bool = True
    scene_motion_enabled: bool = True
    scene_object_enabled: bool = True
    scene_model_enabled: bool = True
    scene_proximity_enabled: bool = True
    scene_speaking_enabled: bool = True
    scene_arms_enabled: bool = True
    scene_environment_enabled: bool = True
    scene_object_detection_enabled: bool = True
    scene_memory_enabled: bool = True
    model_path: Optional[str] = None
    memory_path: Optional[str] = None


def clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def lerp(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if abs(in_max - in_min) <= 1e-6:
        return out_max
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def quantize_percent(value: float, step: int = 1) -> int:
    if step <= 1:
        return max(1, min(100, int(round(value))))
    quantized = int(round(value / step) * step)
    return max(1, min(100, quantized))


def interpolate_rgb(
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(amount)))
    return tuple(
        int(round(a + (b - a) * t))
        for a, b in zip(color_a, color_b)
    )


def bgr_to_rgb(rgb_like: tuple[float, float, float] | np.ndarray) -> tuple[int, int, int]:
    b, g, r = rgb_like
    return int(round(r)), int(round(g)), int(round(b))


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def scale_rgb(
    rgb: tuple[int, int, int],
    factor: float,
    *,
    floor: int = 0,
) -> tuple[int, int, int]:
    bounded_factor = max(0.0, float(factor))
    return tuple(
        max(0 if channel == 0 else floor, min(255, int(round(channel * bounded_factor))))
        for channel in rgb
    )


def rgb_saturation_value(rgb: tuple[int, int, int]) -> tuple[float, float]:
    r, g, b = rgb
    _h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return float(s), float(v)


def is_neutral_rgb(rgb: tuple[int, int, int], saturation_threshold: float = 0.12) -> bool:
    saturation, _value = rgb_saturation_value(rgb)
    return saturation <= saturation_threshold


def sample_frame_palette(
    frame: np.ndarray,
    stop_count: int = CAMERA_PALETTE_STOP_COUNT,
) -> list[tuple[int, int, int]]:
    if frame.size == 0:
        return [WHITE for _ in range(max(1, stop_count))]

    stop_count = max(1, int(stop_count))
    height = frame.shape[0]
    y1 = int(round(height * 0.18))
    y2 = int(round(height * 0.82))
    crop = frame[y1:y2] if y2 > y1 else frame
    blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=9, sigmaY=9)
    reduced = cv2.resize(blurred, (stop_count, 1), interpolation=cv2.INTER_AREA)
    return [bgr_to_rgb(reduced[0, index]) for index in range(stop_count)]


def smooth_palette(
    previous: Optional[list[tuple[int, int, int]]],
    current: list[tuple[int, int, int]],
    amount: float,
) -> list[tuple[int, int, int]]:
    if not current:
        return []
    if not previous or len(previous) != len(current):
        return [tuple(color) for color in current]
    return [
        interpolate_rgb(previous[index], current[index], amount)
        for index in range(len(current))
    ]


class SceneBlobTracker:
    def __init__(
        self,
        *,
        max_distance_px: float = SCENE_OBJECT_MATCH_PX,
        max_age_sec: float = SCENE_OBJECT_MAX_AGE_SEC,
    ) -> None:
        self.max_distance_px = float(max_distance_px)
        self.max_age_sec = float(max_age_sec)
        self._next_id = 1
        self._tracks: dict[int, dict[str, Any]] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks = {}

    def update(self, detections: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
        active_tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if (now - float(track["last_seen"])) <= self.max_age_sec
        }
        self._tracks = active_tracks

        unmatched_ids = set(active_tracks.keys())
        unmatched_detection_indexes = set(range(len(detections)))
        candidate_pairs: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            cx = float(detection["center_x_px"])
            cy = float(detection["center_y_px"])
            for track_id, track in active_tracks.items():
                distance = math.hypot(cx - float(track["center_x_px"]), cy - float(track["center_y_px"]))
                candidate_pairs.append((distance, track_id, detection_index))

        for distance, track_id, detection_index in sorted(candidate_pairs, key=lambda item: item[0]):
            if distance > self.max_distance_px:
                continue
            if track_id not in unmatched_ids or detection_index not in unmatched_detection_indexes:
                continue
            detection = dict(detections[detection_index])
            detection["id"] = track_id
            detection["last_seen"] = now
            prev = active_tracks[track_id]
            dt = max(1e-4, now - float(prev.get("last_seen", now)))
            dx = float(detection["center_x_px"]) - float(prev.get("center_x_px", detection["center_x_px"]))
            dy = float(detection["center_y_px"]) - float(prev.get("center_y_px", detection["center_y_px"]))
            detection["velocity_px_s"] = math.hypot(dx, dy) / dt
            active_tracks[track_id] = detection
            unmatched_ids.remove(track_id)
            unmatched_detection_indexes.remove(detection_index)

        for detection_index in sorted(unmatched_detection_indexes):
            detection = dict(detections[detection_index])
            detection["id"] = self._next_id
            detection["last_seen"] = now
            detection.setdefault("velocity_px_s", 0.0)
            active_tracks[self._next_id] = detection
            self._next_id += 1

        self._tracks = active_tracks
        return sorted(
            self._tracks.values(),
            key=lambda track: (-float(track.get("area_ratio", 0.0)), int(track["id"])),
        )


class SpeakingDetector:
    def __init__(self, window: int = SPEAKING_WINDOW_FRAMES) -> None:
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, mouth_open: float) -> bool:
        self._buf.append(float(mouth_open))
        if len(self._buf) < 6:
            return False
        mean = sum(self._buf) / len(self._buf)
        variance = sum((x - mean) ** 2 for x in self._buf) / len(self._buf)
        return mean > SPEAKING_MEAN_THRESHOLD and variance > SPEAKING_VAR_THRESHOLD

    def reset(self) -> None:
        self._buf.clear()


class SceneMemory:
    SAVE_INTERVAL_SEC = 30.0
    FAMILIARITY_SATURATE = 80
    MAX_BOOST = 0.35

    def __init__(self, memory_file: Path) -> None:
        self.memory_file = memory_file
        self._mem: dict[str, dict[str, Any]] = {}
        self._last_save: float = 0.0
        self._dirty = False
        self._load()

    def _load(self) -> None:
        try:
            if self.memory_file.exists():
                data = json.loads(self.memory_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._mem = data
        except Exception:
            self._mem = {}

    def save(self, force: bool = False) -> None:
        if not self._dirty:
            return
        t = time.monotonic()
        if not force and (t - self._last_save) < self.SAVE_INTERVAL_SEC:
            return
        try:
            self.memory_file.parent.mkdir(parents=True, exist_ok=True)
            self.memory_file.write_text(json.dumps(self._mem, indent=2), encoding="utf-8")
            self._last_save = t
            self._dirty = False
        except Exception:
            pass

    def record(self, label: str, confidence: float, center_x: float) -> None:
        now_ts = time.time()
        if label not in self._mem:
            self._mem[label] = {
                "sightings": 0,
                "first_seen": now_ts,
                "last_seen": now_ts,
                "avg_confidence": float(confidence),
                "typical_x": float(center_x),
            }
        entry = self._mem[label]
        entry["sightings"] += 1
        entry["last_seen"] = now_ts
        alpha = 0.06
        entry["avg_confidence"] = (1 - alpha) * entry["avg_confidence"] + alpha * confidence
        entry["typical_x"] = (1 - alpha) * entry["typical_x"] + alpha * center_x
        self._dirty = True

    def familiarity(self, label: str) -> float:
        entry = self._mem.get(label)
        if entry is None:
            return 0.0
        return min(1.0, entry["sightings"] / self.FAMILIARITY_SATURATE)

    def boost_confidence(self, label: str, raw: float) -> float:
        return min(1.0, raw * (1.0 + self.familiarity(label) * self.MAX_BOOST))

    def effective_threshold(self, label: str, base_threshold: float) -> float:
        return max(0.18, base_threshold - self.familiarity(label) * 0.12)

    def recent_labels(self, max_age_sec: float = 300.0) -> list[tuple[str, int]]:
        now_ts = time.time()
        return sorted(
            [
                (label, int(entry["sightings"]))
                for label, entry in self._mem.items()
                if (now_ts - entry["last_seen"]) <= max_age_sec
            ],
            key=lambda item: self._mem[item[0]]["last_seen"],
            reverse=True,
        )


class DetectionBuffer:
    def __init__(self, window: int = 10, stability_ratio: float = 0.30) -> None:
        self._history: deque[list[dict[str, Any]]] = deque(maxlen=window)
        self._stability_ratio = stability_ratio
        self._stable: list[dict[str, Any]] = []

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._history.append(list(detections))
        if len(self._history) < 3:
            self._stable = list(detections)
            return self._stable

        label_frame_count: dict[str, int] = {}
        label_best: dict[str, dict[str, Any]] = {}
        for frame_dets in self._history:
            seen_this_frame: set[str] = set()
            for det in frame_dets:
                label = det["label"]
                if label not in seen_this_frame:
                    label_frame_count[label] = label_frame_count.get(label, 0) + 1
                    seen_this_frame.add(label)
                if label not in label_best or det["confidence"] > label_best[label]["confidence"]:
                    label_best[label] = det

        threshold = len(self._history) * self._stability_ratio
        self._stable = [
            det for label, det in label_best.items()
            if label_frame_count.get(label, 0) >= threshold
        ]
        return self._stable

    def reset(self) -> None:
        self._history.clear()
        self._stable = []


def _detection_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / max(1, union)


def _nms_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.40,
) -> list[dict[str, Any]]:
    sorted_dets = sorted(detections, key=lambda det: det["confidence"], reverse=True)
    kept: list[dict[str, Any]] = []
    for det in sorted_dets:
        if all(_detection_iou(det, existing) < iou_threshold for existing in kept):
            kept.append(det)
    return kept


class SceneObjectDetector:
    _MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
    )
    _MIN_CROP_DIM = 48
    _HAND_CROP_PAD = 0.55
    _HAND_CROP_TARGET = 320

    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        max_results: int = 10,
        score_threshold: float = 0.30,
        detect_every: int = 5,
    ) -> None:
        self._max_results = max_results
        self._base_threshold = score_threshold
        self._detect_every = detect_every
        self._detector = None
        self._available = False
        self._init_error: Optional[str] = None
        self._cached: list[dict[str, Any]] = []
        self._frame_counter = 0
        self._model_candidates = [model_path] if model_path else [DEFAULT_MODEL_PATH]

    def _resolve_model_path(self) -> Path:
        for candidate in self._model_candidates:
            if candidate is not None and candidate.exists():
                return candidate
        primary = next(candidate for candidate in self._model_candidates if candidate is not None)
        primary.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Downloading EfficientDet-Lite0 model (~5 MB) to %s", primary)
        urllib.request.urlretrieve(self._MODEL_URL, str(primary))
        return primary

    def _ensure_ready(self) -> bool:
        if self._detector is not None:
            return True
        if self._init_error is not None:
            return False
        try:
            model_path = self._resolve_model_path()
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.core import base_options as mp_base

            model_bytes = model_path.read_bytes()
            options = mp_vision.ObjectDetectorOptions(
                base_options=mp_base.BaseOptions(model_asset_buffer=model_bytes),
                max_results=self._max_results,
                score_threshold=self._base_threshold,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self._detector = mp_vision.ObjectDetector.create_from_options(options)
            self._available = True
            logging.info("SceneObjectDetector ready")
        except Exception as exc:
            self._init_error = str(exc)
            logging.warning("SceneObjectDetector unavailable: %s", exc)
        return self._available

    def _run_on_bgr(
        self,
        bgr: np.ndarray,
        full_w: int,
        full_h: int,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
        scale: float = 1.0,
        held: bool = False,
    ) -> list[dict[str, Any]]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)
        out: list[dict[str, Any]] = []
        for det in result.detections or []:
            category = det.categories[0] if det.categories else None
            if category is None:
                continue
            bbox = det.bounding_box
            fx = offset_x + int(bbox.origin_x / scale)
            fy = offset_y + int(bbox.origin_y / scale)
            fw = max(1, int(bbox.width / scale))
            fh = max(1, int(bbox.height / scale))
            fx = max(0, min(full_w - 1, fx))
            fy = max(0, min(full_h - 1, fy))
            fw = min(full_w - fx, fw)
            fh = min(full_h - fy, fh)
            cx = clip_unit((fx + fw / 2.0) / max(1, full_w))
            cy = clip_unit((fy + fh / 2.0) / max(1, full_h))
            out.append({
                "label": category.category_name or "object",
                "confidence": float(category.score),
                "bbox": (fx, fy, fw, fh),
                "center_x": cx,
                "center_y": cy,
                "center_x_px": fx + fw // 2,
                "center_y_px": fy + fh // 2,
                "held": held,
            })
        return out

    def update(
        self,
        frame_bgr: np.ndarray,
        hand_bboxes: Optional[list[tuple[int, int, int, int]]] = None,
        memory: Optional[SceneMemory] = None,
    ) -> list[dict[str, Any]]:
        self._frame_counter += 1
        if self._frame_counter % self._detect_every != 0:
            return self._cached
        if not self._ensure_ready():
            return self._cached
        try:
            height, width = frame_bgr.shape[:2]
            all_dets: list[dict[str, Any]] = []
            all_dets.extend(self._run_on_bgr(frame_bgr, width, height))

            if hand_bboxes:
                for hx, hy, hw, hh in hand_bboxes:
                    pad_x = int(hw * self._HAND_CROP_PAD)
                    pad_y = int(hh * self._HAND_CROP_PAD)
                    rx = max(0, hx - pad_x)
                    ry = max(0, hy - pad_y)
                    rx2 = min(width, hx + hw + pad_x)
                    ry2 = min(height, hy + hh + pad_y)
                    crop_w, crop_h = rx2 - rx, ry2 - ry
                    if crop_w < self._MIN_CROP_DIM or crop_h < self._MIN_CROP_DIM:
                        continue
                    crop = frame_bgr[ry:ry2, rx:rx2]
                    longest = max(crop_w, crop_h)
                    scale = self._HAND_CROP_TARGET / max(1, longest)
                    if scale > 1.0:
                        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    else:
                        scale = 1.0
                    all_dets.extend(
                        self._run_on_bgr(
                            crop,
                            width,
                            height,
                            offset_x=rx,
                            offset_y=ry,
                            scale=scale,
                            held=True,
                        )
                    )

            if memory is not None:
                boosted: list[dict[str, Any]] = []
                for det in all_dets:
                    label = det["label"]
                    effective_threshold = memory.effective_threshold(label, self._base_threshold)
                    boosted_confidence = memory.boost_confidence(label, det["confidence"])
                    if boosted_confidence >= effective_threshold:
                        det = dict(det)
                        det["confidence"] = boosted_confidence
                        boosted.append(det)
                all_dets = boosted

            self._cached = _nms_detections(all_dets)
        except Exception as exc:
            logging.debug("SceneObjectDetector.update error: %s", exc)
        return self._cached

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None


def landmark_xy(landmarks, index: int) -> tuple[float, float]:
    landmark = landmarks.landmark[index]
    return float(landmark.x), float(landmark.y)


def average_landmark_xy(landmarks, indices: tuple[int, ...] | list[int]) -> tuple[float, float]:
    xs = [float(landmarks.landmark[index].x) for index in indices]
    ys = [float(landmarks.landmark[index].y) for index in indices]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def normalized_distance(landmarks, a: int, b: int) -> float:
    ax, ay = landmark_xy(landmarks, a)
    bx, by = landmark_xy(landmarks, b)
    return float(math.hypot(ax - bx, ay - by))


def face_bbox_from_landmarks(face_landmarks, width: int, height: int) -> tuple[int, int, int, int]:
    xs = [float(point.x) for point in face_landmarks.landmark]
    ys = [float(point.y) for point in face_landmarks.landmark]
    min_x = int(round(max(0.0, min(xs)) * width))
    max_x = int(round(min(1.0, max(xs)) * width))
    min_y = int(round(max(0.0, min(ys)) * height))
    max_y = int(round(min(1.0, max(ys)) * height))
    return min_x, min_y, max(1, max_x - min_x), max(1, max_y - min_y)


def eye_gaze_ratio(iris_x: float, corner_a_x: float, corner_b_x: float) -> float:
    min_x = min(corner_a_x, corner_b_x)
    max_x = max(corner_a_x, corner_b_x)
    span = max(1e-6, max_x - min_x)
    return clip_unit((iris_x - min_x) / span)


def extract_face_scene_state(
    face_landmarks,
    width: int,
    height: int,
    *,
    speaking: bool = False,
) -> dict[str, Any]:
    bbox = face_bbox_from_landmarks(face_landmarks, width, height)
    left_eye_outer = landmark_xy(face_landmarks, FACE_LEFT_EYE_CORNERS[0])
    left_eye_inner = landmark_xy(face_landmarks, FACE_LEFT_EYE_CORNERS[1])
    right_eye_outer = landmark_xy(face_landmarks, FACE_RIGHT_EYE_CORNERS[0])
    right_eye_inner = landmark_xy(face_landmarks, FACE_RIGHT_EYE_CORNERS[1])
    left_iris = average_landmark_xy(face_landmarks, list(FACE_LEFT_IRIS))
    right_iris = average_landmark_xy(face_landmarks, list(FACE_RIGHT_IRIS))
    nose_tip = landmark_xy(face_landmarks, FACE_NOSE_TIP)
    forehead = landmark_xy(face_landmarks, FACE_FOREHEAD)
    chin = landmark_xy(face_landmarks, FACE_CHIN)

    left_gaze_ratio = eye_gaze_ratio(left_iris[0], left_eye_outer[0], left_eye_inner[0])
    right_gaze_ratio = eye_gaze_ratio(right_iris[0], right_eye_inner[0], right_eye_outer[0])
    glance_x = (((left_gaze_ratio - 0.5) * 2.0) + ((right_gaze_ratio - 0.5) * 2.0)) / 2.0

    left_eye_center = average_landmark_xy(
        face_landmarks,
        [FACE_LEFT_EYE_CORNERS[0], FACE_LEFT_EYE_CORNERS[1], FACE_LEFT_EYE_LIDS[0], FACE_LEFT_EYE_LIDS[1]],
    )
    right_eye_center = average_landmark_xy(
        face_landmarks,
        [FACE_RIGHT_EYE_CORNERS[0], FACE_RIGHT_EYE_CORNERS[1], FACE_RIGHT_EYE_LIDS[0], FACE_RIGHT_EYE_LIDS[1]],
    )
    eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2.0
    eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2.0

    face_width = max(1e-6, normalized_distance(face_landmarks, 33, 263))
    face_height = max(1e-6, normalized_distance(face_landmarks, FACE_FOREHEAD, FACE_CHIN))
    smile_ratio = normalized_distance(face_landmarks, FACE_MOUTH_LEFT, FACE_MOUTH_RIGHT) / face_width
    mouth_open = normalized_distance(face_landmarks, FACE_MOUTH_TOP, FACE_MOUTH_BOTTOM) / face_height
    eye_open = (
        normalized_distance(face_landmarks, FACE_LEFT_EYE_LIDS[0], FACE_LEFT_EYE_LIDS[1])
        + normalized_distance(face_landmarks, FACE_RIGHT_EYE_LIDS[0], FACE_RIGHT_EYE_LIDS[1])
    ) / (2.0 * face_width)
    head_yaw = max(-1.0, min(1.0, ((nose_tip[0] - eye_center_x) / face_width) * 1.6))

    nose_above_eyes = eye_center_y - nose_tip[1]
    head_pitch = max(-1.0, min(1.0, nose_above_eyes / max(1e-6, face_height) * 3.5))

    left_brow_mid = average_landmark_xy(face_landmarks, list(FACE_LEFT_BROW_MID))
    right_brow_mid = average_landmark_xy(face_landmarks, list(FACE_RIGHT_BROW_MID))
    brow_height = max(0.0, (
        (left_eye_center[1] - left_brow_mid[1]) + (right_eye_center[1] - right_brow_mid[1])
    ) / (2.0 * max(1e-6, face_height)))

    inner_brow_spread = abs(
        landmark_xy(face_landmarks, FACE_LEFT_BROW_INNER)[0]
        - landmark_xy(face_landmarks, FACE_RIGHT_BROW_INNER)[0]
    ) / max(1e-6, face_width)

    face_proximity = clip_unit(face_width / 0.55)

    mood_label = "focused"
    mood_rgb = (100, 245, 220)
    mood_energy_delta = 0

    eye_very_wide = eye_open > 0.09
    mouth_very_open = mouth_open > 0.13
    brow_raised = brow_height > 0.26
    brow_furrowed = inner_brow_spread < 0.28
    strongly_smiling = smile_ratio >= 0.50
    mouth_notably_open = mouth_open >= 0.10
    eyes_narrow = eye_open <= 0.048
    eyes_nearly_closed = eye_open <= 0.032

    if eye_very_wide and mouth_very_open and brow_raised:
        mood_label = "surprised"
        mood_rgb = (255, 240, 80)
        mood_energy_delta = 18
    elif eyes_nearly_closed and mouth_open <= 0.04 and not strongly_smiling:
        mood_label = "sleepy"
        mood_rgb = (60, 80, 200)
        mood_energy_delta = -18
    elif brow_furrowed and not strongly_smiling and eye_open < 0.08 and not speaking:
        mood_label = "tense"
        mood_rgb = (220, 80, 80)
        mood_energy_delta = 6
    elif strongly_smiling or (mouth_notably_open and strongly_smiling):
        mood_label = "energized"
        mood_rgb = (255, 175, 74)
        mood_energy_delta = 12
    elif speaking:
        mood_label = "engaged"
        mood_rgb = (120, 255, 160)
        mood_energy_delta = 8
    elif eyes_narrow and mouth_open <= 0.05:
        mood_label = "calm"
        mood_rgb = (88, 170, 255)
        mood_energy_delta = -10
    elif abs(glance_x) >= 0.4:
        mood_label = "curious"
        mood_rgb = (182, 130, 255)
        mood_energy_delta = 4

    face_center_x = clip_unit((bbox[0] + (bbox[2] / 2.0)) / float(max(1, width)))
    face_center_y = clip_unit((bbox[1] + (bbox[3] / 2.0)) / float(max(1, height)))
    attention_x = clip_unit(face_center_x + (glance_x * 0.18) + (head_yaw * 0.12))

    return {
        "bbox": bbox,
        "center_x": face_center_x,
        "center_y": face_center_y,
        "attention_x": attention_x,
        "glance_x": max(-1.0, min(1.0, glance_x)),
        "head_yaw": head_yaw,
        "head_pitch": head_pitch,
        "smile_ratio": smile_ratio,
        "mouth_open": mouth_open,
        "eye_open": eye_open,
        "brow_height": brow_height,
        "brow_furrow": inner_brow_spread,
        "face_proximity": face_proximity,
        "speaking": speaking,
        "mood_label": mood_label,
        "mood_rgb": mood_rgb,
        "mood_energy_delta": mood_energy_delta,
    }


def extract_pose_scene_state(pose_landmarks) -> Optional[dict[str, Any]]:
    if pose_landmarks is None:
        return None

    visibility_floor = 0.25
    body_indices = [11, 12, 23, 24]
    visible_points = []
    for index in body_indices:
        landmark = pose_landmarks.landmark[index]
        if getattr(landmark, "visibility", 0.0) >= visibility_floor:
            visible_points.append((float(landmark.x), float(landmark.y)))

    if not visible_points:
        return None

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]

    shoulder_ys: list[float] = []
    for index in (11, 12):
        landmark = pose_landmarks.landmark[index]
        if getattr(landmark, "visibility", 0.0) >= visibility_floor:
            shoulder_ys.append(float(landmark.y))
    if not shoulder_ys:
        for index in (13, 14):
            landmark = pose_landmarks.landmark[index]
            if getattr(landmark, "visibility", 0.0) >= visibility_floor:
                shoulder_ys.append(float(landmark.y) + 0.10)
    shoulder_y = sum(shoulder_ys) / len(shoulder_ys) if shoulder_ys else None

    body_lean = 0.0
    if len(xs) >= 2:
        body_lean = max(-1.0, min(1.0, (sum(xs) / len(xs) - 0.5) * 2.4))

    arms_raised = False
    if shoulder_y is not None:
        wrists_above = 0
        for index in (15, 16):
            wrist = pose_landmarks.landmark[index]
            if getattr(wrist, "visibility", 0.0) >= visibility_floor and float(wrist.y) < shoulder_y - 0.05:
                wrists_above += 1
        if wrists_above == 0:
            for index in (13, 14):
                elbow = pose_landmarks.landmark[index]
                if getattr(elbow, "visibility", 0.0) >= visibility_floor and float(elbow.y) < shoulder_y - 0.04:
                    wrists_above += 1
        arms_raised = wrists_above >= 1

    upper_indices = [11, 12, 13, 14, 15, 16, 23, 24]
    visible_count = sum(
        1
        for index in upper_indices
        if getattr(pose_landmarks.landmark[index], "visibility", 0.0) >= visibility_floor
    )
    activity_score = clip_unit(float(visible_count) / float(len(upper_indices)))

    return {
        "center_x": clip_unit(sum(xs) / len(xs)),
        "center_y": clip_unit(sum(ys) / len(ys)),
        "bbox_norm": (
            clip_unit(min(xs)),
            clip_unit(min(ys)),
            clip_unit(max(xs) - min(xs)),
            clip_unit(max(ys) - min(ys)),
        ),
        "body_lean": body_lean,
        "arms_raised": arms_raised,
        "activity_score": activity_score,
    }


def detect_motion_regions(
    frame: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    *,
    max_regions: int = SCENE_OBJECT_MAX_COUNT,
) -> list[dict[str, Any]]:
    if frame.size == 0:
        return []

    height, width = frame.shape[:2]
    detect_width = min(width, CAMERA_MOTION_DETECT_WIDTH)
    detect_height = max(1, int(round(height * (detect_width / float(max(1, width))))))
    reduced = cv2.resize(frame, (detect_width, detect_height), interpolation=cv2.INTER_AREA)
    fg_mask = subtractor.apply(reduced, learningRate=0.03)
    fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
    _, thresh = cv2.threshold(fg_mask, 210, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    x_scale = width / float(max(1, detect_width))
    y_scale = height / float(max(1, detect_height))
    detections: list[dict[str, Any]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        area_ratio = area / float(max(1, detect_width * detect_height))
        if area_ratio < CAMERA_MOTION_MIN_AREA_RATIO or area_ratio > 0.72:
            continue

        x, y, w_box, h_box = cv2.boundingRect(contour)
        full_x = int(round(x * x_scale))
        full_y = int(round(y * y_scale))
        full_w = max(1, int(round(w_box * x_scale)))
        full_h = max(1, int(round(h_box * y_scale)))
        x2 = min(width, full_x + full_w)
        y2 = min(height, full_y + full_h)
        roi = frame[full_y:y2, full_x:x2]
        roi_rgb = bgr_to_rgb(cv2.mean(roi)[:3]) if roi.size else WHITE

        moments = cv2.moments(contour)
        if moments["m00"] > 1e-6:
            center_x = float(moments["m10"] / moments["m00"]) / float(detect_width)
            center_y = float(moments["m01"] / moments["m00"]) / float(detect_height)
        else:
            center_x = (x + (w_box / 2.0)) / float(detect_width)
            center_y = (y + (h_box / 2.0)) / float(detect_height)

        detections.append({
            "bbox": (full_x, full_y, max(1, x2 - full_x), max(1, y2 - full_y)),
            "center_x": clip_unit(center_x),
            "center_y": clip_unit(center_y),
            "center_x_px": int(round(clip_unit(center_x) * max(1, width - 1))),
            "center_y_px": int(round(clip_unit(center_y) * max(1, height - 1))),
            "area_ratio": area_ratio,
            "rgb": roi_rgb,
        })
        if len(detections) >= max_regions:
            break

    return detections


def extract_scene_environment(frame: np.ndarray) -> dict[str, Any]:
    if frame.size == 0:
        return {"frame_brightness": 0.5, "warmth": 0.5, "edge_density": 0.0, "contrast": 0.5}

    small = cv2.resize(frame, SCENE_ENV_RESIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    p5 = float(np.percentile(gray, 5))
    p50 = float(np.percentile(gray, 50))
    ambient_raw = (p5 * 0.70 + p50 * 0.30) / 255.0
    frame_brightness = min(1.0, ambient_raw * 1.6)

    p10 = float(np.percentile(gray, 10))
    p90 = float(np.percentile(gray, 90))
    contrast = min(1.0, (p90 - p10) / 180.0)

    dark_mask = gray < int(p50)
    b_channel = small[:, :, 0].astype(np.float32)
    r_channel = small[:, :, 2].astype(np.float32)
    dark_r = float(np.mean(r_channel[dark_mask])) if np.any(dark_mask) else float(np.mean(r_channel))
    dark_b = float(np.mean(b_channel[dark_mask])) if np.any(dark_mask) else float(np.mean(b_channel))
    denom = dark_r + dark_b
    warmth = (dark_r / max(1.0, denom)) if denom > 1.0 else 0.5

    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))

    return {
        "frame_brightness": frame_brightness,
        "warmth": warmth,
        "edge_density": edge_density,
        "contrast": contrast,
    }


def _bbox_overlap_ratio(
    ax: int, ay: int, aw: int, ah: int,
    bx: int, by: int, bw: int, bh: int,
) -> float:
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, aw * ah)
    return inter / area_a


def classify_motion_blob(
    blob: dict[str, Any],
    *,
    real_detections: Optional[list[dict[str, Any]]],
    hand_bboxes: Optional[list[tuple[int, int, int, int]]],
    face_bbox: Optional[tuple[int, int, int, int]],
    pose_bbox_norm: Optional[tuple[float, float, float, float]],
    frame_width: int,
    frame_height: int,
) -> tuple[str, float]:
    bx, by, bw, bh = blob["bbox"]

    if real_detections:
        best_label = ""
        best_conf = 0.0
        best_overlap = 0.0
        for det in real_detections:
            dx, dy, dw, dh = det["bbox"]
            overlap = _bbox_overlap_ratio(bx, by, bw, bh, dx, dy, dw, dh)
            if overlap > 0.20 and overlap > best_overlap:
                best_overlap = overlap
                best_label = det["label"]
                best_conf = det["confidence"]
        if best_label:
            return best_label, best_conf

    if hand_bboxes:
        for hx, hy, hw, hh in hand_bboxes:
            if _bbox_overlap_ratio(bx, by, bw, bh, hx, hy, hw, hh) > 0.25:
                return "hand", 0.90

    if face_bbox is not None:
        fx, fy, fw, fh = face_bbox
        if _bbox_overlap_ratio(bx, by, bw, bh, fx, fy, fw, fh) > 0.30:
            return "face", 0.85

    if pose_bbox_norm is not None:
        px = int(round(pose_bbox_norm[0] * frame_width))
        py = int(round(pose_bbox_norm[1] * frame_height))
        pw = max(1, int(round(pose_bbox_norm[2] * frame_width)))
        ph = max(1, int(round(pose_bbox_norm[3] * frame_height)))
        if _bbox_overlap_ratio(bx, by, bw, bh, px, py, pw, ph) > 0.20:
            return "person", 0.75

    aspect = bh / max(1, bw)
    area_ratio = float(blob.get("area_ratio", 0.0))
    if aspect > 1.8 and area_ratio > 0.04:
        return "person", 0.40
    if aspect < 0.5 and area_ratio > 0.06:
        return "surface", 0.40
    if area_ratio < 0.012:
        return "small object", 0.30
    return "object", 0.30


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(inner) for inner in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def scene_result_to_payload(result: dict[str, Any], *, source: Optional[str] = None) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "source": source,
        **result,
    }
    return _json_ready(payload)


class _LandmarkListAdapter:
    """Wraps a Tasks-API landmark list so it exposes the .landmark[i] interface."""
    __slots__ = ("_lms",)

    def __init__(self, landmarks):
        self._lms = landmarks

    @property
    def landmark(self):
        return self._lms


def _adapt_face_result(task_result):
    class _R:
        pass
    r = _R()
    r.multi_face_landmarks = (
        [_LandmarkListAdapter(lms) for lms in task_result.face_landmarks]
        if task_result.face_landmarks else None
    )
    return r


def _adapt_pose_result(task_result):
    class _R:
        pass
    r = _R()
    r.pose_landmarks = (
        _LandmarkListAdapter(task_result.pose_landmarks[0])
        if task_result.pose_landmarks else None
    )
    return r


def _adapt_hand_result(task_result):
    class _R:
        pass
    r = _R()
    r.multi_hand_landmarks = (
        [_LandmarkListAdapter(lms) for lms in task_result.hand_landmarks]
        if task_result.hand_landmarks else None
    )
    return r


class _FaceMeshWrapper:
    _TASK_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    _TASK_PATH = APP_DIR / "face_landmarker.task"

    def __init__(self, *, max_num_faces=1, refine_landmarks=True,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._sol = None
        self._landmarker = None
        self._num_faces = max_num_faces
        self._ts_ms = 0

        solutions = _get_mp_solutions()
        if solutions is not None:
            try:
                self._sol = solutions.face_mesh.FaceMesh(
                    max_num_faces=max_num_faces,
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                return
            except Exception:
                pass

        self._init_tasks()

    def _init_tasks(self):
        import time as _time
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base
        path = self._TASK_PATH
        if not path.exists():
            logging.info("Downloading %s (~30 MB)...", path.name)
            urllib.request.urlretrieve(self._TASK_URL, str(path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_buffer=path.read_bytes()),
            num_faces=self._num_faces,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._ts_ms = int(_time.monotonic() * 1000)
        logging.info("Face Mesh ready (Tasks API)")

    def process(self, rgb):
        if self._sol is not None:
            return self._sol.process(rgb)
        import time as _time
        ts = max(self._ts_ms + 1, int(_time.monotonic() * 1000))
        self._ts_ms = ts
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return _adapt_face_result(self._landmarker.detect_for_video(mp_img, ts))

    def close(self):
        for obj in (self._sol, self._landmarker):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass


class _PoseWrapper:
    _TASK_URL = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    )
    _TASK_PATH = APP_DIR / "pose_landmarker_lite.task"

    def __init__(self, *, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._sol = None
        self._landmarker = None
        self._ts_ms = 0

        solutions = _get_mp_solutions()
        if solutions is not None:
            try:
                self._sol = solutions.pose.Pose(
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                return
            except Exception:
                pass

        self._init_tasks()

    def _init_tasks(self):
        import time as _time
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base
        path = self._TASK_PATH
        if not path.exists():
            logging.info("Downloading %s (~3 MB)...", path.name)
            urllib.request.urlretrieve(self._TASK_URL, str(path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_buffer=path.read_bytes()),
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._ts_ms = int(_time.monotonic() * 1000)
        logging.info("Pose ready (Tasks API)")

    def process(self, rgb):
        if self._sol is not None:
            return self._sol.process(rgb)
        import time as _time
        ts = max(self._ts_ms + 1, int(_time.monotonic() * 1000))
        self._ts_ms = ts
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return _adapt_pose_result(self._landmarker.detect_for_video(mp_img, ts))

    def close(self):
        for obj in (self._sol, self._landmarker):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass


class _HandsWrapper:
    _TASK_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    _TASK_PATH = APP_DIR / "hand_landmarker.task"

    def __init__(self, *, min_detection_confidence=0.6, min_tracking_confidence=0.5,
                 max_num_hands=2):
        self._sol = None
        self._landmarker = None
        self._max_hands = max_num_hands
        self._ts_ms = 0

        solutions = _get_mp_solutions()
        if solutions is not None:
            try:
                self._sol = solutions.hands.Hands(
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    max_num_hands=max_num_hands,
                )
                return
            except Exception:
                pass

        self._init_tasks()

    def _init_tasks(self):
        import time as _time
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base
        path = self._TASK_PATH
        if not path.exists():
            logging.info("Downloading %s (~9 MB)...", path.name)
            urllib.request.urlretrieve(self._TASK_URL, str(path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_buffer=path.read_bytes()),
            num_hands=self._max_hands,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._ts_ms = int(_time.monotonic() * 1000)
        logging.info("Hands ready (Tasks API)")

    def process(self, rgb):
        if self._sol is not None:
            return self._sol.process(rgb)
        import time as _time
        ts = max(self._ts_ms + 1, int(_time.monotonic() * 1000))
        self._ts_ms = ts
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return _adapt_hand_result(self._landmarker.detect_for_video(mp_img, ts))

    def close(self):
        for obj in (self._sol, self._landmarker):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass


def _format_solution_init_error(solution_name: str, exc: Exception) -> str:
    protobuf_version = "unknown"
    try:
        from google.protobuf import __version__ as protobuf_version
    except Exception:
        pass

    raw_error = " ".join(str(exc).split())
    compact_error = raw_error[:220] + ("..." if len(raw_error) > 220 else "")
    message = f"{solution_name} unavailable and will be disabled: {compact_error}"
    if solution_name == "Face Mesh":
        major = protobuf_version.split(".", 1)[0]
        if major.isdigit() and int(major) >= 5:
            message += (
                f" mediapipe {mp.__version__} with protobuf {protobuf_version} can fail here;"
                ' install "protobuf<5" for full face-driven cues.'
            )
    return message


class SceneIntelligenceEngine:
    def __init__(self, settings: Optional[SceneSettings] = None) -> None:
        self.settings = settings or SceneSettings()
        self._runtime_warnings: list[str] = []
        self._face_mesh = self._safe_create_solution(
            "Face Mesh",
            lambda: _FaceMeshWrapper(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ),
        )
        self._pose = self._safe_create_solution(
            "Pose",
            lambda: _PoseWrapper(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ),
        )
        self._hands = self._safe_create_solution(
            "Hands",
            lambda: _HandsWrapper(
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
                max_num_hands=2,
            ),
        )
        self._motion_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=180,
            varThreshold=32,
            detectShadows=False,
        )
        self._scene_object_tracker = SceneBlobTracker()
        self._speaking_detector = SpeakingDetector()
        model_path = Path(self.settings.model_path) if self.settings.model_path else None
        memory_path = Path(self.settings.memory_path) if self.settings.memory_path else DEFAULT_MEMORY_PATH
        self._scene_object_detector = SceneObjectDetector(model_path=model_path)
        self._scene_memory = SceneMemory(memory_path)
        self._detection_buffer = DetectionBuffer(window=10, stability_ratio=0.30)
        self._camera_palette_colors: Optional[list[tuple[int, int, int]]] = None
        self._scene_smoothed_focus_x: Optional[float] = None
        self._scene_smoothed_focus_y: Optional[float] = None
        self._last_motion_seen_time = 0.0
        self._scene_env_smoothed = {
            "frame_brightness": 0.5,
            "warmth": 0.5,
            "edge_density": 0.0,
            "contrast": 0.5,
        }

    def _safe_create_solution(self, solution_name: str, factory):
        try:
            return factory()
        except Exception as exc:
            message = _format_solution_init_error(solution_name, exc)
            self._runtime_warnings.append(message)
            logging.warning(message)
            return None

    def close(self) -> None:
        self._scene_memory.save(force=True)
        self._scene_object_detector.close()
        if self._hands is not None:
            self._hands.close()
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._pose is not None:
            self._pose.close()

    @staticmethod
    def _weighted_average(candidates: list[tuple[float, float, str]]) -> tuple[Optional[float], float]:
        if not candidates:
            return None, 0.0
        total_weight = sum(weight for _value, weight, _label in candidates)
        if total_weight <= 1e-6:
            return None, 0.0
        weighted_value = sum(value * weight for value, weight, _label in candidates) / total_weight
        return weighted_value, total_weight

    def process_frame(self, frame_bgr: np.ndarray, *, now: Optional[float] = None) -> dict[str, Any]:
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("frame_bgr must be a non-empty BGR image")

        timestamp_mono = time.monotonic() if now is None else float(now)
        timestamp_wall = time.time()
        image = frame_bgr
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        need_face = (
            self.settings.scene_face_enabled
            or self.settings.scene_glance_enabled
            or self.settings.scene_mood_enabled
            or self.settings.scene_model_enabled
            or self.settings.scene_speaking_enabled
            or self.settings.scene_proximity_enabled
        )
        face_result = self._face_mesh.process(rgb) if (need_face and self._face_mesh is not None) else None
        pose_result = None
        if self._pose is not None and (
            self.settings.scene_person_enabled
            or self.settings.scene_arms_enabled
            or self.settings.scene_model_enabled
        ):
            pose_result = self._pose.process(rgb)

        scene_palette = sample_frame_palette(image, stop_count=CAMERA_PALETTE_STOP_COUNT)
        self._camera_palette_colors = smooth_palette(
            self._camera_palette_colors,
            scene_palette,
            self.settings.camera_smoothing,
        )
        scene_average_rgb = self._camera_palette_colors[len(self._camera_palette_colors) // 2]

        scene_env: dict[str, Any] = {
            "frame_brightness": 0.5,
            "warmth": 0.5,
            "edge_density": 0.0,
            "contrast": 0.5,
        }
        if self.settings.scene_environment_enabled:
            raw_env = extract_scene_environment(image)
            env_alpha = 0.012
            for key in ("frame_brightness", "warmth", "edge_density", "contrast"):
                self._scene_env_smoothed[key] = (
                    (1.0 - env_alpha) * self._scene_env_smoothed.get(key, raw_env[key])
                    + env_alpha * raw_env[key]
                )
            scene_env = dict(self._scene_env_smoothed)

        speaking_now = False
        face_state = None
        if face_result is not None and face_result.multi_face_landmarks:
            raw_mouth_open = normalized_distance(
                face_result.multi_face_landmarks[0], FACE_MOUTH_TOP, FACE_MOUTH_BOTTOM
            ) / max(
                1e-6,
                normalized_distance(face_result.multi_face_landmarks[0], FACE_FOREHEAD, FACE_CHIN),
            )
            speaking_now = self.settings.scene_speaking_enabled and self._speaking_detector.update(raw_mouth_open)
            face_state = extract_face_scene_state(
                face_result.multi_face_landmarks[0],
                width,
                height,
                speaking=speaking_now,
            )
        else:
            self._speaking_detector.update(0.0)

        pose_state = extract_pose_scene_state(
            pose_result.pose_landmarks if pose_result is not None else None
        )

        motion_regions = detect_motion_regions(
            image,
            self._motion_subtractor,
            max_regions=SCENE_OBJECT_MAX_COUNT,
        )
        motion = motion_regions[0] if motion_regions else None
        if motion is not None:
            self._last_motion_seen_time = timestamp_mono

        tracked_objects = self._scene_object_tracker.update(
            motion_regions if (
                self.settings.scene_object_enabled
                or self.settings.scene_model_enabled
                or self.settings.scene_motion_enabled
            ) else [],
            timestamp_mono,
        )

        hand_bboxes: list[tuple[int, int, int, int]] = []
        if self.settings.scene_object_detection_enabled and self._hands is not None:
            hand_result = self._hands.process(rgb)
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    xs = [int(round(landmark.x * width)) for landmark in hand_landmarks.landmark]
                    ys = [int(round(landmark.y * height)) for landmark in hand_landmarks.landmark]
                    hand_bboxes.append((
                        max(0, min(xs) - 10),
                        max(0, min(ys) - 10),
                        min(width, max(xs) - min(xs) + 20),
                        min(height, max(ys) - min(ys) + 20),
                    ))

        raw_detections: list[dict[str, Any]] = []
        if self.settings.scene_object_detection_enabled:
            raw_detections = self._scene_object_detector.update(
                image,
                hand_bboxes=hand_bboxes or None,
                memory=self._scene_memory if self.settings.scene_memory_enabled else None,
            )

        stable_detections = self._detection_buffer.update(raw_detections)
        real_detections = stable_detections if self.settings.scene_object_detection_enabled else []

        if self.settings.scene_memory_enabled and real_detections:
            for det in real_detections:
                self._scene_memory.record(det["label"], det["confidence"], det["center_x"])
            self._scene_memory.save()

        face_bbox = face_state["bbox"] if face_state is not None else None
        pose_bbox_norm = pose_state["bbox_norm"] if pose_state is not None else None
        for track in tracked_objects:
            label, confidence = classify_motion_blob(
                track,
                real_detections=real_detections,
                hand_bboxes=hand_bboxes or None,
                face_bbox=face_bbox,
                pose_bbox_norm=pose_bbox_norm,
                frame_width=width,
                frame_height=height,
            )
            track["label"] = label
            track["label_conf"] = confidence

        lead_object = tracked_objects[0] if tracked_objects else None
        motion_velocity = float(lead_object.get("velocity_px_s", 0.0)) if lead_object is not None else 0.0
        motion_velocity_norm = clip_unit(motion_velocity / 400.0)

        x_candidates: list[tuple[float, float, str]] = []
        y_candidates: list[tuple[float, float, str]] = []
        if self.settings.scene_person_enabled and pose_state is not None:
            lean_nudge = float(pose_state.get("body_lean", 0.0)) * 0.08
            x_candidates.append((clip_unit(float(pose_state["center_x"]) + lean_nudge), 1.0, "pose"))
            y_candidates.append((float(pose_state["center_y"]), 1.0, "pose"))
        if self.settings.scene_face_enabled and face_state is not None:
            x_candidates.append((float(face_state["center_x"]), 1.15, "face"))
            y_candidates.append((float(face_state["center_y"]), 1.15, "face"))
        if self.settings.scene_glance_enabled and face_state is not None:
            x_candidates.append((float(face_state["attention_x"]), 1.55, "glance"))
        if self.settings.scene_motion_enabled and motion is not None:
            motion_weight = 0.8 + min(1.4, float(motion["area_ratio"]) * 12.0 + motion_velocity_norm * 0.6)
            x_candidates.append((float(motion["center_x"]), motion_weight, "motion"))
            y_candidates.append((float(motion["center_y"]), motion_weight, "motion"))
        if self.settings.scene_object_enabled and lead_object is not None:
            x_candidates.append((float(lead_object["center_x"]), 0.7, "object"))
            y_candidates.append((float(lead_object["center_y"]), 0.7, "object"))

        raw_focus_x, focus_weight = self._weighted_average(x_candidates)
        raw_focus_y, _focus_weight_y = self._weighted_average(y_candidates)

        if raw_focus_x is not None:
            if self._scene_smoothed_focus_x is None:
                self._scene_smoothed_focus_x = float(raw_focus_x)
            else:
                smoothing = self.settings.camera_smoothing
                self._scene_smoothed_focus_x = (
                    (1.0 - smoothing) * self._scene_smoothed_focus_x
                    + smoothing * float(raw_focus_x)
                )
            if raw_focus_y is not None:
                if self._scene_smoothed_focus_y is None:
                    self._scene_smoothed_focus_y = float(raw_focus_y)
                else:
                    smoothing = self.settings.camera_smoothing
                    self._scene_smoothed_focus_y = (
                        (1.0 - smoothing) * self._scene_smoothed_focus_y
                        + smoothing * float(raw_focus_y)
                    )
        elif (timestamp_mono - self._last_motion_seen_time) > CAMERA_MOTION_IDLE_TIMEOUT_SEC:
            self._scene_smoothed_focus_x = None
            self._scene_smoothed_focus_y = None

        focus_x = self._scene_smoothed_focus_x
        focus_y = self._scene_smoothed_focus_y

        env_warmth = float(scene_env["warmth"])
        env_brightness_factor = float(scene_env["frame_brightness"])
        warm_tint = (255, 190, 90)
        cool_tint = (60, 130, 255)
        environment_tint_rgb = interpolate_rgb(cool_tint, warm_tint, env_warmth)

        ambient_rgb = interpolate_rgb(CAMERA_IDLE_BASE, scene_average_rgb, 0.38)
        if self.settings.scene_environment_enabled:
            ambient_rgb = interpolate_rgb(ambient_rgb, environment_tint_rgb, 0.22)
        accent_rgb = scene_average_rgb

        mood_label = "ambient"
        mood_delta = 0
        if face_state is not None:
            if self.settings.scene_mood_enabled:
                mood_label = str(face_state["mood_label"])
                mood_delta = int(face_state["mood_energy_delta"])
                accent_rgb = interpolate_rgb(accent_rgb, tuple(face_state["mood_rgb"]), 0.58)
                ambient_rgb = interpolate_rgb(ambient_rgb, tuple(face_state["mood_rgb"]), 0.2)
            if self.settings.scene_proximity_enabled:
                proximity = float(face_state["face_proximity"])
                mood_delta += int(round(lerp(proximity, 0.0, 1.0, 4.0, -8.0)))
                accent_rgb = interpolate_rgb(accent_rgb, warm_tint, proximity * 0.38)

        if motion is not None and self.settings.scene_motion_enabled:
            motion_rgb = tuple(motion["rgb"])
            if is_neutral_rgb(motion_rgb, saturation_threshold=0.08):
                motion_rgb = interpolate_rgb(scene_average_rgb, warm_tint, 0.4)
            motion_blend = 0.38 + motion_velocity_norm * 0.22
            accent_rgb = interpolate_rgb(accent_rgb, motion_rgb, min(0.65, motion_blend))

        if lead_object is not None and self.settings.scene_object_enabled:
            accent_rgb = interpolate_rgb(accent_rgb, tuple(lead_object["rgb"]), 0.32)

        if self.settings.scene_environment_enabled and float(scene_env["edge_density"]) > 0.04:
            complexity_boost = min(0.3, (float(scene_env["edge_density"]) - 0.04) * 8.0)
            hue, saturation, value = colorsys.rgb_to_hsv(
                accent_rgb[0] / 255.0, accent_rgb[1] / 255.0, accent_rgb[2] / 255.0
            )
            boosted_saturation = min(1.0, saturation + complexity_boost * 0.5)
            r_channel, g_channel, b_channel = colorsys.hsv_to_rgb(hue, boosted_saturation, value)
            accent_rgb = (
                int(round(r_channel * 255)),
                int(round(g_channel * 255)),
                int(round(b_channel * 255)),
            )

        motion_ratio = float(motion["area_ratio"]) if (motion is not None and self.settings.scene_motion_enabled) else 0.0
        attention_bonus = 0
        if face_state is not None and self.settings.scene_glance_enabled:
            attention_bonus = int(round(abs(float(face_state["glance_x"])) * 8.0))
        arms_bonus = 10 if (
            pose_state is not None
            and self.settings.scene_arms_enabled
            and pose_state.get("arms_raised")
        ) else 0
        speaking_bonus = 6 if speaking_now and self.settings.scene_speaking_enabled else 0

        env_base_brightness = int(round(
            lerp(env_brightness_factor, 0.0, 1.0, 14.0, 32.0)
        )) if self.settings.scene_environment_enabled else 22

        scene_energy = env_base_brightness
        if x_candidates:
            scene_energy += 12
        if motion_ratio > 0.0:
            scene_energy += int(round(lerp(
                motion_ratio,
                CAMERA_MOTION_MIN_AREA_RATIO,
                CAMERA_MOTION_MAX_AREA_RATIO,
                0.0,
                34.0,
            )))
        if motion_velocity_norm > 0.1:
            scene_energy += int(round(motion_velocity_norm * 10.0))
        scene_energy = max(
            14,
            min(100, scene_energy + mood_delta + attention_bonus + arms_bonus + speaking_bonus),
        )

        active_signals = [label for _value, _weight, label in x_candidates]
        focus_confidence = clip_unit(focus_weight / 4.0)
        focus_spread = min(0.32, 0.08 + (motion_ratio * 0.45) + ((1.0 - focus_confidence) * 0.08)) if focus_x is not None else None
        summary_parts = [f"mood={mood_label}", f"energy={scene_energy}%"]
        if focus_x is not None:
            summary_parts.insert(0, f"focus={int(round(float(focus_x) * 100))}%")
        if speaking_now:
            summary_parts.append("speaking")
        if arms_bonus:
            summary_parts.append("arms_up")
        summary = " | ".join(summary_parts)

        return {
            "timestamp_unix": timestamp_wall,
            "timestamp_monotonic": timestamp_mono,
            "frame": {"width": width, "height": height},
            "summary": summary,
            "focus_x": focus_x,
            "focus_y": focus_y,
            "focus_confidence": focus_confidence,
            "focus_spread": focus_spread,
            "active_signals": active_signals,
            "palette": scene_palette,
            "average_rgb": scene_average_rgb,
            "ambient_rgb": ambient_rgb,
            "accent_rgb": accent_rgb,
            "scene_energy": scene_energy,
            "mood_label": mood_label,
            "speaking": speaking_now,
            "environment": scene_env,
            "face": face_state,
            "pose": pose_state,
            "motion": motion,
            "tracked_objects": tracked_objects,
            "detections": real_detections,
            "hand_bboxes": hand_bboxes,
            "raw_focus_x": raw_focus_x,
            "raw_focus_y": raw_focus_y,
            "motion_velocity_px_s": motion_velocity,
            "motion_velocity_norm": motion_velocity_norm,
            "warnings": list(self._runtime_warnings),
        }

    def annotate_frame(self, frame_bgr: np.ndarray, result: dict[str, Any]) -> np.ndarray:
        frame = frame_bgr.copy()
        height, width = frame.shape[:2]
        face_state = result.get("face")
        pose_state = result.get("pose")
        motion = result.get("motion")
        tracked_objects = result.get("tracked_objects") or []
        detections = result.get("detections") or []
        hand_bboxes = result.get("hand_bboxes") or []
        mood_label = str(result.get("mood_label", "ambient"))
        speaking_now = bool(result.get("speaking"))

        focus_x = result.get("focus_x")
        focus_y = result.get("focus_y")
        if focus_x is not None:
            pointer_x = int(round(float(focus_x) * max(1, width - 1)))
            cv2.line(frame, (pointer_x, 0), (pointer_x, height), (160, 160, 255), 1)
        if focus_y is not None:
            pointer_y = int(round(float(focus_y) * max(1, height - 1)))
            cv2.line(frame, (0, pointer_y), (width, pointer_y), (110, 110, 210), 1)
        if focus_x is not None and focus_y is not None:
            focus_point = (
                int(round(float(focus_x) * max(1, width - 1))),
                int(round(float(focus_y) * max(1, height - 1))),
            )
            cv2.circle(frame, focus_point, 7, (255, 255, 255), 1)

        self._draw_focus_band(
            frame,
            focus_x=focus_x,
            focus_confidence=result.get("focus_confidence"),
            focus_spread=result.get("focus_spread"),
            ambient_rgb=tuple(result.get("ambient_rgb") or CAMERA_IDLE_BASE),
            accent_rgb=tuple(result.get("accent_rgb") or WHITE),
        )
        self._draw_color_swatches(
            frame,
            ambient_rgb=tuple(result.get("ambient_rgb") or CAMERA_IDLE_BASE),
            accent_rgb=tuple(result.get("accent_rgb") or WHITE),
        )

        if self.settings.scene_model_enabled:
            if face_state is not None:
                x, y, w_box, h_box = face_state["bbox"]
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (255, 180, 0), 2)
                face_label = mood_label
                if speaking_now:
                    face_label += " speaking"
                prox_pct = int(round(float(face_state.get("face_proximity", 0.0)) * 100))
                face_label += f" prox:{prox_pct}%"
                cv2.putText(
                    frame,
                    face_label,
                    (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (255, 180, 0),
                    1,
                )
                line_y = y + max(14, h_box // 3)
                start_x = x + (w_box // 2)
                end_x = int(round(clip_unit(float(face_state["attention_x"])) * max(1, width - 1)))
                cv2.line(frame, (start_x, line_y), (end_x, line_y), (255, 180, 0), 2)
                eye_cx = x + w_box // 2
                eye_cy = y + h_box // 3
                glance_x = float(face_state["glance_x"])
                arrow_end = (int(eye_cx + glance_x * max(60, w_box // 2)), eye_cy)
                cv2.arrowedLine(frame, (eye_cx, eye_cy), arrow_end, (255, 220, 60), 2, tipLength=0.35)

            if pose_state is not None and self.settings.scene_person_enabled:
                bx = int(round(float(pose_state["bbox_norm"][0]) * width))
                by = int(round(float(pose_state["bbox_norm"][1]) * height))
                bw = int(round(float(pose_state["bbox_norm"][2]) * width))
                bh = int(round(float(pose_state["bbox_norm"][3]) * height))
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 255, 140), 2)
                pose_label = "Person"
                if pose_state.get("arms_raised") and self.settings.scene_arms_enabled:
                    pose_label += " arms_up"
                cv2.putText(
                    frame,
                    pose_label,
                    (bx, max(18, by - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (80, 255, 140),
                    1,
                )

            if motion is not None and self.settings.scene_motion_enabled:
                x, y, w_box, h_box = motion["bbox"]
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 220, 255), 2)
                velocity = float(result.get("motion_velocity_px_s", 0.0))
                velocity_str = f" v={int(round(velocity))}px/s" if velocity > 5 else ""
                cv2.putText(
                    frame,
                    f"Motion {int(round(float(motion['area_ratio']) * 1000.0))}{velocity_str}",
                    (x, min(height - 12, y + h_box + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 220, 255),
                    1,
                )

            if tracked_objects and self.settings.scene_object_enabled:
                for track in tracked_objects[:SCENE_OBJECT_MAX_COUNT]:
                    x, y, w_box, h_box = track["bbox"]
                    cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (170, 120, 255), 1)
                    track_label = f"#{track['id']} {track.get('label', 'object')}"
                    confidence = float(track.get("label_conf", 0.0))
                    if confidence > 0.45:
                        track_label += f" {int(round(confidence * 100))}%"
                    cv2.putText(
                        frame,
                        track_label,
                        (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (170, 120, 255),
                        1,
                    )

            if self.settings.scene_object_detection_enabled:
                for det in detections:
                    dx, dy, dw, dh = det["bbox"]
                    held = bool(det.get("held"))
                    box_color = (0, 220, 255) if held else (255, 220, 60)
                    det_conf = int(round(float(det["confidence"]) * 100))
                    prefix = "held " if held else ""
                    cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), box_color, 1 if not held else 2)
                    cv2.putText(
                        frame,
                        f"{prefix}{det['label']} {det_conf}%",
                        (dx, max(14, dy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        box_color,
                        1,
                    )

                for hx, hy, hw, hh in hand_bboxes:
                    cv2.rectangle(frame, (hx, hy), (hx + hw, hy + hh), (0, 180, 255), 1)
                    cv2.putText(
                        frame,
                        "hand",
                        (hx, max(14, hy - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.40,
                        (0, 180, 255),
                        1,
                    )

        summary = str(result.get("summary", ""))
        cue_text = ", ".join(result.get("active_signals") or []) or "none"
        env = result.get("environment") or {}
        env_text = (
            f"env bri={int(round(float(env.get('frame_brightness', 0.5)) * 100))}% "
            f"warm={int(round(float(env.get('warmth', 0.5)) * 100))}% "
            f"edge={round(float(env.get('edge_density', 0.0)) * 100, 1)}%"
        )
        cv2.putText(frame, summary[:96], (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1)
        cv2.putText(frame, f"signals: {cue_text}"[:110], (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 215, 255), 1)
        cv2.putText(frame, env_text[:110], (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 255), 1)
        return frame

    def _draw_focus_band(
        self,
        frame: np.ndarray,
        *,
        focus_x: Optional[float],
        focus_confidence: Optional[float],
        focus_spread: Optional[float],
        ambient_rgb: tuple[int, int, int],
        accent_rgb: tuple[int, int, int],
    ) -> None:
        height, width = frame.shape[:2]
        margin = 12
        bar_top = max(84, height - 56)
        bar_height = 14
        usable_width = max(1, width - (margin * 2))
        base_rgb = scale_rgb(ambient_rgb, 0.4, floor=6)
        sigma = max(0.02, float(focus_spread if focus_spread is not None else 0.12))
        confidence = clip_unit(0.25 if focus_x is None else float(focus_confidence if focus_confidence is not None else 0.5))

        for offset in range(usable_width):
            x_norm = offset / float(max(1, usable_width - 1))
            blend = 0.0
            if focus_x is not None:
                blend = math.exp(-0.5 * (((x_norm - float(focus_x)) / sigma) ** 2)) * confidence
            rgb = interpolate_rgb(base_rgb, accent_rgb, blend)
            x = margin + offset
            cv2.line(frame, (x, bar_top), (x, bar_top + bar_height), rgb_to_bgr(rgb), 1)

        cv2.rectangle(frame, (margin, bar_top), (width - margin, bar_top + bar_height), (40, 40, 40), 1)
        if focus_x is not None:
            px = margin + int(round(clip_unit(float(focus_x)) * usable_width))
            cv2.line(frame, (px, bar_top - 4), (px, bar_top + bar_height + 4), (255, 255, 255), 1)

    def _draw_color_swatches(
        self,
        frame: np.ndarray,
        *,
        ambient_rgb: tuple[int, int, int],
        accent_rgb: tuple[int, int, int],
    ) -> None:
        height, width = frame.shape[:2]
        chip_size = 16
        top = max(88, height - 84)
        left = max(12, width - 180)
        cv2.putText(frame, "ambient", (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)
        cv2.rectangle(frame, (left, top), (left + chip_size, top + chip_size), rgb_to_bgr(ambient_rgb), -1)
        cv2.rectangle(frame, (left, top), (left + chip_size, top + chip_size), (255, 255, 255), 1)
        accent_left = left + 68
        cv2.putText(frame, "accent", (accent_left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)
        cv2.rectangle(frame, (accent_left, top), (accent_left + chip_size, top + chip_size), rgb_to_bgr(accent_rgb), -1)
        cv2.rectangle(frame, (accent_left, top), (accent_left + chip_size, top + chip_size), (255, 255, 255), 1)
