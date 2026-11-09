from __future__ import annotations

import copy
import glob
import hashlib
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# ----------------------------------------------------------------------
# Optional scientific stack
# ----------------------------------------------------------------------
_SCIPY_TRIED = False
_CUBIC_SPLINE = None


def _cubic_spline():
    global _SCIPY_TRIED, _CUBIC_SPLINE
    if not _SCIPY_TRIED:
        try:
            from scipy.interpolate import CubicSpline as _CS
            _CUBIC_SPLINE = _CS
        except Exception:
            _CUBIC_SPLINE = None
        _SCIPY_TRIED = True
    return _CUBIC_SPLINE


_TORCH_TRIED = False
_TORCH = None


def _torch_module():
    global _TORCH_TRIED, _TORCH
    if not _TORCH_TRIED:
        try:
            import torch as _t
            _TORCH = _t
        except Exception:
            _TORCH = None
        _TORCH_TRIED = True
    return _TORCH


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


# ======================================================================
# 1. Animation specification parser
# ======================================================================
DEFAULT_SPEC: Dict[str, Any] = {
    "name": "Procedural Motion",
    "duration": 3.0,
    "fps": 24,
    "frame_start": 1,
    "interpolation": "BEZIER",
    "keyframe_step": 1,
    "import_model": None,
    "import_as": None,
    "camera": {
        "type": "orbit",
        "location": [6.0, -6.0, 4.0],
        "target": [0.0, 0.0, 0.0],
        "from": [6.0, -6.0, 4.0],
        "to": [-6.0, 6.0, 4.0],
        "offset": [4.0, -4.0, 3.0],
        "target_object": None,
        "radius": 6.0,
        "height": 3.0,
        "speed": 45.0,
        "phase": 0.0,
        "lens": 35.0,
        "keyframes": [],
    },
    "objects": [],
    "events": [],
}

DEFAULT_OBJECT: Dict[str, Any] = {
    "name": "Object",
    "kind": "cube",
    "location": [0.0, 0.0, 0.0],
    "rotation": [0.0, 0.0, 0.0],
    "scale": [1.0, 1.0, 1.0],
    "color": [0.85, 0.87, 0.90, 1.0],
    "motion": {"type": "idle"},
}

VALID_KINDS = {
    "cube",
    "sphere",
    "cylinder",
    "torus",
    "cone",
    "plane",
    "gear",
    "crank",
    "slider",
    "empty",
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


class AnimationSpecificationParser:
    """
    Normalizes user/template JSON into a predictable internal specification.
    """

    @staticmethod
    def parse(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raw = raw or {}
        spec = copy.deepcopy(DEFAULT_SPEC)
        _deep_update(spec, raw)

        spec["name"] = str(spec.get("name", "Procedural Motion"))
        spec["duration"] = max(0.04, float(spec.get("duration", 3.0)))
        spec["fps"] = max(1, int(spec.get("fps", 24)))
        spec["frame_start"] = max(1, int(spec.get("frame_start", 1)))
        spec["interpolation"] = str(spec.get("interpolation", "BEZIER")).upper()
        spec["keyframe_step"] = max(1, int(spec.get("keyframe_step", 1)))

        frame_end = spec["frame_start"] + max(1, int(round(spec["duration"] * spec["fps"])))

        objects: List[Dict[str, Any]] = []
        used_names: Dict[str, int] = {}
        for idx, item in enumerate(spec.get("objects", []) or []):
            obj = copy.deepcopy(DEFAULT_OBJECT)
            _deep_update(obj, item if isinstance(item, dict) else {})

            name = str(obj.get("name") or f"Object_{idx + 1}")
            if name in used_names:
                used_names[name] += 1
                name = f"{name}_{used_names[name]}"
            else:
                used_names[name] = 1
            obj["name"] = name

            kind = str(obj.get("kind", "cube")).lower()
            obj["kind"] = kind if kind in VALID_KINDS else "cube"

            obj["location"] = _vec3(obj.get("location"), [0.0, 0.0, 0.0])
            obj["rotation"] = _vec3(obj.get("rotation"), [0.0, 0.0, 0.0])
            obj["scale"] = _vec3(obj.get("scale"), [1.0, 1.0, 1.0])
            obj["color"] = _rgba(obj.get("color"), [0.85, 0.87, 0.90, 1.0])

            motion = obj.get("motion", {})
            if isinstance(motion, str):
                motion = {"type": motion}
            obj["motion"] = motion if isinstance(motion, dict) else {"type": "idle"}
            obj["motion"].setdefault("type", "idle")

            objects.append(obj)

        spec["objects"] = objects

        camera = spec.get("camera", {})
        if isinstance(camera, str):
            camera = {"type": camera}
        spec["camera"] = camera if isinstance(camera, dict) else copy.deepcopy(DEFAULT_SPEC["camera"])
        spec["camera"].setdefault("type", "orbit")
        spec["camera"]["target"] = _vec3(spec["camera"].get("target"), [0.0, 0.0, 0.0])
        spec["camera"]["location"] = _vec3(spec["camera"].get("location"), [6.0, -6.0, 4.0])
        spec["camera"]["from"] = _vec3(spec["camera"].get("from"), spec["camera"]["location"])
        spec["camera"]["to"] = _vec3(spec["camera"].get("to"), [-6.0, 6.0, 4.0])
        spec["camera"]["offset"] = _vec3(spec["camera"].get("offset"), [4.0, -4.0, 3.0])
        spec["camera"]["radius"] = float(spec["camera"].get("radius", 6.0))
        spec["camera"]["height"] = float(spec["camera"].get("height", 3.0))
        spec["camera"]["speed"] = float(spec["camera"].get("speed", 45.0))
        spec["camera"]["phase"] = float(spec["camera"].get("phase", 0.0))
        spec["camera"]["lens"] = float(spec["camera"].get("lens", 35.0))

        events = []
        for ev in spec.get("events", []) or []:
            if isinstance(ev, dict):
                frame = int(ev.get("frame", spec["frame_start"]))
                frame = max(spec["frame_start"], min(frame_end, frame))
                events.append({"name": str(ev.get("name", "Event")), "frame": frame})
        spec["events"] = events

        if spec.get("import_model"):
            spec["import_model"] = str(spec["import_model"])
        return spec


def _vec3(value: Any, default: List[float]) -> List[float]:
    try:
        arr = np.asarray(value, dtype=float).flatten().tolist()
    except Exception:
        arr = []
    while len(arr) < 3:
        arr.append(float(default[len(arr)]) if len(arr) < len(default) else 0.0)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _rgba(value: Any, default: List[float]) -> List[float]:
    try:
        arr = np.asarray(value, dtype=float).flatten().tolist()
    except Exception:
        arr = []
    while len(arr) < 4:
        arr.append(float(default[len(arr)]) if len(arr) < len(default) else 1.0)
    return [float(np.clip(arr[i], 0.0, 1.0)) for i in range(4)]


# ======================================================================
# 2. Timeline manager
# ======================================================================
@dataclass
class Timeline:
    fps: int = 24
    frame_start: int = 1
    frame_end: int = 73
    duration: float = 3.0

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "Timeline":
        fps = max(1, int(spec.get("fps", 24)))
        start = max(1, int(spec.get("frame_start", 1)))
        duration = max(0.04, float(spec.get("duration", 3.0)))
        end = start + max(1, int(round(duration * fps)))
        return cls(fps=fps, frame_start=start, frame_end=end, duration=duration)

    @property
    def frames(self) -> range:
        return range(self.frame_start, self.frame_end + 1)

    @property
    def frame_count(self) -> int:
        return self.frame_end - self.frame_start + 1

    def time_at_frame(self, frame: int) -> float:
        return float(frame - self.frame_start) / float(max(1, self.fps))

    def frame_at_time(self, t: float) -> int:
        return int(self.frame_start + round(max(0.0, t) * self.fps))


# ======================================================================
# 3. Interpolation system
# ======================================================================
class InterpolationSystem:
    BLENDER_MAP = {
        "BEZIER": "BEZIER",
        "LINEAR": "LINEAR",
        "CONSTANT": "CONSTANT",
        "SINE": "SINE",
        "QUAD": "QUAD",
        "CUBIC": "CUBIC",
        "QUART": "QUART",
        "QUINT": "QUINT",
        "EXPO": "EXPO",
        "CIRC": "CIRC",
    }

    @classmethod
    def blender_name(cls, name: str) -> str:
        return cls.BLENDER_MAP.get(str(name).upper(), "BEZIER")

    @staticmethod
    def sample_curve(frames: List[int], values: List[float], sample_count: int = 64) -> List[float]:
        CS = _cubic_spline()
        if CS is not None and len(frames) >= 4:
            x = np.asarray(frames, dtype=float)
            y = np.asarray(values, dtype=float)
            spline = CS(x, y, bc_type="natural")
            xs = np.linspace(x[0], x[-1], max(2, int(sample_count)))
            return [float(v) for v in spline(xs)]
        if not values:
            return []
        xs = np.linspace(frames[0], frames[-1], max(2, int(sample_count))) if len(frames) > 1 else np.asarray(frames)
        ys = np.interp(xs, np.asarray(frames, dtype=float), np.asarray(values, dtype=float))
        return [float(v) for v in ys]


def ease(u: float, kind: str = "smoothstep") -> float:
    u = float(np.clip(u, 0.0, 1.0))
    kind = str(kind or "smoothstep").lower()

    if kind.startswith("torch_"):
        torch = _torch_module()
        if torch is not None:
            t = torch.tensor(u, dtype=torch.float32)
            if kind == "torch_smoothstep":
                y = t * t * (3.0 - 2.0 * t)
            elif kind == "torch_sine":
                y = torch.sin(t * math.pi / 2.0)
            elif kind == "torch_elastic":
                y = torch.pow(2.0, -10.0 * t) * torch.sin((t * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0
            else:
                y = t * t * (3.0 - 2.0 * t)
            return float(torch.clamp(y, 0.0, 1.5).item())
        kind = "smoothstep"

    if kind == "linear":
        return u
    if kind in {"smoothstep", "ease_in_out"}:
        return u * u * (3.0 - 2.0 * u)
    if kind == "sine":
        return 0.5 * (1.0 - math.cos(math.pi * u))
    if kind == "elastic":
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return pow(2.0, -10.0 * u) * math.sin((u * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0
    if kind == "back":
        c1 = 1.70158
        c3 = c1 + 1.0
        return c3 * u * u * u - c1 * u * u
    if kind == "bounce":
        n1 = 7.5625
        d1 = 2.75
        if u < 1.0 / d1:
            return n1 * u * u
        if u < 2.0 / d1:
            u -= 1.5 / d1
            return n1 * u * u + 0.75
        if u < 2.5 / d1:
            u -= 2.25 / d1
            return n1 * u * u + 0.9375
        u -= 2.625 / d1
        return n1 * u * u + 0.984375
    return u * u * (3.0 - 2.0 * u)


# ======================================================================
# 4. Object motion engine
# ======================================================================
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _waveform(kind: str, phase: float) -> float:
    kind = str(kind or "sin").lower()
    if kind == "abs":
        return abs(math.sin(phase))
    if kind == "cos":
        return math.cos(phase)
    if kind == "saw":
        return ((phase / (2.0 * math.pi)) % 1.0) * 2.0 - 1.0
    return math.sin(phase)


class ObjectMotionEngine:
    """
    Computes analytic object transforms for a given time.
    Supported initial motion types:
      translation, rotation, scaling, repeated movement, circular motion,
      mechanical rotation, crank/slider, basic interaction, idle.
    """

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.duration = max(0.04, float(spec.get("duration", 3.0)))
        self.objects_by_name = {o["name"]: o for o in spec.get("objects", [])}

    def compute_base_all(self, t: float) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for obj in self.spec.get("objects", []):
            out[obj["name"]] = self.base_transform(obj, t)
        return out

    def apply_interactions(self, transforms: Dict[str, Dict[str, Any]], t: float) -> None:
        for obj in self.spec.get("objects", []):
            motion = obj.get("motion", {})
            if str(motion.get("type", "")).lower() != "interaction":
                continue
            name = obj["name"]
            target_name = motion.get("target")
            if name not in transforms or target_name not in transforms:
                continue

            own = np.asarray(transforms[name]["location"], dtype=float)
            target = np.asarray(transforms[target_name]["location"], dtype=float)
            diff = target - own
            dist = float(np.linalg.norm(diff))
            radius = max(1e-4, float(motion.get("radius", 2.0)))
            strength = float(motion.get("strength", 0.45))
            mode = str(motion.get("mode", "attract")).lower()
            falloff = math.exp(-((dist / radius) ** 2)) if dist > 0 else 1.0

            if mode == "attract":
                own += diff * strength * falloff
            elif mode == "repel":
                if dist > 1e-5:
                    own -= (diff / dist) * strength * falloff * radius
            elif mode == "pulse":
                pass

            scale = np.asarray(transforms[name]["scale"], dtype=float)
            scale_factor = 1.0 + float(motion.get("scale_effect", 0.18)) * falloff
            scale *= max(0.05, scale_factor)

            rot = np.asarray(transforms[name]["rotation_euler"], dtype=float)
            rot[2] += float(motion.get("rotation_effect", 0.35)) * falloff * math.pi

            transforms[name]["location"] = own.tolist()
            transforms[name]["scale"] = scale.tolist()
            transforms[name]["rotation_euler"] = rot.tolist()

    def base_transform(self, obj: Dict[str, Any], t: float) -> Dict[str, Any]:
        loc = np.asarray(obj["location"], dtype=float).copy()
        rot = np.deg2rad(np.asarray(obj["rotation"], dtype=float)).copy()
        scale = np.asarray(obj["scale"], dtype=float).copy()
        motion = obj.get("motion", {})
        mtype = str(motion.get("type", "idle")).lower()
        u = float(np.clip(t / self.duration, 0.0, 1.0))
        loop = bool(motion.get("loop", False))
        if loop:
            u = float((t % self.duration) / self.duration)

        if mtype == "translation":
            frm = np.asarray(motion.get("from", loc), dtype=float)
            to = np.asarray(motion.get("to", loc + np.array([1.0, 0.0, 0.0])), dtype=float)
            p = ease(u, motion.get("easing", "smoothstep"))
            loc = frm + (to - frm) * p

        elif mtype == "rotation":
            axis = AXIS_INDEX.get(str(motion.get("axis", "z")).lower(), 2)
            speed = float(motion.get("speed", 90.0))
            direction = float(motion.get("direction", 1.0))
            rot[axis] += math.radians(speed * t * direction)
            if "oscillate_amplitude" in motion:
                amp = float(motion.get("oscillate_amplitude", 30.0))
                freq = float(motion.get("frequency", 0.5))
                phase = float(motion.get("phase", 0.0))
                rot[axis] += math.radians(amp * math.sin(2.0 * math.pi * freq * t + phase))

        elif mtype == "scaling":
            frm = np.asarray(motion.get("from_scale", scale), dtype=float)
            to = np.asarray(motion.get("to_scale", scale), dtype=float)
            p = ease(u, motion.get("easing", "smoothstep"))
            scale = frm + (to - frm) * p
            amp = float(motion.get("pulse_amplitude", 0.0))
            freq = float(motion.get("pulse_frequency", 0.0))
            if amp:
                scale *= 1.0 + amp * math.sin(2.0 * math.pi * freq * t)
            scale = np.maximum(scale, 0.01)

        elif mtype == "repeated":
            channel = str(motion.get("channel", "location")).lower()
            axis = AXIS_INDEX.get(str(motion.get("axis", "z")).lower(), 2)
            amp = float(motion.get("amplitude", 0.5))
            freq = float(motion.get("frequency", 1.0))
            phase = float(motion.get("phase", 0.0))
            wave = _waveform(motion.get("waveform", "sin"), 2.0 * math.pi * freq * t + phase)
            value = amp * wave
            if channel == "location":
                loc[axis] += value
            elif channel == "rotation":
                rot[axis] += math.radians(value)
            else:
                scale[axis] *= max(0.05, 1.0 + value)

        elif mtype == "circular":
            center = np.asarray(motion.get("center", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(motion.get("radius", 2.0))
            speed = float(motion.get("speed", 90.0))
            rpm = motion.get("rpm")
            if rpm is not None:
                speed = float(rpm) * 6.0
            phase = float(motion.get("phase", 0.0))
            plane = str(motion.get("plane", "xy")).lower()
            angle = math.radians(speed * t + phase)
            c, s = math.cos(angle), math.sin(angle)
            if plane == "xz":
                loc = center + np.array([radius * c, 0.0, radius * s])
                if bool(motion.get("face_tangent", False)):
                    rot[1] = -angle
            elif plane == "yz":
                loc = center + np.array([0.0, radius * c, radius * s])
                if bool(motion.get("face_tangent", False)):
                    rot[0] = angle
            else:
                loc = center + np.array([radius * c, radius * s, 0.0])
                if bool(motion.get("face_tangent", False)):
                    rot[2] = angle + math.pi / 2.0

        elif mtype == "mechanical_rotation":
            axis = AXIS_INDEX.get(str(motion.get("axis", "z")).lower(), 2)
            rpm = float(motion.get("rpm", 12.0))
            ratio = float(motion.get("ratio", 1.0))
            direction = float(motion.get("direction", 1.0))
            radius = float(motion.get("radius", 1.0))
            angle_rad = 2.0 * math.pi * (rpm / 60.0) * t * ratio * direction
            rot[axis] += angle_rad
            rack_axis = motion.get("rack_axis")
            if rack_axis:
                idx = AXIS_INDEX.get(str(rack_axis).lower(), 0)
                loc[idx] += radius * angle_rad

        elif mtype == "crank_slider":
            axis = AXIS_INDEX.get(str(motion.get("axis", "x")).lower(), 0)
            rpm = float(motion.get("rpm", 20.0))
            ratio = float(motion.get("ratio", 1.0))
            direction = float(motion.get("direction", 1.0))
            r = max(0.01, float(motion.get("radius", 0.8)))
            l = max(r + 0.01, float(motion.get("rod_length", 2.0)))
            theta = 2.0 * math.pi * (rpm / 60.0) * t * ratio * direction
            sin_term = r * math.sin(theta)
            root = math.sqrt(max(0.0, l * l - sin_term * sin_term))
            x = r * math.cos(theta) + root
            x0 = r + l
            base = np.asarray(motion.get("base", obj["location"]), dtype=float).copy()
            base[axis] += x - x0
            loc = base

        elif mtype == "interaction":
            pass

        return {
            "location": [float(v) for v in loc],
            "rotation_euler": [float(v) for v in rot],
            "scale": [float(v) for v in scale],
        }


# ======================================================================
# 5. Camera animation manager
# ======================================================================
def _normalize_quat(q: List[float]) -> List[float]:
    arr = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(arr))
    if n < 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v / n) for v in arr]


def _matrix_to_quaternion(m: np.ndarray) -> List[float]:
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _normalize_quat([w, x, y, z])


def _look_at_quaternion(eye: np.ndarray, target: np.ndarray, up: Optional[np.ndarray] = None) -> List[float]:
    up = np.asarray(up or [0.0, 0.0, 1.0], dtype=float)
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    n = float(np.linalg.norm(forward))
    if n < 1e-6:
        forward = np.array([0.0, -1.0, 0.0])
    else:
        forward = forward / n

    z = -forward
    x = np.cross(up, z)
    if float(np.linalg.norm(x)) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
    x = x / max(1e-8, float(np.linalg.norm(x)))
    y = np.cross(z, x)
    r = np.column_stack((x, y, z))
    return _matrix_to_quaternion(r)


class CameraAnimationManager:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        self.camera = spec.get("camera", {})
        self.duration = max(0.04, float(spec.get("duration", 3.0)))
        self.fps = max(1, int(spec.get("fps", 24)))
        self.frame_start = max(1, int(spec.get("frame_start", 1)))

    def compute(self, t: float, object_transforms: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        cam = self.camera
        ctype = str(cam.get("type", "orbit")).lower()
        target = np.asarray(cam.get("target", [0.0, 0.0, 0.0]), dtype=float)
        lens = float(cam.get("lens", 35.0))
        u = float(np.clip(t / self.duration, 0.0, 1.0))

        if ctype == "static":
            loc = np.asarray(cam.get("location", [6.0, -6.0, 4.0]), dtype=float)

        elif ctype == "orbit":
            radius = float(cam.get("radius", 6.0))
            height = float(cam.get("height", 3.0))
            speed = float(cam.get("speed", 45.0))
            phase = float(cam.get("phase", 0.0))
            angle = math.radians(speed * t + phase)
            loc = target + np.array([radius * math.cos(angle), radius * math.sin(angle), height])

        elif ctype == "dolly":
            frm = np.asarray(cam.get("from", [6.0, -6.0, 4.0]), dtype=float)
            to = np.asarray(cam.get("to", [-6.0, 6.0, 4.0]), dtype=float)
            p = ease(u, cam.get("easing", "smoothstep"))
            loc = frm + (to - frm) * p

        elif ctype == "crane":
            frm = np.asarray(cam.get("from", [0.0, -8.0, 1.0]), dtype=float)
            to = np.asarray(cam.get("to", [0.0, -8.0, 6.0]), dtype=float)
            p = ease(u, cam.get("easing", "smoothstep"))
            loc = frm + (to - frm) * p

        elif ctype == "follow":
            target_object = cam.get("target_object")
            offset = np.asarray(cam.get("offset", [4.0, -4.0, 3.0]), dtype=float)
            if object_transforms and target_object in object_transforms:
                target = np.asarray(object_transforms[target_object]["location"], dtype=float)
            loc = target + offset

        elif ctype == "manual":
            keys = cam.get("keyframes", []) or []
            parsed = []
            for k in keys:
                if not isinstance(k, dict):
                    continue
                if "frame" in k:
                    kt = float(int(k["frame"]) - self.frame_start) / float(self.fps)
                else:
                    kt = float(k.get("time", 0.0))
                parsed.append(
                    (
                        kt,
                        np.asarray(k.get("location", [6.0, -6.0, 4.0]), dtype=float),
                        np.asarray(k.get("target", [0.0, 0.0, 0.0]), dtype=float),
                    )
                )
            parsed.sort(key=lambda x: x[0])
            if not parsed:
                loc = np.asarray(cam.get("location", [6.0, -6.0, 4.0]), dtype=float)
            elif t <= parsed[0][0]:
                loc, target = parsed[0][1], parsed[0][2]
            elif t >= parsed[-1][0]:
                loc, target = parsed[-1][1], parsed[-1][2]
            else:
                for i in range(len(parsed) - 1):
                    t0, l0, tg0 = parsed[i]
                    t1, l1, tg1 = parsed[i + 1]
                    if t0 <= t <= t1:
                        p = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                        loc = l0 + (l1 - l0) * p
                        target = tg0 + (tg1 - tg0) * p
                        break
                else:
                    loc, target = parsed[-1][1], parsed[-1][2]
        else:
            loc = np.asarray(cam.get("location", [6.0, -6.0, 4.0]), dtype=float)

        quat = _look_at_quaternion(loc, target)
        return {
            "location": [float(v) for v in loc],
            "rotation_quaternion": quat,
            "lens": float(lens),
        }


# ======================================================================
# 6. Keyframe generation engine
# ======================================================================
class KeyframeGenerationEngine:
    def generate(self, spec: Dict[str, Any], timeline: Timeline) -> Dict[str, Any]:
        step = max(1, int(spec.get("keyframe_step", 1)))
        all_frames = list(timeline.frames)
        frames = all_frames[::step]
        if all_frames and (not frames or frames[-1] != all_frames[-1]):
            frames.append(all_frames[-1])

        motion_engine = ObjectMotionEngine(spec)
        camera_manager = CameraAnimationManager(spec)

        channels: Dict[str, Any] = {"objects": {}, "camera": []}
        for obj in spec.get("objects", []):
            channels["objects"][obj["name"]] = []

        for frame in frames:
            t = timeline.time_at_frame(frame)
            transforms = motion_engine.compute_base_all(t)
            motion_engine.apply_interactions(transforms, t)

            for name, tr in transforms.items():
                channels["objects"].setdefault(name, []).append(
                    {
                        "frame": int(frame),
                        "location": [float(v) for v in tr["location"]],
                        "rotation_euler": [float(v) for v in tr["rotation_euler"]],
                        "scale": [float(v) for v in tr["scale"]],
                    }
                )

            cam = camera_manager.compute(t, transforms)
            channels["camera"].append(
                {
                    "frame": int(frame),
                    "location": [float(v) for v in cam["location"]],
                    "rotation_quaternion": [float(v) for v in cam["rotation_quaternion"]],
                    "lens": float(cam["lens"]),
                }
            )

        return channels


# ======================================================================
# 7. Procedural animation templates / sample projects
# ======================================================================
def template_rotating_object() -> Dict[str, Any]:
    return {
        "name": "Rotating Object",
        "duration": 3.0,
        "fps": 24,
        "camera": {"type": "static", "location": [4.5, -4.5, 3.0], "target": [0, 0, 0]},
        "objects": [
            {
                "name": "Spinner",
                "kind": "cube",
                "location": [0, 0, 0],
                "scale": [1.2, 1.2, 1.2],
                "color": [0.95, 0.45, 0.20, 1.0],
                "motion": {"type": "rotation", "axis": "z", "speed": 120.0},
            }
        ],
    }


def template_translation_scaling() -> Dict[str, Any]:
    return {
        "name": "Translation + Scaling",
        "duration": 4.0,
        "fps": 24,
        "camera": {"type": "orbit", "radius": 7.0, "height": 3.0, "speed": 25.0},
        "objects": [
            {
                "name": "Mover",
                "kind": "sphere",
                "location": [-3.0, 0.0, 0.5],
                "scale": [0.8, 0.8, 0.8],
                "color": [0.20, 0.70, 1.00, 1.0],
                "motion": {
                    "type": "translation",
                    "from": [-3.0, 0.0, 0.5],
                    "to": [3.0, 0.0, 0.5],
                    "easing": "ease_in_out",
                },
            },
            {
                "name": "Pulsar",
                "kind": "torus",
                "location": [0.0, -2.5, 0.4],
                "scale": [1.0, 1.0, 1.0],
                "color": [0.65, 0.95, 0.45, 1.0],
                "motion": {
                    "type": "scaling",
                    "from_scale": [0.7, 0.7, 0.7],
                    "to_scale": [1.35, 1.35, 1.35],
                    "easing": "sine",
                    "pulse_amplitude": 0.12,
                    "pulse_frequency": 1.0,
                },
            },
        ],
    }


def template_camera_movement() -> Dict[str, Any]:
    return {
        "name": "Camera Movement",
        "duration": 4.0,
        "fps": 24,
        "camera": {"type": "orbit", "radius": 6.5, "height": 2.5, "speed": 60.0, "target": [0, 0, 0.5]},
        "objects": [
            {
                "name": "Subject",
                "kind": "cone",
                "location": [0, 0, 0.5],
                "scale": [1.0, 1.0, 1.4],
                "color": [0.90, 0.35, 0.65, 1.0],
                "motion": {"type": "rotation", "axis": "z", "speed": 30.0},
            }
        ],
    }


def template_circular_motion() -> Dict[str, Any]:
    return {
        "name": "Circular Motion",
        "duration": 4.0,
        "fps": 24,
        "camera": {"type": "static", "location": [0.0, -8.0, 5.5], "target": [0, 0, 0]},
        "objects": [
            {
                "name": "Sun",
                "kind": "sphere",
                "location": [0, 0, 0],
                "scale": [1.3, 1.3, 1.3],
                "color": [1.0, 0.75, 0.20, 1.0],
                "motion": {"type": "rotation", "axis": "z", "speed": 20.0},
            },
            {
                "name": "Planet",
                "kind": "sphere",
                "location": [3.0, 0, 0],
                "scale": [0.55, 0.55, 0.55],
                "color": [0.25, 0.75, 1.0, 1.0],
                "motion": {
                    "type": "circular",
                    "center": [0, 0, 0],
                    "radius": 3.0,
                    "speed": 90.0,
                    "plane": "xy",
                    "face_tangent": True,
                },
            },
        ],
    }


def template_repeated_movement() -> Dict[str, Any]:
    return {
        "name": "Repeated Movement",
        "duration": 4.0,
        "fps": 24,
        "camera": {"type": "static", "location": [5.0, -5.0, 3.5], "target": [0, 0, 0.8]},
        "objects": [
            {
                "name": "Bouncer",
                "kind": "cube",
                "location": [0, 0, 1.0],
                "scale": [0.8, 0.8, 0.8],
                "color": [0.95, 0.35, 0.35, 1.0],
                "motion": {
                    "type": "repeated",
                    "channel": "location",
                    "axis": "z",
                    "amplitude": 0.8,
                    "frequency": 1.2,
                    "waveform": "abs",
                },
            },
            {
                "name": "Waver",
                "kind": "cylinder",
                "location": [2.0, 0, 0.8],
                "scale": [0.35, 0.35, 1.2],
                "color": [0.45, 0.85, 0.95, 1.0],
                "motion": {
                    "type": "repeated",
                    "channel": "rotation",
                    "axis": "y",
                    "amplitude": 25.0,
                    "frequency": 0.8,
                    "waveform": "sin",
                },
            },
        ],
    }


def template_mechanical_gears() -> Dict[str, Any]:
    return {
        "name": "Mechanical Gear Pair",
        "duration": 4.0,
        "fps": 24,
        "camera": {"type": "orbit", "radius": 6.0, "height": 3.0, "speed": 20.0, "target": [0.5, 0, 0]},
        "objects": [
            {
                "name": "Gear_A",
                "kind": "gear",
                "location": [0, 0, 0],
                "scale": [2.0, 2.0, 0.45],
                "color": [0.88, 0.58, 0.22, 1.0],
                "motion": {"type": "mechanical_rotation", "axis": "z", "rpm": 15.0, "ratio": 1.0, "radius": 1.0},
            },
            {
                "name": "Gear_B",
                "kind": "gear",
                "location": [1.55, 0, 0],
                "scale": [0.9, 0.9, 0.4],
                "color": [0.75, 0.80, 0.85, 1.0],
                "motion": {"type": "mechanical_rotation", "axis": "z", "rpm": 15.0, "ratio": -2.0, "radius": 0.55},
            },
        ],
    }


def template_crank_slider() -> Dict[str, Any]:
    return {
        "name": "Crank Slider Mechanism",
        "duration": 4.0,
        "fps": 24,
        "camera": {"type": "static", "location": [0.0, -7.0, 4.0], "target": [0.0, 0.0, 0.5]},
        "objects": [
            {
                "name": "Crank",
                "kind": "crank",
                "location": [-2.0, 0, 0.5],
                "scale": [0.8, 0.2, 0.2],
                "color": [0.95, 0.40, 0.25, 1.0],
                "motion": {"type": "mechanical_rotation", "axis": "z", "rpm": 24.0, "ratio": 1.0, "radius": 0.8},
            },
            {
                "name": "Slider",
                "kind": "slider",
                "location": [1.2, 0, 0.5],
                "scale": [0.7, 0.5, 0.5],
                "color": [0.35, 0.75, 0.95, 1.0],
                "motion": {
                    "type": "crank_slider",
                    "axis": "x",
                    "rpm": 24.0,
                    "radius": 0.8,
                    "rod_length": 2.4,
                    "base": [1.2, 0, 0.5],
                },
            },
        ],
    }


def template_basic_interaction() -> Dict[str, Any]:
    return {
        "name": "Basic Object Interaction",
        "duration": 4.0,
        "fps": 24,
        "camera": {"type": "orbit", "radius": 6.5, "height": 3.0, "speed": 25.0},
        "objects": [
            {
                "name": "Attractor",
                "kind": "sphere",
                "location": [0, 0, 0.5],
                "scale": [0.9, 0.9, 0.9],
                "color": [1.0, 0.65, 0.20, 1.0],
                "motion": {"type": "circular", "center": [0, 0, 0.5], "radius": 1.8, "speed": 70.0, "plane": "xy"},
            },
            {
                "name": "Responder",
                "kind": "cube",
                "location": [3.0, 0, 0.5],
                "scale": [0.7, 0.7, 0.7],
                "color": [0.55, 0.35, 0.95, 1.0],
                "motion": {
                    "type": "interaction",
                    "target": "Attractor",
                    "mode": "attract",
                    "radius": 2.2,
                    "strength": 0.45,
                    "scale_effect": 0.25,
                    "rotation_effect": 0.4,
                },
            },
        ],
    }


def template_full_demo() -> Dict[str, Any]:
    return {
        "name": "Full Procedural Demo",
        "duration": 3.0,
        "fps": 24,
        "interpolation": "BEZIER",
        "keyframe_step": 1,
        "events": [
            {"name": "Open", "frame": 1},
            {"name": "Cycle", "frame": 49},
            {"name": "Hold", "frame": 73},
        ],
        "camera": {
            "type": "orbit",
            "radius": 7.8,
            "height": 3.3,
            "target": [0.0, 0.0, 0.45],
            "speed": 38.0,
            "lens": 35.0,
        },
        "objects": [
            {
                "name": "CentralGear",
                "kind": "gear",
                "location": [0, 0, 0],
                "scale": [2.0, 2.0, 0.45],
                "color": [0.92, 0.56, 0.16, 1.0],
                "motion": {"type": "mechanical_rotation", "axis": "z", "rpm": 18.0, "ratio": 1.0, "radius": 1.0},
            },
            {
                "name": "Pinion",
                "kind": "gear",
                "location": [1.55, 0, 0],
                "scale": [0.9, 0.9, 0.4],
                "color": [0.76, 0.82, 0.88, 1.0],
                "motion": {"type": "mechanical_rotation", "axis": "z", "rpm": 18.0, "ratio": -2.0, "radius": 0.55},
            },
            {
                "name": "OrbitSphere",
                "kind": "sphere",
                "location": [3.0, 0, 0.25],
                "scale": [0.65, 0.65, 0.65],
                "color": [0.20, 0.72, 1.00, 1.0],
                "motion": {
                    "type": "circular",
                    "center": [0, 0, 0.25],
                    "radius": 3.0,
                    "speed": 85.0,
                    "plane": "xy",
                    "face_tangent": True,
                },
            },
            {
                "name": "TranslatingCube",
                "kind": "cube",
                "location": [-3.2, -2.2, 0.45],
                "scale": [0.75, 0.75, 0.75],
                "color": [0.96, 0.26, 0.45, 1.0],
                "motion": {
                    "type": "translation",
                    "from": [-3.2, -2.2, 0.45],
                    "to": [3.2, 2.2, 0.85],
                    "easing": "ease_in_out",
                },
            },
            {
                "name": "PulsingTorus",
                "kind": "torus",
                "location": [0, -3.0, 0.35],
                "scale": [1.0, 1.0, 1.0],
                "color": [0.55, 0.95, 0.45, 1.0],
                "motion": {
                    "type": "scaling",
                    "from_scale": [0.72, 0.72, 0.72],
                    "to_scale": [1.28, 1.28, 1.28],
                    "easing": "sine",
                    "pulse_amplitude": 0.12,
                    "pulse_frequency": 1.2,
                },
            },
            {
                "name": "SliderBlock",
                "kind": "slider",
                "location": [2.4, 2.4, 0.30],
                "scale": [0.55, 0.90, 0.55],
                "color": [0.78, 0.62, 1.00, 1.0],
                "motion": {
                    "type": "crank_slider",
                    "axis": "x",
                    "rpm": 24.0,
                    "radius": 0.8,
                    "rod_length": 2.2,
                    "base": [2.4, 2.4, 0.30],
                },
            },
        ],
    }


SAMPLE_PROJECTS: Dict[str, Dict[str, Any]] = {
    "Full Demo": template_full_demo,
    "Rotating Object": template_rotating_object,
    "Translation + Scaling": template_translation_scaling,
    "Camera Movement": template_camera_movement,
    "Circular Motion": template_circular_motion,
    "Repeated Movement": template_repeated_movement,
    "Mechanical Gear Pair": template_mechanical_gears,
    "Crank Slider": template_crank_slider,
    "Basic Interaction": template_basic_interaction,
}


# ======================================================================
# 8. Preview and rendering engine
# ======================================================================
@dataclass
class RenderResult:
    video_bytes: Optional[bytes] = None
    video_mime: str = "video/mp4"
    video_filename: str = "motion_demo.mp4"
    blend_bytes: Optional[bytes] = None
    log: str = ""
    used_blender: bool = False
    render_seconds: float = 0.0


def find_blender() -> str:
    for name in ("blender", "blender.exe"):
        found = shutil.which(name)
        if found:
            return found

    candidates: List[Path] = [
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path.home() / "snap" / "blender" / "current" / "blender",
        Path("C:/Program Files/Blender Foundation"),
        Path("C:/Program Files (x86)/Blender Foundation"),
        Path.home() / "AppData/Local/Programs/Blender Foundation",
        Path.home() / "AppData/Local/Microsoft/WindowsApps/blender.exe",
    ]

    for p in candidates:
        try:
            if p.is_file() and "blender" in p.name.lower():
                return str(p)
            if p.is_dir():
                for pattern in ("blender.exe", "blender", "*/blender.exe", "*/blender"):
                    for exe in p.glob(pattern):
                        if exe.is_file() and "blender" in exe.name.lower():
                            return str(exe)
        except Exception:
            pass
    return ""


def _resolve_blender_path(path: str) -> str:
    path = str(path or "").strip()
    if not path:
        return find_blender()
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for pattern in ("blender.exe", "blender", "*/blender.exe", "*/blender"):
            for exe in Path(path).glob(pattern):
                if exe.is_file() and "blender" in exe.name.lower():
                    return str(exe)
    return ""


BLENDER_SCRIPT = r'''
import sys
import os
import json
import math
import argparse
import shutil
import glob

import bpy
from mathutils import Vector, Euler, Quaternion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blend-output", default="")
    parser.add_argument("--engine", default="BLENDER_WORKBENCH")
    parser.add_argument("--resolution-x", type=int, default=480)
    parser.add_argument("--resolution-y", type=int, default=270)
    parser.add_argument("--fps", type=int, default=24)
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    args, _ = parser.parse_known_args(argv)
    return args


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def set_material(obj, color):
    if getattr(obj, "type", "") == "EMPTY":
        return
    rgba = list(color) + [1.0] * max(0, 4 - len(color))
    rgba = [float(rgba[i]) for i in range(4)]
    mat = bpy.data.materials.new(name=obj.name + "_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = rgba
    if hasattr(obj.data, "materials"):
        obj.data.materials.append(mat)
    try:
        obj.color = rgba
    except Exception:
        pass


def create_primitive(kind, name, location):
    loc = Vector(location)
    if kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=16, radius=0.5, location=loc)
    elif kind in {"cylinder", "gear", "crank"}:
        verts = 28 if kind == "gear" else 24
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=0.5, depth=1.0, location=loc)
    elif kind == "torus":
        bpy.ops.mesh.primitive_torus_add(major_radius=0.65, minor_radius=0.18, location=loc)
    elif kind == "cone":
        bpy.ops.mesh.primitive_cone_add(vertices=28, radius1=0.5, depth=1.0, location=loc)
    elif kind == "plane":
        bpy.ops.mesh.primitive_plane_add(size=2.0, location=loc)
    elif kind == "empty":
        obj = bpy.data.objects.new(name, None)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = loc
        obj.empty_display_type = "PLAIN_AXES"
        obj.empty_display_size = 0.7
        return obj
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)

    obj = bpy.context.active_object
    obj.name = name
    try:
        if kind in {"sphere", "cylinder", "gear", "crank", "torus", "cone"}:
            bpy.ops.object.shade_smooth()
    except Exception:
        pass
    return obj


def import_model(path, import_as=None):
    if not path or not os.path.exists(path):
        return
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".obj":
            try:
                bpy.ops.wm.obj_import(filepath=path)
            except AttributeError:
                bpy.ops.import_scene.obj(filepath=path)
        elif ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=path)
        elif ext in {".glb", ".gltf"}:
            bpy.ops.import_scene.gltf(filepath=path)
        elif ext == ".blend":
            with bpy.data.libraries.load(path, link=False) as (data_from, data_to):
                data_to.objects = data_from.objects
            for ob in data_to.objects:
                if ob is not None:
                    bpy.context.scene.collection.objects.link(ob)
    except Exception as exc:
        print("IMPORT_WARNING:", exc)

    if import_as:
        selected = list(getattr(bpy.context, "selected_objects", []))
        active = getattr(bpy.context, "active_object", None)
        target = active if active is not None else (selected[0] if selected else None)
        if target is not None:
            target.name = import_as


def set_render_engine(scene, engine_name):
    tried = [engine_name, "BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"]
    for eng in tried:
        try:
            scene.render.engine = eng
            break
        except Exception:
            continue
    try:
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = 16
            scene.cycles.use_denoising = False
    except Exception:
        pass
    try:
        if "EEVEE" in scene.render.engine:
            scene.eevee.taa_render_samples = 16
    except Exception:
        pass
    try:
        if scene.render.engine == "BLENDER_WORKBENCH":
            scene.display.shading.light = "STUDIO"
            scene.display.shading.color_type = "MATERIAL"
    except Exception:
        pass


def set_world(scene):
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.035, 0.045, 0.065, 1.0)
        bg.inputs[1].default_value = 1.0


def add_sun(scene):
    has_light = any(getattr(ob, "type", "") == "LIGHT" for ob in scene.objects)
    if has_light:
        return
    light_data = bpy.data.lights.new("Sun", type="SUN")
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new("Sun", light_data)
    scene.collection.objects.link(light_obj)
    light_obj.rotation_euler = Euler((math.radians(52.0), math.radians(28.0), math.radians(15.0)), "XYZ")


def insert_interpolation(scene, interp):
    valid = {"BEZIER", "LINEAR", "CONSTANT", "SINE", "QUAD", "CUBIC", "QUART", "QUINT", "EXPO", "CIRC"}
    interp = str(interp).upper()
    if interp not in valid:
        interp = "BEZIER"
    for action in bpy.data.actions:
        for fc in action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = interp
                if interp == "BEZIER":
                    try:
                        kp.handle_left_type = "AUTO_CLAMPED"
                        kp.handle_right_type = "AUTO_CLAMPED"
                    except Exception:
                        pass


def finalize_movie_output(output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    base = os.path.splitext(output_path)[0]
    candidates = glob.glob(base + "*.mp4") + glob.glob(base + "*")
    for cand in candidates:
        if os.path.isfile(cand) and os.path.getsize(cand) > 0 and cand != output_path:
            try:
                shutil.copy(cand, output_path)
                return True
            except Exception:
                pass
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def main():
    args = parse_args()
    with open(args.payload, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    spec = payload.get("spec", {})
    channels = payload.get("channels", {})
    timeline = payload.get("timeline", {})

    clear_scene()

    scene = bpy.context.scene
    scene.frame_start = int(timeline.get("frame_start", 1))
    scene.frame_end = int(timeline.get("frame_end", scene.frame_start + 1))
    scene.render.fps = int(timeline.get("fps", args.fps))
    scene.render.resolution_x = int(args.resolution_x)
    scene.render.resolution_y = int(args.resolution_y)
    scene.render.resolution_percentage = 100

    try:
        scene.render.use_overwrite = True
    except Exception:
        pass
    try:
        scene.render.use_place_holder = False
    except Exception:
        pass

    set_render_engine(scene, args.engine)
    set_world(scene)

    cam_data = bpy.data.cameras.new("MotionCamera")
    cam_data.lens = float(spec.get("camera", {}).get("lens", 35.0))
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    add_sun(scene)

    import_model(spec.get("import_model"), spec.get("import_as"))

    created = {}
    for obj_spec in spec.get("objects", []):
        name = obj_spec.get("name", "Object")
        kind = obj_spec.get("kind", "cube")
        loc = obj_spec.get("location", [0, 0, 0])
        obj = create_primitive(kind, name, loc)
        obj.scale = Vector(obj_spec.get("scale", [1, 1, 1]))
        obj.rotation_mode = "XYZ"
        obj.rotation_euler = Euler(obj_spec.get("rotation", [0, 0, 0]), "XYZ")
        set_material(obj, obj_spec.get("color", [0.85, 0.87, 0.90, 1.0]))
        obj["motion_spec_json"] = json.dumps(obj_spec)
        created[name] = obj

    for name, keys in channels.get("objects", {}).items():
        obj = created.get(name) or bpy.data.objects.get(name)
        if obj is None:
            obj = create_primitive("cube", name, [0, 0, 0])
            set_material(obj, [0.85, 0.87, 0.90, 1.0])
        for key in keys:
            frame = int(key["frame"])
            obj.location = Vector(key.get("location", [0, 0, 0]))
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = Euler(key.get("rotation_euler", [0, 0, 0]), "XYZ")
            obj.scale = Vector(key.get("scale", [1, 1, 1]))
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

    for key in channels.get("camera", []):
        frame = int(key["frame"])
        cam_obj.location = Vector(key.get("location", [6, -6, 4]))
        cam_obj.rotation_mode = "QUATERNION"
        cam_obj.rotation_quaternion = Quaternion(key.get("rotation_quaternion", [1, 0, 0, 0]))
        cam_data.lens = float(key.get("lens", 35.0))
        cam_obj.keyframe_insert(data_path="location", frame=frame)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        cam_data.keyframe_insert(data_path="lens", frame=frame)

    insert_interpolation(scene, spec.get("interpolation", "BEZIER"))

    for ev in spec.get("events", []):
        try:
            scene.timeline_markers.new(str(ev.get("name", "Event")), frame=int(ev.get("frame", scene.frame_start)))
        except Exception:
            pass

    scene["motion_spec_json"] = json.dumps(spec)
    scene["motion_channels_json"] = json.dumps(channels)

    txt = bpy.data.curves.new("MotionSpecText", type="TEXT")
    txt.body = json.dumps(payload, indent=2)
    txt_obj = bpy.data.objects.new("MotionSpecText", txt)
    scene.collection.objects.link(txt_obj)
    txt_obj.location = (0, 0, -3)
    try:
        txt_obj.hide_viewport = True
        txt_obj.hide_render = True
    except Exception:
        pass

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    if args.blend_output:
        blend_dir = os.path.dirname(os.path.abspath(args.blend_output))
        os.makedirs(blend_dir, exist_ok=True)
        try:
            bpy.ops.wm.save_as_mainfile(filepath=args.blend_output)
        except Exception as exc:
            print("BLEND_SAVE_WARNING:", exc)

    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    try:
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    except Exception:
        pass
    try:
        scene.render.ffmpeg.audio_codec = "NONE"
    except Exception:
        pass
    scene.render.filepath = args.output

    bpy.ops.render.render(animation=True)

    ok = finalize_movie_output(args.output)
    if not ok:
        raise RuntimeError("Blender did not produce the expected movie file.")
    print("BLENDER_RENDER_OK")


if __name__ == "__main__":
    main()
'''


def _payload_tmp_files(spec: Dict[str, Any], channels: Dict[str, Any], timeline: Timeline) -> Tuple[str, str, str, str, str]:
    tmpdir = tempfile.mkdtemp(prefix="motion_design_")
    payload_path = os.path.join(tmpdir, "payload.json")
    script_path = os.path.join(tmpdir, "blender_motion_script.py")
    output_mp4 = os.path.join(tmpdir, "motion_demo.mp4")
    output_blend = os.path.join(tmpdir, "motion_demo.blend")
    with open(payload_path, "w", encoding="utf-8") as fh:
        json.dump({"spec": spec, "channels": channels, "timeline": asdict(timeline)}, fh)
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(BLENDER_SCRIPT)
    return tmpdir, payload_path, script_path, output_mp4, output_blend


def _render_with_blender(spec: Dict[str, Any], channels: Dict[str, Any], timeline: Timeline, settings: Dict[str, Any]) -> RenderResult:
    blender_path = _resolve_blender_path(str(settings.get("blender_path") or ""))
    if not blender_path or not os.path.isfile(blender_path):
        raise RuntimeError("Blender executable was not found. Set the Blender path in the sidebar or install Blender.")

    tmpdir, payload_path, script_path, output_mp4, output_blend = _payload_tmp_files(spec, channels, timeline)
    cmd = [
        blender_path,
        "--background",
        "--factory-startup",
        "--python",
        script_path,
        "--",
        "--payload",
        payload_path,
        "--output",
        output_mp4,
        "--blend-output",
        output_blend,
        "--engine",
        str(settings.get("engine", "BLENDER_WORKBENCH")),
        "--resolution-x",
        str(int(settings.get("width", 320))),
        "--resolution-y",
        str(int(settings.get("height", 180))),
        "--fps",
        str(int(spec.get("fps", 24))),
    ]

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=float(settings.get("timeout", 180)),
        )
        seconds = time.time() - start
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode != 0 or not os.path.exists(output_mp4) or os.path.getsize(output_mp4) == 0:
            raise RuntimeError(f"Blender render failed. Exit={proc.returncode}\n{log[-5000:]}")
        with open(output_mp4, "rb") as fh:
            mp4 = fh.read()
        blend = None
        if os.path.exists(output_blend):
            with open(output_blend, "rb") as fh:
                blend = fh.read()
        return RenderResult(
            video_bytes=mp4,
            video_mime="video/mp4",
            video_filename="motion_demo.mp4",
            blend_bytes=blend,
            log=log[-12000:],
            used_blender=True,
            render_seconds=seconds,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ----------------------------------------------------------------------
# Pure-Python GIF encoder fallback
# ----------------------------------------------------------------------
def _rgb_to_gif_indices(frame: np.ndarray) -> bytes:
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    r = np.clip((arr[..., 0].astype(np.int16) + 25) // 51, 0, 5)
    g = np.clip((arr[..., 1].astype(np.int16) + 25) // 51, 0, 5)
    b = np.clip((arr[..., 2].astype(np.int16) + 25) // 51, 0, 5)
    idx = (r * 36 + g * 6 + b).astype(np.uint8)
    return idx.tobytes()


def _lzw_encode_gif(data: bytes, min_code_size: int = 8) -> bytes:
    data = bytes(data)
    clear_code = 1 << min_code_size
    eoi_code = clear_code + 1
    next_code = eoi_code + 1
    code_size = min_code_size + 1
    max_code = (1 << code_size) - 1
    dictionary = {bytes([i]): i for i in range(clear_code)}

    out = bytearray()
    bit_buf = 0
    bit_count = 0

    def emit(code: int) -> None:
        nonlocal bit_buf, bit_count
        bit_buf |= int(code) << bit_count
        bit_count += code_size
        while bit_count >= 8:
            out.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8

    emit(clear_code)

    if not data:
        emit(eoi_code)
        while bit_count > 0:
            out.append(bit_buf & 0xFF)
            bit_buf >>= 8
            bit_count -= 8
        return bytes(out)

    old = bytes(data[0:1])

    for i in range(1, len(data)):
        b = bytes(data[i:i + 1])
        curr = old + b
        if curr in dictionary:
            old = curr
        else:
            emit(dictionary[old])

            if next_code > 4095:
                emit(clear_code)
                dictionary = {bytes([j]): j for j in range(clear_code)}
                next_code = eoi_code + 1
                code_size = min_code_size + 1
                max_code = (1 << code_size) - 1
            else:
                dictionary[curr] = next_code
                next_code += 1
                if next_code > max_code and code_size < 12:
                    code_size += 1
                    max_code = (1 << code_size) - 1

            old = b

    emit(dictionary[old])
    emit(eoi_code)

    while bit_count > 0:
        out.append(bit_buf & 0xFF)
        bit_buf >>= 8
        bit_count -= 8

    return bytes(out)


def _encode_gif(frames: List[np.ndarray], fps: float, loop: int = 0) -> bytes:
    if not frames:
        raise RuntimeError("No frames supplied to GIF encoder.")

    first = np.asarray(frames[0], dtype=np.uint8)
    if first.ndim == 2:
        first = np.stack([first, first, first], axis=-1)
    if first.shape[-1] == 4:
        first = first[..., :3]
    height, width = first.shape[:2]

    buf = bytearray()
    buf.extend(b"GIF89a")
    buf.extend(struct.pack("<HH", int(width), int(height)))
    buf.append(0xF7)  # global color table, 256 colors
    buf.append(0)
    buf.append(0)

    for i in range(256):
        if i < 216:
            b = i % 6
            g = (i // 6) % 6
            r = i // 36
            buf.extend(bytes((r * 51, g * 51, b * 51)))
        else:
            v = ((i - 216) * 255) // 39
            buf.extend(bytes((v, v, v)))

    # Netscape looping extension
    buf.extend(b"\x21\xFF\x0BNETSCAPE2.0\x03\x01")
    buf.extend(struct.pack("<H", int(loop)))
    buf.append(0)

    delay = max(1, int(round(100.0 / max(1.0, float(fps)))))

    for frame in frames:
        arr = np.asarray(frame, dtype=np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[0] != height or arr.shape[1] != width:
            arr = _resize_nearest(arr, width, height)

        indices = _rgb_to_gif_indices(arr)

        # Graphic Control Extension
        buf.extend(b"\x21\xF9\x04")
        buf.append(0x00)
        buf.extend(struct.pack("<H", delay))
        buf.append(0x00)

        # Image Descriptor
        buf.append(0x2C)
        buf.extend(struct.pack("<HHHH", 0, 0, int(width), int(height)))
        buf.append(0x00)
        buf.append(8)

        compressed = _lzw_encode_gif(indices, 8)
        for i in range(0, len(compressed), 255):
            chunk = compressed[i:i + 255]
            buf.append(len(chunk))
            buf.extend(chunk)
        buf.append(0)

    buf.append(0x3B)
    return bytes(buf)


# ----------------------------------------------------------------------
# Fallback CPU simulator/renderer so the app never shows a blank dashboard
# ----------------------------------------------------------------------
CUBE_VERTICES = np.array(
    [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ],
    dtype=float,
)

CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _euler_to_matrix(e: List[float]) -> np.ndarray:
    rx, ry, rz = [float(v) for v in e]
    cr, sr = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz_m @ ry_m @ rx_m


def _quat_rotate(q: List[float], v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    qv = np.array([x, y, z], dtype=float)
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _quat_inverse(q: List[float]) -> List[float]:
    return [float(q[0]), float(-q[1]), float(-q[2]), float(-q[3])]


def _quat_slerp(q0: List[float], q1: List[float], t: float) -> List[float]:
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = float(sum(a * b for a, b in zip(q0, q1)))
    if dot < 0.0:
        q1 = [-v for v in q1]
        dot = -dot
    if dot > 0.9995:
        out = [q0[i] + t * (q1[i] - q0[i]) for i in range(4)]
        return _normalize_quat(out)
    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    out = [s0 * q0[i] + s1 * q1[i] for i in range(4)]
    return _normalize_quat(out)


def _interp_list(a: List[float], b: List[float], t: float) -> List[float]:
    return [float(a[i] + (b[i] - a[i]) * t) for i in range(max(len(a), len(b)))]


def _channel_at(keys: List[Dict[str, Any]], frame: int) -> Optional[Dict[str, Any]]:
    if not keys:
        return None
    if frame <= keys[0]["frame"]:
        return copy.deepcopy(keys[0])
    if frame >= keys[-1]["frame"]:
        return copy.deepcopy(keys[-1])
    for i in range(len(keys) - 1):
        k0, k1 = keys[i], keys[i + 1]
        if k0["frame"] <= frame <= k1["frame"]:
            span = max(1, k1["frame"] - k0["frame"])
            t = float(frame - k0["frame"]) / float(span)
            out = {"frame": frame}
            for field in ("location", "rotation_euler", "scale"):
                if field in k0 and field in k1:
                    out[field] = _interp_list(k0[field], k1[field], t)
            if "rotation_quaternion" in k0 and "rotation_quaternion" in k1:
                out["rotation_quaternion"] = _quat_slerp(k0["rotation_quaternion"], k1["rotation_quaternion"], t)
            if "lens" in k0 and "lens" in k1:
                out["lens"] = float(k0["lens"] + (k1["lens"] - k0["lens"]) * t)
            return out
    return copy.deepcopy(keys[-1])


def _object_color(spec: Dict[str, Any], name: str) -> List[float]:
    for obj in spec.get("objects", []):
        if obj.get("name") == name:
            return _rgba(obj.get("color"), [0.85, 0.87, 0.90, 1.0])
    return [0.85, 0.87, 0.90, 1.0]


def _transform_vertices(tr: Dict[str, Any]) -> np.ndarray:
    scale = np.asarray(tr.get("scale", [1, 1, 1]), dtype=float)
    rot = np.asarray(tr.get("rotation_euler", [0, 0, 0]), dtype=float)
    loc = np.asarray(tr.get("location", [0, 0, 0]), dtype=float)
    v = CUBE_VERTICES * scale
    m = _euler_to_matrix(rot.tolist())
    return (m @ v.T).T + loc


def _world_to_camera(point: np.ndarray, cam: Dict[str, Any]) -> np.ndarray:
    eye = np.asarray(cam.get("location", [6, -6, 4]), dtype=float)
    q = cam.get("rotation_quaternion", [1, 0, 0, 0])
    return _quat_rotate(_quat_inverse(q), point - eye)


_CURRENT_CAMERA: List[Dict[str, Any]] = [{}]


def _project_point(point: np.ndarray, width: int, height: int, fov: float = 50.0) -> Optional[Tuple[float, float]]:
    p = _world_to_camera(point, _CURRENT_CAMERA[0])
    depth = -float(p[2])
    if depth <= 0.02:
        return None
    focal = 0.5 * width / math.tan(math.radians(fov) / 2.0)
    sx = width / 2.0 + focal * float(p[0]) / depth
    sy = height / 2.0 - focal * float(p[1]) / depth
    return sx, sy


def _resize_nearest(img: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = img.shape[:2]
    if src_w == width and src_h == height:
        return img
    x = np.linspace(0, src_w - 1, max(1, width)).astype(int)
    y = np.linspace(0, src_h - 1, max(1, height)).astype(int)
    return img[np.ix_(y, x)]


def _draw_frame(ax, spec: Dict[str, Any], channels: Dict[str, Any], frame: int, width: int, height: int) -> None:
    ax.clear()
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.set_facecolor("#0b1020")

    cam = _channel_at(channels.get("camera", []), frame)
    if cam is None:
        cam = {
            "location": [6.0, -6.0, 4.0],
            "rotation_quaternion": _look_at_quaternion(np.array([6.0, -6.0, 4.0]), np.array([0.0, 0.0, 0.0])),
            "lens": 35.0,
        }
    _CURRENT_CAMERA[0] = cam

    for i in range(-5, 6):
        p1 = _project_point(np.array([float(i), -5.0, 0.0]), width, height)
        p2 = _project_point(np.array([float(i), 5.0, 0.0]), width, height)
        if p1 and p2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#243044", linewidth=0.6, zorder=1)
        p1 = _project_point(np.array([-5.0, float(i), 0.0]), width, height)
        p2 = _project_point(np.array([5.0, float(i), 0.0]), width, height)
        if p1 and p2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#243044", linewidth=0.6, zorder=1)

    for name, keys in channels.get("objects", {}).items():
        tr = _channel_at(keys, frame)
        if tr is None:
            continue
        verts = _transform_vertices(tr)
        color = _object_color(spec, name)[:3]
        for a, b in CUBE_EDGES:
            pa = _project_point(verts[a], width, height)
            pb = _project_point(verts[b], width, height)
            if pa and pb:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, linewidth=1.6, zorder=5)

    ax.text(8, height - 12, f"frame {frame}", color="#d8e4ff", fontsize=8, family="monospace")
    ax.text(width - 8, height - 12, "procedural motion", color="#7890b8", fontsize=8, ha="right", family="monospace")


def _sample_frames(timeline: Timeline, max_frames: int) -> Tuple[List[int], int]:
    all_frames = list(timeline.frames)
    if not all_frames:
        return [], max(1, timeline.fps)
    if len(all_frames) <= max_frames:
        return all_frames, max(1, timeline.fps)

    step = max(1, math.ceil(len(all_frames) / max(2, max_frames)))
    sampled = all_frames[::step]
    if sampled[-1] != all_frames[-1]:
        sampled.append(all_frames[-1])

    effective_fps = max(1, int(round(len(sampled) / max(0.04, timeline.duration))))
    return sampled, effective_fps


def _mp4_dimensions(settings: Dict[str, Any]) -> Tuple[int, int]:
    width = max(64, int(settings.get("width", 320)))
    height = max(36, int(settings.get("height", 180)))
    scale = min(1.0, 480.0 / float(width), 270.0 / float(height))
    width = max(64, int(width * scale))
    height = max(36, int(height * scale))
    width = (width // 2) * 2
    height = (height // 2) * 2
    return width, height


def _gif_dimensions(settings: Dict[str, Any]) -> Tuple[int, int]:
    width = max(64, int(settings.get("width", 320)))
    height = max(36, int(settings.get("height", 180)))
    scale = min(1.0, 240.0 / float(width), 160.0 / float(height))
    width = max(64, int(width * scale))
    height = max(36, int(height * scale))
    return width, height


def _render_matplotlib_frames(
    spec: Dict[str, Any],
    channels: Dict[str, Any],
    frames: List[int],
    width: int,
    height: int,
) -> List[np.ndarray]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    out: List[np.ndarray] = []
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#0b1020")

    try:
        for frame in frames:
            _draw_frame(ax, spec, channels, int(frame), width, height)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            rgb = rgba[:, :, :3].copy()
            rgb = _resize_nearest(rgb, width, height)
            out.append(np.asarray(rgb, dtype=np.uint8))
    finally:
        plt.close(fig)

    return out


def _write_mp4_imageio(frames: List[np.ndarray], fps: int) -> bytes:
    if not frames:
        raise RuntimeError("No frames to write MP4.")

    import imageio.v2 as imageio

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        writer = imageio.get_writer(
            tmp_path,
            fps=max(1, int(fps)),
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
        )
        try:
            for arr in frames:
                writer.append_data(np.asarray(arr, dtype=np.uint8))
        finally:
            writer.close()

        with open(tmp_path, "rb") as fh:
            data = fh.read()
        if not data:
            raise RuntimeError("MP4 writer produced an empty file.")
        return data
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _draw_filled_rect(img: np.ndarray, cx: int, cy: int, size: int, color: np.ndarray) -> None:
    h, w = img.shape[:2]
    half = max(1, size // 2)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)


def _render_synthetic_gif(spec: Dict[str, Any], channels: Dict[str, Any], timeline: Timeline) -> Tuple[bytes, int, int, int]:
    frames, fps = _sample_frames(timeline, max_frames=36)
    width, height = 200, 112

    image_frames: List[np.ndarray] = []
    for frame in frames:
        img = np.full((height, width, 3), [11, 16, 32], dtype=np.uint8)
        img[int(height * 0.75):int(height * 0.75) + 1, :] = [36, 48, 68]

        cam = _channel_at(channels.get("camera", []), frame)
        if cam is None:
            cam = {
                "location": [6.0, -6.0, 4.0],
                "rotation_quaternion": _look_at_quaternion(np.array([6.0, -6.0, 4.0]), np.array([0.0, 0.0, 0.0])),
                "lens": 35.0,
            }
        _CURRENT_CAMERA[0] = cam

        items: List[Tuple[float, Tuple[float, float], Dict[str, Any], str]] = []
        for name, keys in channels.get("objects", {}).items():
            tr = _channel_at(keys, frame)
            if tr is None:
                continue
            loc = np.asarray(tr.get("location", [0, 0, 0]), dtype=float)
            p = _project_point(loc, width, height)
            if p is None:
                continue
            pc = _world_to_camera(loc, cam)
            depth = -float(pc[2])
            if depth <= 0.02:
                continue
            items.append((depth, p, tr, name))

        if not items:
            t = timeline.time_at_frame(frame)
            cx = int(width / 2 + 35 * math.sin(t * 2.0))
            cy = int(height / 2 + 18 * math.cos(t * 3.0))
            _draw_filled_rect(img, cx, cy, 18, np.array([255, 140, 40], dtype=np.uint8))
        else:
            items.sort(key=lambda x: -x[0])
            focal = 0.5 * width / math.tan(math.radians(50.0) / 2.0)
            for depth, p, tr, name in items:
                color = np.asarray([int(255 * c) for c in _object_color(spec, name)[:3]], dtype=np.uint8)
                scale_avg = float(np.mean(np.abs(np.asarray(tr.get("scale", [1, 1, 1]), dtype=float))))
                size = int(max(4, min(width // 2, focal * scale_avg / max(0.1, depth) * 0.35)))
                _draw_filled_rect(img, int(p[0]), int(p[1]), size, color)

        image_frames.append(img)

    gif = _encode_gif(image_frames, fps=fps, loop=0)
    return gif, fps, width, height


def _render_fallback(spec: Dict[str, Any], channels: Dict[str, Any], timeline: Timeline, settings: Dict[str, Any]) -> RenderResult:
    start = time.time()
    logs: List[str] = []

    # 1) Try MP4 through imageio if available.
    try:
        frames, fps = _sample_frames(timeline, max_frames=90)
        width, height = _mp4_dimensions(settings)
        mats = _render_matplotlib_frames(spec, channels, frames, width, height)
        mp4 = _write_mp4_imageio(mats, fps)
        logs.append(f"MP4 fallback rendered at {width}x{height}, {fps} fps, {len(frames)} frames.")
        return RenderResult(
            video_bytes=mp4,
            video_mime="video/mp4",
            video_filename="motion_demo.mp4",
            blend_bytes=None,
            log="\n".join(logs),
            used_blender=False,
            render_seconds=time.time() - start,
        )
    except Exception as exc:
        logs.append(f"MP4 fallback failed: {exc}\n{traceback.format_exc()[-3000:]}")

    # 2) Try animated GIF through matplotlib + pure Python GIF encoder.
    try:
        frames, fps = _sample_frames(timeline, max_frames=48)
        width, height = _gif_dimensions(settings)
        mats = _render_matplotlib_frames(spec, channels, frames, width, height)
        gif = _encode_gif(mats, fps=fps, loop=0)
        logs.append(f"GIF fallback rendered at {width}x{height}, {fps} fps, {len(frames)} frames.")
        return RenderResult(
            video_bytes=gif,
            video_mime="image/gif",
            video_filename="motion_demo.gif",
            blend_bytes=None,
            log="\n".join(logs),
            used_blender=False,
            render_seconds=time.time() - start,
        )
    except Exception as exc:
        logs.append(f"GIF matplotlib fallback failed: {exc}\n{traceback.format_exc()[-3000:]}")

    # 3) Last-resort synthetic GIF, no matplotlib required.
    try:
        gif, fps, width, height = _render_synthetic_gif(spec, channels, timeline)
        logs.append(f"Synthetic GIF fallback rendered at {width}x{height}, {fps} fps.")
        return RenderResult(
            video_bytes=gif,
            video_mime="image/gif",
            video_filename="motion_demo.gif",
            blend_bytes=None,
            log="\n".join(logs),
            used_blender=False,
            render_seconds=time.time() - start,
        )
    except Exception as exc:
        logs.append(f"Synthetic GIF fallback failed: {exc}\n{traceback.format_exc()[-3000:]}")

    return RenderResult(
        video_bytes=None,
        video_mime="video/mp4",
        video_filename="motion_demo.mp4",
        blend_bytes=None,
        log="\n".join(logs),
        used_blender=False,
        render_seconds=time.time() - start,
    )


def _render_single_frame_png(spec: Dict[str, Any], channels: Dict[str, Any], frame: int, width: int, height: int) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    width = max(64, int(width))
    height = max(64, int(height))
    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#0b1020")
    try:
        _draw_frame(ax, spec, channels, int(frame), width, height)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf.read()
    finally:
        plt.close(fig)


class PreviewRenderingEngine:
    def render(self, spec: Dict[str, Any], channels: Dict[str, Any], settings: Dict[str, Any]) -> RenderResult:
        timeline = Timeline.from_spec(spec)
        start = time.time()
        try:
            result = _render_with_blender(spec, channels, timeline, settings)
            result.render_seconds = time.time() - start
            return result
        except Exception as exc:
            blender_log = f"Blender render unavailable/failed: {exc}\n{traceback.format_exc()[-4000:]}"
            fallback = _render_fallback(spec, channels, timeline, settings)
            fallback.log = blender_log + "\n" + fallback.log
            fallback.render_seconds = time.time() - start
            return fallback


# ======================================================================
# 9. Evaluation metrics
# ======================================================================
def _analytic_transforms(spec: Dict[str, Any], t: float) -> Dict[str, Dict[str, Any]]:
    engine = ObjectMotionEngine(spec)
    transforms = engine.compute_base_all(t)
    engine.apply_interactions(transforms, t)
    return transforms


def evaluate_animation(
    spec: Dict[str, Any],
    channels: Dict[str, Any],
    result: RenderResult,
    timeline: Timeline,
) -> Dict[str, Any]:
    CS = _cubic_spline()
    accuracy_errors: List[float] = []
    continuity_values: List[float] = []

    for name, keys in channels.get("objects", {}).items():
        if len(keys) < 2:
            continue
        frames = [int(k["frame"]) for k in keys]
        locs = np.asarray([k["location"] for k in keys], dtype=float)

        for i in range(len(keys) - 1):
            f0, f1 = frames[i], frames[i + 1]
            if f1 == f0:
                continue
            dt = float(f1 - f0) / float(max(1, timeline.fps))
            p0 = np.asarray(keys[i]["location"], dtype=float)
            p1 = np.asarray(keys[i + 1]["location"], dtype=float)
            r0 = np.asarray(keys[i]["rotation_euler"], dtype=float)
            r1 = np.asarray(keys[i + 1]["rotation_euler"], dtype=float)
            pos_rate = float(np.linalg.norm(p1 - p0) / max(1e-6, dt))
            rot_rate = float(np.linalg.norm(r1 - r0) / max(1e-6, dt))
            continuity_values.extend([pos_rate, rot_rate])

        if CS is not None and len(keys) >= 4:
            try:
                spline = CS(frames, locs, axis=0, bc_type="natural")
                for i in range(len(keys) - 1):
                    mid_frame = 0.5 * (frames[i] + frames[i + 1])
                    mid_t = float(mid_frame - timeline.frame_start) / float(max(1, timeline.fps))
                    analytic = _analytic_transforms(spec, mid_t).get(name)
                    if analytic is None:
                        continue
                    approx = spline(np.asarray([mid_frame]))[0]
                    err = float(np.linalg.norm(np.asarray(analytic["location"], dtype=float) - approx))
                    accuracy_errors.append(err)
            except Exception:
                pass

    max_continuity = float(max(continuity_values)) if continuity_values else 0.0
    mean_continuity = float(np.mean(continuity_values)) if continuity_values else 0.0
    std_continuity = float(np.std(continuity_values)) if continuity_values else 0.0
    cv = std_continuity / mean_continuity if mean_continuity > 1e-6 else 0.0
    curve_stability = 1.0 / (1.0 + cv)

    frame_count = int(len(channels.get("camera", [])) or timeline.frame_count)
    render_fps = frame_count / result.render_seconds if result.render_seconds > 0 else 0.0
    fps_ratio = min(1.0, render_fps / float(max(1, spec.get("fps", 24))))
    playback_stability = float(np.clip(0.65 * curve_stability + 0.35 * fps_ratio, 0.0, 1.0))

    if result.used_blender:
        renderer = "Blender"
    elif result.video_mime.startswith("video/"):
        renderer = "Fallback MP4"
    elif result.video_bytes:
        renderer = "Fallback GIF"
    else:
        renderer = "Failed"

    return {
        "keyframe_accuracy_max_error": float(max(accuracy_errors)) if accuracy_errors else 0.0,
        "motion_continuity_max_units_per_s": max_continuity,
        "playback_stability_score": playback_stability,
        "rendering_time_s": float(result.render_seconds),
        "export_success": bool(result.video_bytes and len(result.video_bytes) > 0),
        "frames_generated": frame_count,
        "renderer": renderer,
    }


# ======================================================================
# 10. Export manager
# ======================================================================
class ExportManager:
    @staticmethod
    def spec_json(spec: Dict[str, Any]) -> str:
        return json.dumps(spec, indent=2, sort_keys=True)

    @staticmethod
    def channels_json(channels: Dict[str, Any]) -> str:
        return json.dumps(channels, indent=2, sort_keys=True)

    @staticmethod
    def blender_script() -> str:
        return BLENDER_SCRIPT

    @staticmethod
    def payload_json(spec: Dict[str, Any], channels: Dict[str, Any], timeline: Timeline) -> str:
        return json.dumps({"spec": spec, "channels": channels, "timeline": asdict(timeline)}, indent=2, sort_keys=True)


# ======================================================================
# Render pipeline
# ======================================================================
def _render_pipeline(spec_json: str, settings_json: str) -> Dict[str, Any]:
    spec_raw = json.loads(spec_json)
    settings = json.loads(settings_json)

    parser = AnimationSpecificationParser()
    spec = parser.parse(spec_raw)
    timeline = Timeline.from_spec(spec)
    channels = KeyframeGenerationEngine().generate(spec, timeline)

    engine = PreviewRenderingEngine()
    result = engine.render(spec, channels, settings)
    metrics = evaluate_animation(spec, channels, result, timeline)

    return {
        "video_bytes": result.video_bytes,
        "video_mime": result.video_mime,
        "video_filename": result.video_filename,
        "blend": result.blend_bytes,
        "log": result.log,
        "used_blender": result.used_blender,
        "render_seconds": result.render_seconds,
        "metrics": metrics,
        "spec": spec,
        "channels": channels,
        "timeline": asdict(timeline),
    }


# ======================================================================
# Streamlit UI
# ======================================================================
RESOLUTIONS = {
    "Fast 320p": (320, 180),
    "Demo 480p": (480, 270),
    "Quality 640p": (640, 360),
    "High 960p": (960, 540),
}

INTERPOLATIONS = ["BEZIER", "LINEAR", "CONSTANT", "SINE", "QUAD", "CUBIC"]
ENGINE_OPTIONS = ["BLENDER_WORKBENCH", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"]


def default_settings() -> Dict[str, Any]:
    return {
        "blender_path": find_blender(),
        "resolution_label": "Fast 320p",
        "width": RESOLUTIONS["Fast 320p"][0],
        "height": RESOLUTIONS["Fast 320p"][1],
        "engine": "BLENDER_WORKBENCH",
        "timeout": 180.0,
        "auto_render": False,
    }


def init_session_state() -> None:
    if "template_name" not in st.session_state or st.session_state.template_name not in SAMPLE_PROJECTS:
        st.session_state.template_name = "Full Demo"
    if "overrides" not in st.session_state:
        st.session_state.overrides = {}
    if "custom_spec" not in st.session_state:
        st.session_state.custom_spec = None
    if "spec_version" not in st.session_state:
        st.session_state.spec_version = 0
    if "settings" not in st.session_state:
        st.session_state.settings = default_settings()
    if "result" not in st.session_state:
        st.session_state.result = None
    if "result_key" not in st.session_state:
        st.session_state.result_key = None
    if "pending_render" not in st.session_state:
        st.session_state.pending_render = True
    if "render_cache" not in st.session_state:
        st.session_state.render_cache = {}


def bump_spec_version() -> None:
    st.session_state.spec_version = int(st.session_state.get("spec_version", 0)) + 1


def current_raw_base() -> Dict[str, Any]:
    if st.session_state.custom_spec is not None:
        return copy.deepcopy(st.session_state.custom_spec)
    return SAMPLE_PROJECTS[st.session_state.template_name]()


def compose_spec() -> Dict[str, Any]:
    base = current_raw_base()
    ov = st.session_state.overrides

    if not isinstance(base.get("camera"), dict):
        base["camera"] = copy.deepcopy(DEFAULT_SPEC["camera"])

    clean_objects = []
    for obj in base.get("objects", []) or []:
        if not isinstance(obj, dict):
            obj = {"name": str(obj)}
        motion = obj.get("motion")
        if isinstance(motion, str):
            obj["motion"] = {"type": motion}
        elif not isinstance(motion, dict):
            obj["motion"] = {"type": "idle"}
        clean_objects.append(obj)
    base["objects"] = clean_objects

    base["duration"] = float(ov.get("duration", base.get("duration", 3.0)))
    base["fps"] = int(ov.get("fps", base.get("fps", 24)))
    base["interpolation"] = str(ov.get("interpolation", base.get("interpolation", "BEZIER")))
    base["keyframe_step"] = int(ov.get("keyframe_step", base.get("keyframe_step", 1)))

    camera = base.setdefault("camera", {})
    camera["speed"] = float(ov.get("camera_speed", camera.get("speed", 45.0)))

    motion_mult = float(ov.get("motion_speed", 1.0))
    amplitude_mult = float(ov.get("amplitude_scale", 1.0))
    for obj in base.get("objects", []):
        motion = obj.setdefault("motion", {})
        for key in ("speed", "rpm"):
            if key in motion:
                motion[key] = float(motion[key]) * motion_mult
        for key in ("amplitude", "radius", "oscillate_amplitude"):
            if key in motion:
                motion[key] = float(motion[key]) * amplitude_mult

    return AnimationSpecificationParser().parse(base)


def settings_hashable(settings: Dict[str, Any]) -> str:
    return json.dumps(settings, sort_keys=True)


def render_key_for(spec: Dict[str, Any], settings: Dict[str, Any]) -> str:
    payload = json.dumps(spec, sort_keys=True) + "|" + settings_hashable(settings)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def perform_render(spec_json: str, settings_json: str, key: str) -> None:
    cache = st.session_state.setdefault("render_cache", {})
    if key in cache:
        st.session_state.result = cache[key]
        st.session_state.result_key = key
        return

    try:
        with st.spinner("Generating keyframes and rendering live preview..."):
            result = _render_pipeline(spec_json, settings_json)
    except Exception as exc:
        try:
            fallback_spec = AnimationSpecificationParser().parse(json.loads(spec_json))
        except Exception:
            fallback_spec = AnimationSpecificationParser().parse({})
        result = {
            "video_bytes": None,
            "video_mime": "video/mp4",
            "video_filename": "motion_demo.mp4",
            "blend": None,
            "log": f"Render pipeline exception:\n{exc}\n{traceback.format_exc()}",
            "used_blender": False,
            "render_seconds": 0.0,
            "metrics": {},
            "spec": fallback_spec,
            "channels": {"objects": {}, "camera": []},
            "timeline": asdict(Timeline()),
        }

    st.session_state.result = result
    st.session_state.result_key = key

    if result.get("video_bytes"):
        cache.clear()
        cache[key] = result


def sidebar_controls() -> None:
    st.sidebar.header("⚙️ Platform Controls")

    template_names = list(SAMPLE_PROJECTS.keys())
    current_index = template_names.index(st.session_state.template_name) if st.session_state.template_name in template_names else 0
    selected_template = st.sidebar.selectbox("Motion template", template_names, index=current_index, key="template_select")
    if selected_template != st.session_state.template_name:
        st.session_state.template_name = selected_template
        st.session_state.overrides = {}
        st.session_state.custom_spec = None
        st.session_state.pending_render = True
        bump_spec_version()

    base = AnimationSpecificationParser().parse(current_raw_base())
    ov = st.session_state.overrides
    suffix = f"{st.session_state.template_name}_{st.session_state.spec_version}"

    st.sidebar.subheader("Timeline")
    duration = st.sidebar.slider(
        "Duration (seconds)",
        0.5,
        30.0,
        _clamp(float(ov.get("duration", base.get("duration", 3.0))), 0.5, 30.0),
        0.1,
        key=f"duration_{suffix}",
    )
    fps = st.sidebar.number_input(
        "FPS",
        min_value=1,
        max_value=120,
        value=int(_clamp(float(ov.get("fps", base.get("fps", 24))), 1, 120)),
        step=1,
        key=f"fps_{suffix}",
    )
    keyframe_step = st.sidebar.number_input(
        "Keyframe step",
        min_value=1,
        max_value=12,
        value=int(_clamp(float(ov.get("keyframe_step", base.get("keyframe_step", 1))), 1, 12)),
        key=f"kf_step_{suffix}",
    )
    interp_value = str(ov.get("interpolation", base.get("interpolation", "BEZIER"))).upper()
    if interp_value not in INTERPOLATIONS:
        interp_value = "BEZIER"
    interpolation = st.sidebar.selectbox(
        "Keyframe interpolation",
        INTERPOLATIONS,
        index=INTERPOLATIONS.index(interp_value),
        key=f"interp_{suffix}",
    )

    st.sidebar.subheader("Motion Parameters")
    motion_speed = st.sidebar.slider(
        "Motion speed multiplier",
        0.1,
        3.0,
        _clamp(float(ov.get("motion_speed", 1.0)), 0.1, 3.0),
        0.05,
        key=f"motion_speed_{suffix}",
    )
    amplitude_scale = st.sidebar.slider(
        "Amplitude / radius scale",
        0.1,
        3.0,
        _clamp(float(ov.get("amplitude_scale", 1.0)), 0.1, 3.0),
        0.05,
        key=f"amplitude_{suffix}",
    )
    camera_speed = st.sidebar.slider(
        "Camera speed °/s",
        -180.0,
        180.0,
        _clamp(float(ov.get("camera_speed", base.get("camera", {}).get("speed", 45.0))), -180.0, 180.0),
        1.0,
        key=f"camera_speed_{suffix}",
    )

    ov.update(
        {
            "duration": duration,
            "fps": fps,
            "keyframe_step": int(keyframe_step),
            "interpolation": interpolation,
            "motion_speed": motion_speed,
            "amplitude_scale": amplitude_scale,
            "camera_speed": camera_speed,
        }
    )

    st.sidebar.subheader("Renderer")
    settings = st.session_state.settings
    settings["blender_path"] = st.sidebar.text_input(
        "Blender executable path",
        value=settings.get("blender_path", ""),
        help="Leave blank to autodetect. Install Blender separately; it is not a pip package.",
        key="blender_path",
    )
    res_label = st.sidebar.selectbox(
        "Preview resolution",
        list(RESOLUTIONS.keys()),
        index=list(RESOLUTIONS.keys()).index(settings.get("resolution_label", "Fast 320p")),
        key="resolution_label",
    )
    settings["resolution_label"] = res_label
    settings["width"], settings["height"] = RESOLUTIONS[res_label]
    settings["engine"] = st.sidebar.selectbox(
        "Blender render engine",
        ENGINE_OPTIONS,
        index=ENGINE_OPTIONS.index(settings.get("engine", "BLENDER_WORKBENCH")) if settings.get("engine", "BLENDER_WORKBENCH") in ENGINE_OPTIONS else 0,
        key="engine",
    )
    settings["timeout"] = float(
        st.sidebar.number_input(
            "Blender timeout seconds",
            min_value=30.0,
            max_value=900.0,
            value=float(settings.get("timeout", 180.0)),
            step=15.0,
            key="timeout",
        )
    )
    settings["auto_render"] = st.sidebar.checkbox(
        "Auto render on change",
        value=bool(settings.get("auto_render", False)),
        help="Turn on after the initial demo if you want every parameter change to trigger a new render.",
        key="auto_render",
    )

    st.sidebar.caption("No API keys. Synthetic/local demo data only. Blender is optional for real viewport rendering; CPU fallback previews are built in.")


def demo_tab(spec: Dict[str, Any], settings: Dict[str, Any], spec_json: str, settings_json: str, key: str) -> None:
    result = st.session_state.get("result")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Live working demo")
    with col2:
        if st.button("Render / regenerate", type="primary", use_container_width=True, key="render_regenerate_btn"):
            perform_render(spec_json, settings_json, key)
            st.rerun()
    with col3:
        if st.button("Reset to template", use_container_width=True, key="reset_template_btn"):
            st.session_state.custom_spec = None
            st.session_state.overrides = {}
            st.session_state.pending_render = True
            bump_spec_version()
            st.rerun()

    if not result:
        st.info("No render yet. Click Render / regenerate.")
        return

    video_bytes = result.get("video_bytes")
    video_mime = result.get("video_mime", "video/mp4")
    video_filename = result.get("video_filename", "motion_demo.mp4")

    if video_bytes:
        if video_mime.startswith("video/"):
            st.video(video_bytes, format=video_mime)
        else:
            st.image(video_bytes, use_container_width=True)
    else:
        st.error("Video could not be generated. See renderer log below.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Renderer", result.get("metrics", {}).get("renderer", "Fallback"))
    m2.metric("Render time", f"{float(result.get('render_seconds', 0.0)):.2f}s")
    m3.metric("Frames", int(result.get("metrics", {}).get("frames_generated", 0)))
    m4.metric("Export success", "Yes" if result.get("metrics", {}).get("export_success") else "No")

    c1, c2, c3, c4 = st.columns(4)
    if video_bytes:
        label = "Download MP4" if video_mime.startswith("video/") else "Download GIF"
        c1.download_button(
            label,
            data=video_bytes,
            file_name=video_filename,
            mime=video_mime,
            use_container_width=True,
            key="dl_video_demo",
        )
    if result.get("blend"):
        c2.download_button(
            "Download .blend",
            data=result["blend"],
            file_name="motion_demo.blend",
            mime="application/octet-stream",
            use_container_width=True,
            key="dl_blend_demo",
        )
    c3.download_button(
        "Download Blender script",
        data=BLENDER_SCRIPT,
        file_name="blender_motion_script.py",
        mime="text/x-python",
        use_container_width=True,
        key="dl_script_demo",
    )
    c4.download_button(
        "Download spec JSON",
        data=ExportManager.spec_json(spec),
        file_name="motion_spec.json",
        mime="application/json",
        use_container_width=True,
        key="dl_spec_demo",
    )

    with st.expander("Renderer log"):
        st.code(result.get("log", ""), language="log")


def builder_tab(spec: Dict[str, Any]) -> None:
    st.subheader("Motion builder")

    left, right = st.columns([1.1, 1.0])
    with left:
        st.markdown("**Current objects**")
        rows = []
        for obj in spec.get("objects", []):
            motion = obj.get("motion", {})
            rows.append(
                {
                    "name": obj.get("name"),
                    "kind": obj.get("kind"),
                    "motion": motion.get("type"),
                    "location": [round(float(v), 3) for v in obj.get("location", [])],
                    "scale": [round(float(v), 3) for v in obj.get("scale", [])],
                }
            )
        st.dataframe(rows, use_container_width=True)

        st.markdown("**Camera**")
        st.json(spec.get("camera", {}))

    with right:
        st.markdown("**Editable specification JSON**")
        editor_key = f"spec_editor_{st.session_state.spec_version}"
        edited = st.text_area(
            "Paste or edit a motion specification",
            value=ExportManager.spec_json(spec),
            height=360,
            key=editor_key,
        )
        b1, b2 = st.columns(2)
        if b1.button("Apply JSON spec", use_container_width=True, key="apply_json_btn"):
            try:
                parsed = json.loads(edited)
                st.session_state.custom_spec = parsed
                st.session_state.overrides = {}
                st.session_state.pending_render = True
                bump_spec_version()
                st.success("Custom specification applied.")
                st.rerun()
            except Exception as exc:
                st.error(f"Invalid JSON: {exc}")
        if b2.button("Clear custom spec", use_container_width=True, key="clear_custom_btn"):
            st.session_state.custom_spec = None
            st.session_state.overrides = {}
            st.session_state.pending_render = True
            bump_spec_version()
            st.success("Custom specification cleared.")
            st.rerun()

    with st.expander("Import a local 3D model reference"):
        st.markdown(
            textwrap.dedent(
                """
                For real Blender rendering, set import_model to an absolute local path and import_as to the object name used in channels.
                Supported by the generated Blender script: .obj, .fbx, .glb, .gltf, and simple .blend object appending.
                """
            )
        )
        st.code(
            json.dumps(
                {
                    "import_model": "C:/models/my_part.glb",
                    "import_as": "MyPart",
                    "objects": [
                        {
                            "name": "MyPart",
                            "kind": "empty",
                            "motion": {"type": "rotation", "axis": "z", "speed": 60}
                        }
                    ]
                },
                indent=2,
            ),
            language="json",
        )


def timeline_tab(spec: Dict[str, Any], settings: Dict[str, Any]) -> None:
    st.subheader("Timeline manager")
    timeline = Timeline.from_spec(spec)
    channels = KeyframeGenerationEngine().generate(spec, timeline)

    st.write(
        f"**Frames:** {timeline.frame_start}–{timeline.frame_end} | "
        f"**FPS:** {timeline.fps} | "
        f"**Duration:** {timeline.duration:.2f}s | "
        f"**Keyframe step:** {spec.get('keyframe_step', 1)}"
    )

    frame = st.slider(
        "Scrub frame",
        min_value=int(timeline.frame_start),
        max_value=int(timeline.frame_end),
        value=int(timeline.frame_start),
        step=1,
        key="timeline_frame_scrub",
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Preview selected frame", use_container_width=True, key="preview_frame_btn"):
            png = _render_single_frame_png(spec, channels, frame, int(settings.get("width", 320)), int(settings.get("height", 180)))
            st.session_state["frame_preview"] = png
    with c2:
        if st.session_state.get("frame_preview"):
            st.image(st.session_state["frame_preview"], caption=f"Frame {frame}", use_container_width=True)
        else:
            st.info("Click Preview selected frame to render a still from the current procedural timeline.")

    if spec.get("events"):
        st.markdown("**Timeline markers**")
        st.dataframe(spec["events"], use_container_width=True)
    else:
        st.caption("No timeline markers in this specification.")


def evaluation_tab(spec: Dict[str, Any]) -> None:
    st.subheader("Evaluation")
    result = st.session_state.get("result")
    if result and result.get("metrics"):
        metrics = result["metrics"]
    else:
        timeline = Timeline.from_spec(spec)
        channels = KeyframeGenerationEngine().generate(spec, timeline)
        dry = RenderResult(
            video_bytes=None,
            video_mime="video/mp4",
            video_filename="motion_demo.mp4",
            blend_bytes=None,
            log="Dry evaluation",
            used_blender=False,
            render_seconds=0.0,
        )
        metrics = evaluate_animation(spec, channels, dry, timeline)

    st.dataframe([metrics], use_container_width=True)

    st.markdown(
        textwrap.dedent(
            """
            **Metric meaning**
            - **Keyframe accuracy max error:** maximum deviation between dense analytic motion and interpolated keyframe curve.
            - **Motion continuity max units/s:** highest frame-to-frame translational/rotational rate; lower is smoother.
            - **Playback stability score:** combined curve regularity and render throughput estimate.
            - **Rendering time:** wall-clock time for keyframe generation plus preview rendering.
            - **Export success:** whether a playable video file was produced.
            """
        )
    )


def export_tab(spec: Dict[str, Any]) -> None:
    st.subheader("Export manager")
    timeline = Timeline.from_spec(spec)
    channels = KeyframeGenerationEngine().generate(spec, timeline)

    c1, c2, c3 = st.columns(3)
    c1.download_button(
        "Download spec JSON",
        data=ExportManager.spec_json(spec),
        file_name="motion_spec.json",
        mime="application/json",
        use_container_width=True,
        key="dl_spec_export",
    )
    c2.download_button(
        "Download channels JSON",
        data=ExportManager.channels_json(channels),
        file_name="motion_channels.json",
        mime="application/json",
        use_container_width=True,
        key="dl_channels_export",
    )
    c3.download_button(
        "Download payload for Blender",
        data=ExportManager.payload_json(spec, channels, timeline),
        file_name="motion_payload.json",
        mime="application/json",
        use_container_width=True,
        key="dl_payload_export",
    )

    st.download_button(
        "Download standalone Blender Python script",
        data=BLENDER_SCRIPT,
        file_name="blender_motion_script.py",
        mime="text/x-python",
        use_container_width=True,
        key="dl_script_export",
    )

    st.markdown("**Sample projects**")
    sample = st.selectbox("Choose sample project", list(SAMPLE_PROJECTS.keys()), key="sample_project_select")
    sample_spec = AnimationSpecificationParser().parse(SAMPLE_PROJECTS[sample]())
    st.download_button(
        f"Download {sample} sample",
        data=ExportManager.spec_json(sample_spec),
        file_name=f"{sample.lower().replace(' ', '_')}.json",
        mime="application/json",
        use_container_width=True,
        key="dl_sample_export",
    )


def tests_tab() -> None:
    st.subheader("Automated tests")
    st.write("Runs the built-in Pytest suite against this single-file application.")
    if st.button("Run Pytest suite", type="primary", key="run_pytest_btn"):
        file_path = str(Path(globals().get("__file__", "app.py")).resolve())
        cmd = [sys.executable, "-m", "pytest", file_path, "-k", "test_", "-v"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            st.code((proc.stdout or "") + "\n" + (proc.stderr or ""), language="text")
        except FileNotFoundError:
            st.error("pytest was not found in the current Python environment.")
        except Exception as exc:
            st.error(f"Test execution failed: {exc}")


DOCS_MARKDOWN = """
# 3D Motion Design Generation Platform

A single-file Streamlit application for procedural motion design, keyframe generation,
camera animation, mechanical templates, preview rendering, evaluation, and export.

## Technology

- Python
- Streamlit
- Blender Python API through a generated standalone Blender script
- NumPy
- SciPy for spline-based continuity/accuracy checks
- PyTorch for optional torch-based easing curves
- Pytest for automated tests

## Core modules implemented in app.py

1. Animation specification parser
   Normalizes template/user JSON into a deterministic internal spec.

2. Object motion engine
   Produces analytic transforms for translation, rotation, scaling, repeated movement,
   circular motion, mechanical rotation, crank/slider, and basic interaction.

3. Keyframe generation engine
   Converts analytic motion into Blender-editable keyframe channels.

4. Interpolation system
   Maps internal easing/interpolation choices to Blender F-Curve interpolation modes.

5. Camera animation manager
   Supports static, orbit, dolly, crane, follow, and manual camera paths.

6. Procedural animation templates
   Built-in sample projects for common motion-design patterns.

7. Mechanical motion module
   Gear ratios, rack conversion, and crank-slider approximation.

8. Timeline manager
   FPS, frame range, duration, keyframe step, and marker events.

9. Preview and rendering engine
   Uses Blender headless when available; otherwise renders CPU fallback MP4/GIF previews
   so the dashboard is never blank.

10. Export manager
   Downloads video, .blend, Blender script, spec JSON, channels JSON, and payload JSON.

## Workflow

1. Select a scene/template or import a local 3D model reference.
2. Define animation duration, FPS, and keyframe step.
3. Select a motion template.
4. Configure movement parameters.
5. Generate keyframes.
6. Preview the animation.
7. Modify the timeline.
8. Render and export.

## Run locally

pip install -r requirements.txt
streamlit run app.py

Blender is optional for the initial demo because the app includes CPU fallback renderers.
For real Blender rendering, install Blender separately and either put blender on PATH
or set the Blender executable path in the sidebar.

## Editability in Blender

The generated Blender script inserts real object and camera keyframes, sets F-Curve
interpolation, adds timeline markers, stores the original specification as custom
properties, and saves a .blend file when Blender rendering succeeds. This keeps the
animation fully editable in Blender's dope sheet, graph editor, and outliner.

## Evaluation

The app measures:

- keyframe accuracy,
- motion continuity,
- playback stability,
- rendering time,
- export success.

## Automated tests

The file includes Pytest-compatible tests. Use the Tests tab or run:

pytest app.py -k test_ -v
"""


def docs_tab() -> None:
    st.subheader("Documentation")
    st.markdown(DOCS_MARKDOWN)


def main() -> None:
    st.set_page_config(page_title="3D Motion Design Generation Platform", page_icon="🎬", layout="wide")
    init_session_state()

    st.title("🎬 3D Motion Design Generation Platform")
    st.caption("Procedural animation • Blender-editable keyframes • CPU fallback live demo • no API keys")

    sidebar_controls()

    spec = compose_spec()
    settings = copy.deepcopy(st.session_state.settings)
    spec_json = json.dumps(spec, sort_keys=True)
    settings_json = settings_hashable(settings)
    key = render_key_for(spec, settings)

    if st.session_state.pending_render or bool(settings.get("auto_render")):
        if st.session_state.result_key != key or st.session_state.result is None:
            perform_render(spec_json, settings_json, key)
        st.session_state.pending_render = False

    tabs = st.tabs(["Live Demo", "Motion Builder", "Timeline", "Evaluation", "Export", "Tests", "Docs"])
    with tabs[0]:
        demo_tab(spec, settings, spec_json, settings_json, key)
    with tabs[1]:
        builder_tab(spec)
    with tabs[2]:
        timeline_tab(spec, settings)
    with tabs[3]:
        evaluation_tab(spec)
    with tabs[4]:
        export_tab(spec)
    with tabs[5]:
        tests_tab()
    with tabs[6]:
        docs_tab()


# ======================================================================
# Automated tests
# ======================================================================
def test_parser_sets_defaults() -> None:
    parser = AnimationSpecificationParser()
    spec = parser.parse({"objects": [{"name": "A"}]})
    assert spec["fps"] == 24
    assert spec["objects"][0]["kind"] == "cube"
    assert spec["objects"][0]["motion"]["type"] == "idle"


def test_timeline_includes_endpoint() -> None:
    spec = AnimationSpecificationParser().parse({"duration": 1.0, "fps": 24, "frame_start": 1})
    tl = Timeline.from_spec(spec)
    assert tl.frame_start == 1
    assert tl.frame_end == 25
    assert abs(tl.time_at_frame(tl.frame_end) - 1.0) < 1e-6


def test_translation_reaches_target() -> None:
    spec = AnimationSpecificationParser().parse(
        {
            "duration": 1.0,
            "objects": [
                {
                    "name": "A",
                    "motion": {"type": "translation", "from": [0, 0, 0], "to": [1, 2, 3], "easing": "linear"},
                }
            ],
        }
    )
    engine = ObjectMotionEngine(spec)
    tr = engine.base_transform(spec["objects"][0], 1.0)
    assert np.allclose(tr["location"], [1, 2, 3], atol=1e-5)


def test_rotation_speed_conversion() -> None:
    spec = AnimationSpecificationParser().parse(
        {"duration": 1.0, "objects": [{"name": "A", "motion": {"type": "rotation", "axis": "z", "speed": 90}}]}
    )
    engine = ObjectMotionEngine(spec)
    tr = engine.base_transform(spec["objects"][0], 1.0)
    assert abs(tr["rotation_euler"][2] - math.pi / 2.0) < 1e-5


def test_circular_radius_is_constant() -> None:
    spec = AnimationSpecificationParser().parse(
        {
            "duration": 2.0,
            "objects": [
                {
                    "name": "A",
                    "motion": {"type": "circular", "center": [0, 0, 0], "radius": 2.5, "speed": 60, "plane": "xy"},
                }
            ],
        }
    )
    engine = ObjectMotionEngine(spec)
    for t in np.linspace(0, 2, 11):
        tr = engine.base_transform(spec["objects"][0], float(t))
        radius = math.hypot(tr["location"][0], tr["location"][1])
        assert abs(radius - 2.5) < 1e-5


def test_mechanical_gear_ratio_direction() -> None:
    spec = AnimationSpecificationParser().parse(
        {
            "duration": 1.0,
            "objects": [
                {"name": "A", "motion": {"type": "mechanical_rotation", "axis": "z", "rpm": 60, "ratio": -2.0}},
            ],
        }
    )
    engine = ObjectMotionEngine(spec)
    tr = engine.base_transform(spec["objects"][0], 1.0)
    expected = -2.0 * 2.0 * math.pi
    assert abs(tr["rotation_euler"][2] - expected) < 1e-5


def test_crank_slider_finite_and_periodic() -> None:
    spec = AnimationSpecificationParser().parse(
        {
            "duration": 2.0,
            "objects": [
                {
                    "name": "S",
                    "location": [0, 0, 0],
                    "motion": {"type": "crank_slider", "rpm": 60, "radius": 1.0, "rod_length": 3.0, "axis": "x"},
                }
            ],
        }
    )
    engine = ObjectMotionEngine(spec)
    vals = [engine.base_transform(spec["objects"][0], float(t))["location"][0] for t in np.linspace(0, 2, 101)]
    assert all(np.isfinite(vals))
    assert max(vals) - min(vals) > 0.5


def test_interpolation_system_maps_names() -> None:
    assert InterpolationSystem.blender_name("bezier") == "BEZIER"
    assert InterpolationSystem.blender_name("linear") == "LINEAR"
    assert InterpolationSystem.blender_name("unknown") == "BEZIER"


def test_keyframe_generation_counts() -> None:
    spec = AnimationSpecificationParser().parse(
        {"duration": 1.0, "fps": 12, "keyframe_step": 1, "objects": [{"name": "A"}]}
    )
    tl = Timeline.from_spec(spec)
    channels = KeyframeGenerationEngine().generate(spec, tl)
    assert len(channels["objects"]["A"]) == tl.frame_count
    assert len(channels["camera"]) == tl.frame_count


def test_camera_orbit_quaternion_normalized() -> None:
    spec = AnimationSpecificationParser().parse({"duration": 1.0, "camera": {"type": "orbit", "speed": 90}})
    cam = CameraAnimationManager(spec)
    q = cam.compute(0.3)["rotation_quaternion"]
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-5


def test_blender_script_contains_required_api() -> None:
    assert "import bpy" in BLENDER_SCRIPT
    assert "keyframe_insert" in BLENDER_SCRIPT
    assert "FFMPEG" in BLENDER_SCRIPT


def test_gif_encoder_produces_header_and_trailer() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = 255
    gif = _encode_gif([frame, frame], fps=10, loop=0)
    assert gif.startswith(b"GIF89a")
    assert gif.endswith(b"\x3B")
    assert len(gif) > 100


def test_metrics_evaluation_has_expected_keys() -> None:
    spec = AnimationSpecificationParser().parse({"duration": 1.0, "fps": 12, "objects": [{"name": "A"}]})
    tl = Timeline.from_spec(spec)
    channels = KeyframeGenerationEngine().generate(spec, tl)
    result = RenderResult(
        video_bytes=b"x",
        video_mime="video/mp4",
        video_filename="motion_demo.mp4",
        blend_bytes=None,
        log="",
        used_blender=False,
        render_seconds=0.5,
    )
    metrics = evaluate_animation(spec, channels, result, tl)
    for metric_key in (
        "keyframe_accuracy_max_error",
        "motion_continuity_max_units_per_s",
        "playback_stability_score",
        "rendering_time_s",
        "export_success",
    ):
        assert metric_key in metrics


def test_sample_projects_parse() -> None:
    parser = AnimationSpecificationParser()
    for name, fn in SAMPLE_PROJECTS.items():
        spec = parser.parse(fn())
        assert spec["name"]
        assert spec["duration"] > 0
        assert isinstance(spec["objects"], list)


if "pytest" not in sys.modules:
    main()