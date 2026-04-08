"""
shared/schema.py
Single source of truth for payload schema and inter-service constants.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import Enum
import json


class EquipmentState(str, Enum):
    ACTIVE   = "ACTIVE"
    INACTIVE = "INACTIVE"


class Activity(str, Enum):
    # Excavator
    DIGGING  = "DIGGING"
    SWINGING = "SWINGING"
    # Dump truck
    MOVING   = "MOVING"
    # Concrete mixer
    MIXING   = "MIXING"
    # Shared
    DUMPING  = "DUMPING"
    WAITING  = "WAITING"


class MotionSource(str, Enum):
    FULL_BODY   = "full_body"
    ARM_ONLY    = "arm_only"
    TRACKS_ONLY = "tracks_only"
    NONE        = "none"


@dataclass
class UtilizationInfo:
    current_state:    EquipmentState
    current_activity: Activity
    motion_source:    MotionSource

    def to_dict(self) -> dict:
        return {
            "current_state":    self.current_state.value,
            "current_activity": self.current_activity.value,
            "motion_source":    self.motion_source.value,
        }


@dataclass
class TimeAnalytics:
    total_tracked_seconds: float = 0.0
    total_active_seconds:  float = 0.0
    total_idle_seconds:    float = 0.0
    utilization_percent:   float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_tracked_seconds": round(self.total_tracked_seconds, 2),
            "total_active_seconds":  round(self.total_active_seconds, 2),
            "total_idle_seconds":    round(self.total_idle_seconds, 2),
            "utilization_percent":   round(self.utilization_percent, 1),
        }


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EquipmentPayload:
    frame_id:        int
    equipment_id:    str
    equipment_class: str
    timestamp:       str
    utilization:     UtilizationInfo
    bbox:            BoundingBox
    time_analytics:  TimeAnalytics

    def to_dict(self) -> dict:
        return {
            "frame_id":        self.frame_id,
            "equipment_id":    self.equipment_id,
            "equipment_class": self.equipment_class,
            "timestamp":       self.timestamp,
            "utilization":     self.utilization.to_dict(),
            "bbox":            self.bbox.to_dict(),
            "time_analytics":  self.time_analytics.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str | bytes) -> "EquipmentPayload":
        d = json.loads(raw)
        return cls(
            frame_id        = d["frame_id"],
            equipment_id    = d["equipment_id"],
            equipment_class = d["equipment_class"],
            timestamp       = d["timestamp"],
            utilization     = UtilizationInfo(
                current_state    = EquipmentState(d["utilization"]["current_state"]),
                current_activity = Activity(d["utilization"]["current_activity"]),
                motion_source    = MotionSource(d["utilization"]["motion_source"]),
            ),
            bbox           = BoundingBox(**d["bbox"]),
            time_analytics = TimeAnalytics(**d["time_analytics"]),
        )


KAFKA_TOPIC = "equipment_telemetry"