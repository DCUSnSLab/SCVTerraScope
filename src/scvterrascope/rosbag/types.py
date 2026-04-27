"""Lightweight DTOs for the rosbag → GUI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BagFrame:
    """One image message decoded from a ROS 2 bag.

    `idx` is the 0-based position within the chosen topic — useful for
    seek bars / step-by-step navigation. `ros_time_ns` is the original
    ROS receive timestamp; subtract `bag_start_ns` from it to get a
    seconds-from-start float.
    """

    idx: int
    ros_time_ns: int
    image: Any                  # PIL.Image.Image (RGB)
    topic: str
    encoding: str               # original encoding string ('bgra8', 'rgb8', etc.)

    def time_seconds_from(self, bag_start_ns: int) -> float:
        return (self.ros_time_ns - bag_start_ns) / 1e9
