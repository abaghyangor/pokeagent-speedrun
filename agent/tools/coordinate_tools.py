"""Coordinate calibration helpers.

Game Boy Advance emulator coordinates occasionally jitter or report
stale values, especially around cutscenes. This helper smooths the
values so the agent does not chase phantom offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class CalibratedCoords:
    coords: Optional[Tuple[int, int]]
    reliability: str
    raw: Optional[Tuple[int, int]]

    def describe(self) -> str:
        if not self.coords:
            return "Unknown"
        return f"{self.coords} ({self.reliability})"


class CoordinateCalibrator:
    def __init__(self) -> None:
        self._last_coords: Optional[Tuple[int, int]] = None
        self._spike_threshold = 6  # tiles

    def update(self, state: Dict[str, Any]) -> CalibratedCoords:
        coords = self._extract(state)
        reliability = "fresh"
        raw_coords = coords
        if coords is None:
            return CalibratedCoords(coords=self._last_coords, reliability="stale", raw=None)
        if self._last_coords:
            if abs(coords[0] - self._last_coords[0]) > self._spike_threshold or \
               abs(coords[1] - self._last_coords[1]) > self._spike_threshold:
                # treat as sensor spike; keep previous coordinate
                reliability = "smoothed"
                coords = self._last_coords
            else:
                reliability = "stable"
        self._last_coords = coords
        return CalibratedCoords(coords=coords, reliability=reliability, raw=raw_coords)

    @staticmethod
    def _extract(state: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        if not isinstance(state, dict):
            return None

        candidates = []
        player_section = state.get("player")
        if isinstance(player_section, dict):
            candidates.extend(
                player_section.get(key)
                for key in ("position", "coordinates", "coords")
            )

        map_section = state.get("map")
        if isinstance(map_section, dict):
            candidates.extend(
                map_section.get(key)
                for key in ("player_coords", "player_position", "player_local_pos", "player_pos")
            )

        for key in ("player_position", "player_coords", "player_local_pos", "player_pos", "position"):
            candidates.append(state.get(key))

        for candidate in candidates:
            coords = CoordinateCalibrator._coerce_coords(candidate)
            if coords is not None:
                return coords
        return None

    @staticmethod
    def _coerce_coords(value: Any) -> Optional[Tuple[int, int]]:
        if value is None:
            return None
        if isinstance(value, dict):
            x = value.get("x")
            y = value.get("y")
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            x, y = value[0], value[1]
        else:
            return None

        if x is None or y is None:
            return None
        try:
            return int(x), int(y)
        except (TypeError, ValueError):
            return None
