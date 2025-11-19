"""Objective prioritization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ObjectiveSummary:
    primary: str
    secondary: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [f"Primary: {self.primary}"]
        if self.secondary:
            lines.append("Secondary: " + "; ".join(self.secondary))
        if self.optional:
            lines.append("Optional: " + "; ".join(self.optional))
        return "\n".join(lines)


class ObjectivePrioritizer:
    def __init__(self) -> None:
        self._recent_primary: Optional[str] = None

    def evaluate(
        self,
        state: Dict[str, Any],
        coords: Optional[Tuple[int, int]],
        dialogue_summary: str,
    ) -> ObjectiveSummary:
        milestones = state.get("milestones") or {}
        location = (state.get("player", {}).get("location") or state.get("location") or "").upper()
        primary = "Explore the current area"
        secondary: List[str] = []
        optional: List[str] = []

        if not self._is_done(milestones, "PLAYER_HOUSE_ENTERED"):
            primary = "Finish the outdoor conversation and enter the house"
        elif not self._is_done(milestones, "PLAYER_BEDROOM"):
            primary = "Reach your bedroom upstairs"
        elif not self._is_done(milestones, "CLOCK_SET"):
            if "2F" in location:
                primary = "Set the stopped wall clock"
                secondary.append("Stand directly in front of the clock tile and press A")
            else:
                primary = "Go upstairs to set the clock"
        else:
            primary = "Leave the house and head toward Professor Birch"
            secondary.append("Once outside, move northwest to the lab")

        if dialogue_summary:
            optional.append(dialogue_summary[:90])
        if coords:
            optional.append(f"Current coords: {coords}")

        self._recent_primary = primary
        return ObjectiveSummary(primary=primary, secondary=secondary, optional=optional)

    @staticmethod
    def _is_done(milestones: Dict[str, Any], key: str) -> bool:
        info = milestones.get(key)
        if isinstance(info, dict):
            return bool(info.get("completed"))
        return bool(info)
