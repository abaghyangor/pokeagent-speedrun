"""Dialogue analysis helpers.

The stock emulator state exposes both frame-level dialogue detection and
raw script text. We wrap that into a stricter signal so the agent only
enters "dialogue" mode when both the visual detector reports a box AND
there is meaningful text. This prevents false positives where the
`game_state` string still says "dialog" even though the player already
has control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DialogueStatus:
    active: bool
    text: str
    speaker: Optional[str]
    confidence: float

    def summary(self) -> str:
        if not self.text:
            return "No active dialogue"
        speaker = f"{self.speaker}: " if self.speaker else ""
        return f"{speaker}{self.text.strip()}"


class DialogueAnalyzer:
    def __init__(self) -> None:
        self._last_text: str = ""

    def analyze_state(self, state: Dict[str, Any]) -> DialogueStatus:
        text = self._extract_text(state)
        dialogue_visible = self._detect_frame_dialogue(state)
        active = dialogue_visible and bool(text)
        speaker = self._detect_speaker(text)
        confidence = 0.9 if active else 0.1
        if active and text == self._last_text:
            confidence = 0.6  # repeated text -> possibly skippable
        self._last_text = text if active else ""
        return DialogueStatus(active=active, text=text, speaker=speaker, confidence=confidence)

    @staticmethod
    def _detect_frame_dialogue(state: Dict[str, Any]) -> bool:
        visual = state.get("visual")
        if isinstance(visual, dict):
            dv = visual.get("dialogue_detected")
            if isinstance(dv, dict) and dv.get("has_dialogue"):
                return True
        game_section = state.get("game", {})
        if isinstance(game_section, dict):
            dv = game_section.get("dialogue_detected")
            if isinstance(dv, dict) and dv.get("has_dialogue"):
                return True
        return False

    @staticmethod
    def _extract_text(state: Dict[str, Any]) -> str:
        game_section = state.get("game", {})
        if isinstance(game_section, dict):
            text = game_section.get("dialog_text")
            if isinstance(text, str) and text.strip():
                return text.strip()
        fallback = state.get("dialogue")
        if isinstance(fallback, str):
            return fallback.strip()
        return ""

    @staticmethod
    def _detect_speaker(text: str) -> Optional[str]:
        if not text:
            return None
        if ":" in text:
            maybe_speaker = text.split(":", 1)[0]
            if maybe_speaker.isalpha() and len(maybe_speaker) < 12:
                return maybe_speaker.upper()
        return None
