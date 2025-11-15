"""
ReAct Agent for Pokemon Emerald
================================

Implements a ReAct (Reasoning and Acting) agent that follows the pattern:
Thought -> Action -> Observation -> Thought -> ...

This agent explicitly reasons about the game state before taking actions,
making the decision process more interpretable and debuggable.
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from utils.vlm import VLM
from utils.llm_logger import LLMLogger
from utils.state_formatter import format_state_for_llm
from agent.system_prompt import system_prompt


class ActionType(Enum):
    """Possible action types in the ReAct framework."""
    PRESS_BUTTON = "press_button"
    OBSERVE = "observe"
    REMEMBER = "remember"
    PLAN = "plan"
    WAIT = "wait"


@dataclass
class Thought:
    """Represents a reasoning step."""
    content: str
    confidence: float = 0.0
    reasoning_type: str = "general"  # general, tactical, strategic, diagnostic


@dataclass
class Action:
    """Represents an action to take."""
    type: ActionType
    parameters: Dict[str, Any]
    justification: str = ""


@dataclass
class Observation:
    """Represents an observation from the environment."""
    content: str
    source: str  # game_state, memory, perception
    timestamp: float = 0.0


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    thought: Optional[Thought] = None
    action: Optional[Action] = None
    observation: Optional[Observation] = None
    step_number: int = 0


class ReActAgent:
    """
    ReAct Agent that explicitly reasons before acting.
    
    This agent maintains a history of thoughts, actions, and observations
    to make informed decisions about what to do next in the game.
    """
    
    def __init__(
        self,
        vlm_client: Optional[VLM] = None,
        max_history_length: int = 20,
        enable_reflection: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            vlm_client: Vision-language model client for reasoning
            max_history_length: Maximum number of steps to keep in history
            enable_reflection: Whether to periodically reflect on past actions
            verbose: Whether to print detailed reasoning
        """
        self.vlm_client = vlm_client or VLM()
        self.max_history_length = max_history_length
        self.enable_reflection = enable_reflection
        self.verbose = verbose
        
        self.history: List[ReActStep] = []
        self.current_step = 0
        self.current_plan: List[str] = []
        self.memory: Dict[str, Any] = {}
        
        self.llm_logger = LLMLogger()
        self.system_prompt = system_prompt
        self.max_state_chars = 3200
        self.last_action_record: Optional[Dict[str, Any]] = None
        
    def think(self, state: Dict[str, Any], screenshot: Any = None) -> Thought:
        """
        Generate a thought about the current situation.
        
        Args:
            state: Current game state
            screenshot: Current game screenshot
            
        Returns:
            A Thought object with reasoning about the situation
        """
        prompt = self._build_thought_prompt(state, screenshot)
        full_prompt = self._prepend_system_prompt(prompt)
        
        if screenshot:
            response = self.vlm_client.get_query(screenshot, full_prompt, "react")
        else:
            response = self.vlm_client.get_text_query(full_prompt, "react")
        
        self.llm_logger.log_interaction(
            interaction_type="react_think",
            prompt=full_prompt,
            response=response
        )
        
        # Parse the thought from the response
        thought = self._parse_thought(response)
        
        if self.verbose:
            print(f"==> THOUGHT: {thought.content}")
            
        return thought
    
    def act(self, thought: Thought, state: Dict[str, Any]) -> Action:
        """
        Decide on an action based on a thought and current state.
        
        Args:
            thought: The current reasoning
            state: Current game state
            
        Returns:
            An Action object describing what to do
        """
        prompt = self._build_action_prompt(thought, state)
        full_prompt = self._prepend_system_prompt(prompt)
        
        response = self.vlm_client.get_text_query(full_prompt, "react")
        
        self.llm_logger.log_interaction(
            interaction_type="react_act",
            prompt=full_prompt,
            response=response
        )
        
        # Parse the action from the response
        action = self._parse_action(response)
        
        if self.verbose:
            print(f">> ACTION: {action.type.value} - {action.parameters}")
            
        return action
    
    def observe(self, state: Dict[str, Any], action_result: Any = None) -> Observation:
        """
        Make an observation about the environment after an action.
        
        Args:
            state: Current game state after action
            action_result: Result of the previous action
            
        Returns:
            An Observation object describing what changed
        """
        # Compare with previous state if available
        changes = self._detect_changes(state)
        
        observation = Observation(
            content=self._summarize_changes(changes, state),
            source="game_state",
            timestamp=self._extract_timestamp(state)
        )
        
        if self.verbose:
            print(f"=> OBSERVATION: {observation.content}")
            
        return observation
    
    def step(self, state: Dict[str, Any], screenshot: Any = None) -> str:
        """
        Execute one complete ReAct step.
        
        Args:
            state: Current game state
            screenshot: Current game screenshot
            
        Returns:
            Button press command for the game
        """
        self._evaluate_previous_action(state)
        self.current_step += 1
        current_position = self._get_player_position_tuple(state)
        current_context = self._determine_context(state)
        current_location = self._get_player_location(state)
        
        # Think about the situation
        thought = self.think(state, screenshot)
        
        # Decide on an action
        action = self.act(thought, state)
        
        # Create step record (observation will be added after action execution)
        step = ReActStep(
            thought=thought,
            action=action,
            step_number=self.current_step
        )
        
        # Add to history
        self._add_to_history(step)
        
        # Reflect periodically
        if self.enable_reflection and self.current_step % 10 == 0:
            self._reflect_on_progress()
        
        if action.type == ActionType.PRESS_BUTTON:
            self.last_action_record = {
                "button": action.parameters.get("button"),
                "type": action.type,
                "position": current_position,
                "context": current_context,
                "location": current_location
            }
        else:
            self.last_action_record = None
        
        # Convert action to button press
        return self._action_to_button(action)
    
    def _build_thought_prompt(self, state: Dict[str, Any], screenshot: Any) -> str:
        """Build mission-focused prompt for generating thoughts."""
        recent_history = self._get_recent_history_summary()
        recent_count = min(len(self.history), 5)
        state_summary = self._format_state_for_prompt(state)
        plan_text = self._format_plan_for_prompt()
        stuck_context = self.memory.get("stuck_context", "No stuck flags")
        failed_moves = self._format_failed_moves()
        npc_notes = self._format_npc_notes()
        special_directives = self._get_special_directives(state)
        directives_text = "\n".join(f"- {directive}" for directive in special_directives) if special_directives else "None"
        
        prompt = f"""
STATE SNAPSHOT:
{state_summary}

RECENT HISTORY (last {recent_count} steps):
{recent_history}

ACTIVE PLAN / STUCK DATA:
Plan: {plan_text}
Stuck context: {stuck_context}
Failed moves @ location: {failed_moves}
NPC interactions: {npc_notes}

SPECIAL DIRECTIVES:
{directives_text}

Required reasoning:
a) Situational Assessment – identify the location, current context (overworld/battle/menu/dialogue), hazards, and any pending milestone.
b) Immediate Goal – pick the single highest-leverage objective for the next ~3 seconds and cite the evidence.
c) Action Hypothesis – describe the exact button/sequence you expect will progress the goal and what observation will confirm success.
d) Fallback – state what you will do if the hypothesis fails on the next frame.

Return exactly:
REASONING_TYPE: general|tactical|strategic|diagnostic
CONFIDENCE: 0.0-1.0
THOUGHT: situation=<...>; goal=<...>; action_hypothesis=<...>; fallback=<...>
""".strip()
        return prompt
    
    def _build_action_prompt(self, thought: Thought, state: Dict[str, Any]) -> str:
        """Build action prompt that enforces hypothesis validation and structured outputs."""
        state_highlights = self._format_state_highlights(state)
        movement_memory = self._format_failed_moves()
        npc_notes = self._format_npc_notes()
        special_directives = self._get_special_directives(state)
        directives_text = "; ".join(special_directives) if special_directives else "None"
        
        prompt = f"""
LATEST THOUGHT:
{thought.content}

STATE HIGHLIGHTS:
{state_highlights}
Movement memory: {movement_memory}
NPC context: {npc_notes}
Special directives: {directives_text}

Decision protocol:
1. Validate the thought's hypothesis against the latest frame: is the context unchanged, and has the trigger already happened?
2. Choose one of {{press_button, observe, remember, plan, wait}}. Prefer press_button with a single decisive input or a directional hold + duration when movement is safe.
3. Use observe if the frame is a transition, wait only for fade/teleport frames, remember/plan to update objectives or memory.
4. Justify the action in ≤120 characters, citing the active goal or fallback trigger.
5. Encode parameters as JSON, e.g. {{"button": "UP", "duration": 0.6}} or {{"key": "last_portal", "value": "Littleroot->Route101"}}.

Respond exactly:
ACTION_TYPE: press_button/observe/remember/plan/wait
PARAMETERS: {{...}}
JUSTIFICATION: <concise reason tied to goal or fallback>
""".strip()
        return prompt
    
    def _parse_thought(self, response: str) -> Thought:
        """Parse a thought from LLM response."""
        lines = response.strip().split('\n')
        
        reasoning_type = "general"
        confidence = 0.5
        thought_content = response
        
        for line in lines:
            line = line.strip()  # Strip whitespace from each line
            if line.startswith("REASONING_TYPE:"):
                reasoning_type = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("THOUGHT:"):
                thought_content = line.split(":", 1)[1].strip()
        
        return Thought(
            content=thought_content,
            confidence=confidence,
            reasoning_type=reasoning_type
        )
    
    def _parse_action(self, response: str) -> Action:
        """Parse an action from LLM response."""
        lines = response.strip().split('\n')
        
        action_type = ActionType.WAIT
        parameters = {}
        justification = ""
        
        for line in lines:
            line = line.strip()  # Strip whitespace from each line
            if line.startswith("ACTION_TYPE:"):
                type_str = line.split(":", 1)[1].strip()
                try:
                    action_type = ActionType(type_str)
                except:
                    action_type = ActionType.WAIT
            elif line.startswith("PARAMETERS:"):
                param_str = line.split(":", 1)[1].strip()
                try:
                    parameters = json.loads(param_str)
                except:
                    parameters = {}
            elif line.startswith("JUSTIFICATION:"):
                justification = line.split(":", 1)[1].strip()
        
        return Action(
            type=action_type,
            parameters=parameters,
            justification=justification
        )
    
    def _action_to_button(self, action: Action) -> str:
        """Convert an Action to a button press command."""
        if action.type == ActionType.PRESS_BUTTON:
            return action.parameters.get("button", "NONE")
        elif action.type == ActionType.WAIT:
            return "NONE"
        else:
            # For non-button actions, process them and return no button press
            self._process_non_button_action(action)
            return "NONE"
    
    def _process_non_button_action(self, action: Action):
        """Process actions that don't directly press buttons."""
        if action.type == ActionType.REMEMBER:
            key = action.parameters.get("key", "general")
            value = action.parameters.get("value", "")
            self.memory[key] = value
            
        elif action.type == ActionType.PLAN:
            plan = action.parameters.get("plan", [])
            if isinstance(plan, list):
                self.current_plan = plan
                
        elif action.type == ActionType.OBSERVE:
            # Just observing, no action needed
            pass
    
    def _detect_changes(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Detect what changed in the game state."""
        changes = {}
        
        if self.history:
            # Get previous state from history
            for step in reversed(self.history):
                if step.observation:
                    # Compare with previous observed state
                    # This is simplified - you'd implement actual comparison
                    changes["position_changed"] = True
                    break
        
        return changes
    
    def _summarize_changes(self, changes: Dict[str, Any], state: Dict[str, Any]) -> str:
        """Summarize what changed in a human-readable way."""
        if not changes:
            return "No significant changes observed"
        
        summary_parts = []
        if changes.get("position_changed"):
            summary_parts.append(f"Player moved to {self._stringify_position(state)}")
        
        return "; ".join(summary_parts) if summary_parts else "State updated"
    
    def _add_to_history(self, step: ReActStep):
        """Add a step to history, maintaining max length."""
        self.history.append(step)
        
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]
    
    def _get_recent_history_summary(self) -> str:
        """Get a summary of recent history for context."""
        if not self.history:
            return "No previous actions"
        
        recent = self.history[-5:]  # Last 5 steps
        summary = []
        
        for step in recent:
            if step.thought:
                summary.append(f"Step {step.step_number}: Thought: {step.thought.content[:100]}...")
            if step.action:
                summary.append(f"  Action: {step.action.type.value}")
            if step.observation:
                summary.append(f"  Observed: {step.observation.content[:100]}...")
        
        return "\n".join(summary)
    
    def _reflect_on_progress(self):
        """Periodically reflect on progress and adjust strategy."""
        if self.verbose:
            print("= REFLECTING ON PROGRESS...")
        
        reflection_prompt = f"""
Review your recent actions and their outcomes:

RECENT HISTORY:
{self._get_recent_history_summary()}

CURRENT PLAN:
{self.current_plan}

Reflect on:
1. Are you making progress toward your goals?
2. Are there any patterns in failed attempts?
3. Should you adjust your strategy?

Provide a brief reflection and any strategy adjustments.
"""
        
        response = self.vlm_client.get_text_query(reflection_prompt, "react_reflection")
        
        if self.verbose:
            print(f"=> REFLECTION: {response[:200]}...")
        
        # Store reflection in memory
        self.memory["last_reflection"] = response
        self.memory["reflection_step"] = self.current_step
    
    def _evaluate_previous_action(self, state: Dict[str, Any]):
        """Evaluate the outcome of the previous press and update failure memory."""
        if not self.last_action_record:
            return
        record = self.last_action_record
        if record.get("type") != ActionType.PRESS_BUTTON:
            self.last_action_record = None
            return
        
        button = record.get("button")
        if not button:
            self.last_action_record = None
            return
        
        previous_position = record.get("position")
        current_position = self._get_player_position_tuple(state)
        current_context = self._determine_context(state)
        previous_context = record.get("context")
        
        moved = (
            current_position is not None 
            and previous_position is not None 
            and current_position != previous_position
        )
        context_changed = current_context != previous_context
        
        if moved or context_changed:
            self._clear_failed_move(record)
            self.last_action_record = None
            return
        
        # If nothing changed, flag failure for this button at this spot
        location = self._get_player_location(state)
        self._record_failed_move(location, previous_position, button)
        self.last_action_record = None
    
    def _record_failed_move(self, location: str, position, button: str):
        """Record a failed movement or interaction attempt."""
        if position is None:
            return
        key = self._build_location_key(location, position)
        failed = self.memory.setdefault("failed_moves", {})
        entry = failed.setdefault(key, {})
        entry[button] = entry.get(button, 0) + 1
        self.memory["stuck_context"] = f"{key} blocked on {button} x{entry[button]}"
        self._trim_failed_moves()
    
    def _clear_failed_move(self, record: Dict[str, Any]):
        """Remove failure data when progress resumes."""
        failed = self.memory.get("failed_moves")
        if not failed:
            return
        position = record.get("position")
        location = record.get("location")
        button = record.get("button")
        key = self._build_location_key(location, position)
        entry = failed.get(key)
        if entry and button in entry:
            del entry[button]
            if not entry:
                failed.pop(key, None)
        if not failed:
            self.memory.pop("failed_moves", None)
            self.memory.pop("stuck_context", None)
        else:
            first_key = next(iter(failed))
            example_moves = failed[first_key]
            if isinstance(example_moves, dict) and example_moves:
                btn, count = next(iter(example_moves.items()))
                self.memory["stuck_context"] = f"{first_key} blocked on {btn} x{count}"
    
    def _build_location_key(self, location: str, position) -> str:
        """Create a readable key for the current location/position."""
        pos_text = "unknown"
        if position and isinstance(position, tuple):
            pos_text = f"({position[0]}, {position[1]})"
        elif position and isinstance(position, dict):
            pos_text = f"({position.get('x', '?')}, {position.get('y', '?')})"
        loc_text = location or "unknown"
        return f"{loc_text}@{pos_text}"
    
    def _trim_failed_moves(self, max_entries: int = 5):
        """Keep the failed moves memory small."""
        failed = self.memory.get("failed_moves")
        if not failed:
            return
        while len(failed) > max_entries:
            # Drop the oldest inserted key (dict preserves insertion order in py3.7+)
            first_key = next(iter(failed))
            failed.pop(first_key, None)

    def _prepend_system_prompt(self, body: str) -> str:
        """Attach the mission system prompt ahead of a body prompt."""
        system = self.system_prompt.strip()
        return f"{system}\n\n{body}".strip() if system else body

    def _format_plan_for_prompt(self) -> str:
        """Condense the current plan for prompt display."""
        if not self.current_plan:
            return "None"
        preview = self.current_plan[:6]
        suffix = " ..." if len(self.current_plan) > 6 else ""
        return " -> ".join(preview) + suffix

    def _format_failed_moves(self) -> str:
        """Summarize movement failures or stuck data stored in memory."""
        failed = self.memory.get("failed_moves")
        if isinstance(failed, dict) and failed:
            parts = []
            for location, moves in failed.items():
                if isinstance(moves, dict):
                    move_bits = [f"{btn}x{count}" for btn, count in list(moves.items())[:4]]
                    moves_str = "/".join(move_bits) if move_bits else "?"
                elif isinstance(moves, (list, tuple)):
                    moves_str = "/".join(str(m) for m in moves[:4])
                else:
                    moves_str = str(moves)
                parts.append(f"{location}:{moves_str}")
            return ", ".join(parts[:3])
        if isinstance(failed, (list, tuple)) and failed:
            return ", ".join(str(move) for move in failed[:5])
        if isinstance(failed, str) and failed.strip():
            return failed.strip()
        return "None recorded"

    def _format_npc_notes(self) -> str:
        """Summarize NPC interaction notes if provided."""
        notes = self.memory.get("npc_notes") or self.memory.get("npc_interactions")
        if isinstance(notes, dict) and notes:
            parts = [f"{loc}:{info}" for loc, info in list(notes.items())[:3]]
            return ", ".join(parts)
        if isinstance(notes, str) and notes.strip():
            return notes.strip()
        return "None recorded"

    def _format_state_for_prompt(self, state: Dict[str, Any]) -> str:
        """Format state data using the shared formatter with truncation safeguards."""
        if not state:
            return "State data unavailable."
        cleaned_state = self._prune_frame_from_state(state)
        summary = None
        if any(key in cleaned_state for key in ("player", "game", "map")):
            try:
                summary = format_state_for_llm(cleaned_state)
            except Exception:
                summary = None
        if not summary:
            try:
                summary = json.dumps(cleaned_state, indent=2)
            except TypeError:
                summary = str(cleaned_state)
        return self._truncate_text(summary, self.max_state_chars)

    def _format_state_highlights(self, state: Dict[str, Any]) -> str:
        """Create a compact highlight string from the state."""
        highlights = self._extract_state_highlights(state)
        if not highlights:
            return "state=unknown"
        return ", ".join(f"{key}={value}" for key, value in highlights.items())

    def _get_special_directives(self, state: Dict[str, Any]) -> List[str]:
        """Return situational directives that must override generic reasoning."""
        directives: List[str] = []
        if self._is_in_moving_vehicle(state):
            directives.append(
                "MOVING_VAN: ignore decorative box dialogue; walk RIGHT across the truck, step UP through the doorway, then press A once to exit."
            )
        if self._is_in_player_house_first_floor(state):
            directives.append(
                "PLAYER HOUSE 1F: dialogue is over—move STRAIGHT to the door, then UP onto the stairs to reach the bedroom."
            )
        if self._is_in_player_house_second_floor(state):
            directives.append(
                "PLAYER BEDROOM: quickly walk to the clock at (5,1), press 'A', then 'UP and 'A' again to set it, then exit downstairs."
            )
        if self._is_naming_screen(state):
            directives.append("NAMING SCREEN: set the player name to 'A' or 'AA' immediately and press 'START' to move to START button, press 'A' to confirm.")
        return directives

    def _extract_state_highlights(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key-value highlights for the action prompt."""
        if not isinstance(state, dict):
            state = {}
        position = self._stringify_position(state)
        location = self._get_player_location(state)
        facing = self._extract_facing(state)
        context = self._determine_context(state)
        battle_flag = self._is_battle_active(state)
        dialogue_flag = self._detect_dialogue_flag(state)
        
        return {
            "location": location,
            "coords": position,
            "facing": facing,
            "context": context,
            "battle": battle_flag,
            "dialogue_box": dialogue_flag
        }

    def _determine_context(self, state: Dict[str, Any]) -> str:
        """Infer current context (battle/menu/dialogue/overworld)."""
        if not isinstance(state, dict):
            return "unknown"
        game_section = state.get("game", {}) if isinstance(state, dict) else {}
        if isinstance(game_section, dict):
            if game_section.get("is_in_battle") or game_section.get("in_battle"):
                return "battle"
            dialogue = game_section.get("dialogue_detected", {})
            dialog_text = game_section.get("dialog_text") or state.get("dialogue")
            if isinstance(dialogue, dict) and dialogue.get("has_dialogue"):
                return "dialogue"
            if dialog_text:
                return "dialogue"
            menu_section = game_section.get("menu") or state.get("menu")
            if menu_section:
                return "menu"
            for key in ("context", "game_state", "mode"):
                value = game_section.get(key)
                if value and value not in ("dialog", "dialogue"):
                    return str(value)
        if state.get("battle_active"):
            return "battle"
        menu_section = state.get("menu")
        if menu_section:
            return "menu"
        return "overworld"

    def _is_battle_active(self, state: Dict[str, Any]) -> bool:
        """Check if the game is currently in battle mode."""
        if not isinstance(state, dict):
            return False
        game_section = state.get("game", {}) if isinstance(state, dict) else {}
        if isinstance(game_section, dict):
            if game_section.get("is_in_battle") or game_section.get("in_battle"):
                return True
        return bool(state.get("battle_active"))

    def _detect_dialogue_flag(self, state: Dict[str, Any]) -> bool:
        """Detect whether a dialogue box is visible."""
        if not isinstance(state, dict):
            return False
        game_section = state.get("game", {}) if isinstance(state, dict) else {}
        dialogue = game_section.get("dialogue_detected")
        if isinstance(dialogue, dict):
            return bool(dialogue.get("has_dialogue"))
        dialogue_text = game_section.get("dialog_text") or state.get("dialogue")
        return bool(dialogue_text)

    def _is_in_moving_vehicle(self, state: Dict[str, Any]) -> bool:
        """Detect if the player is still inside the moving truck."""
        location = self._get_player_location(state).upper()
        if any(keyword in location for keyword in ("MOVING", "TRUCK", "VAN")):
            return True
        map_info = state.get("map", {})
        if isinstance(map_info, dict):
            name = str(map_info.get("name", "")).upper()
            if any(keyword in name for keyword in ("MOVING", "TRUCK", "VAN")):
                return True
        return False

    def _is_in_player_house_first_floor(self, state: Dict[str, Any]) -> bool:
        """Detect when the agent is on the first floor of the player's house."""
        location = self._get_player_location(state).upper()
        if "HOUSE" not in location or "1F" not in location:
            return False
        return any(token in location for token in ("PLAYER", "BRENDAN", "LITTLEROOT"))

    def _is_in_player_house_second_floor(self, state: Dict[str, Any]) -> bool:
        """Detect when the agent is upstairs in the player's bedroom."""
        location = self._get_player_location(state).upper()
        if "HOUSE" not in location:
            return False
        if "2F" in location or "BEDROOM" in location:
            return True
        map_info = state.get("map", {})
        if isinstance(map_info, dict):
            name = str(map_info.get("name", "")).upper()
            if "HOUSE" in name and ("2F" in name or "BEDROOM" in name):
                return True
        return False

    def _is_naming_screen(self, state: Dict[str, Any]) -> bool:
        """Detect whether the game is currently on the naming screen."""
        if not isinstance(state, dict):
            return False
        game_section = state.get("game", {})
        if isinstance(game_section, dict):
            if game_section.get("naming_active"):
                return True
            menu = game_section.get("menu")
            if self._menu_indicates_naming(menu):
                return True
            dialog_text = str(game_section.get("dialog_text") or "").lower()
            if "your name" in dialog_text or "what's your name" in dialog_text:
                return True
        if self._menu_indicates_naming(state.get("menu")):
            return True
        return False

    def _menu_indicates_naming(self, menu_section: Optional[Dict[str, Any]]) -> bool:
        """Check if a menu section corresponds to the naming interface."""
        if not isinstance(menu_section, dict):
            return False
        menu_type = str(menu_section.get("type", "")).lower()
        if "name" in menu_type or "naming" in menu_type:
            return True
        return False

    def _get_player_location(self, state: Dict[str, Any]) -> str:
        """Return the player's current map/location name."""
        if not isinstance(state, dict):
            return "unknown"
        if "player" in state and isinstance(state["player"], dict):
            location = state["player"].get("location")
            if location:
                return str(location)
        return str(state.get("current_map") or state.get("location") or "unknown")

    def _stringify_position(self, state: Dict[str, Any]) -> str:
        """Return a human-readable player position."""
        if not isinstance(state, dict):
            return "unknown"
        position = self._get_player_position_tuple(state)
        if not position:
            return "unknown"
        return f"({position[0]}, {position[1]})"

    def _get_player_position_tuple(self, state: Dict[str, Any]):
        """Extract player coordinates from either legacy or structured state."""
        if not isinstance(state, dict):
            return None
        player_section = state.get("player") if isinstance(state, dict) else None
        if isinstance(player_section, dict):
            for key in ("position", "coordinates"):
                value = player_section.get(key)
                if isinstance(value, dict) and "x" in value and "y" in value:
                    return (value["x"], value["y"])
        for key in ("player_position", "position"):
            value = state.get(key)
            if isinstance(value, dict) and "x" in value and "y" in value:
                return (value["x"], value["y"])
        return None

    def _extract_facing(self, state: Dict[str, Any]) -> str:
        """Extract player's facing direction if available."""
        if not isinstance(state, dict):
            return "unknown"
        player_section = state.get("player") if isinstance(state, dict) else None
        if isinstance(player_section, dict) and player_section.get("facing"):
            return str(player_section["facing"])
        return str(state.get("facing") or "unknown")

    def _extract_timestamp(self, state: Dict[str, Any]) -> float:
        """Extract timestamp information from the state."""
        if not isinstance(state, dict):
            return 0.0
        if "timestamp" in state:
            return float(state.get("timestamp") or 0.0)
        game_section = state.get("game", {})
        if isinstance(game_section, dict):
            ts = game_section.get("timestamp") or game_section.get("time_seconds")
            if ts is not None:
                return float(ts)
        return 0.0

    def _prune_frame_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Remove non-serializable frame data before formatting."""
        if not isinstance(state, dict):
            return state
        pruned = dict(state)
        pruned.pop("frame", None)
        return pruned

    def _truncate_text(self, text: str, limit: int) -> str:
        """Safely truncate long prompt sections."""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]..."


# Convenience function for integration with existing codebase
def create_react_agent(**kwargs) -> ReActAgent:
    """Create a ReAct agent with default settings."""
    return ReActAgent(**kwargs)
