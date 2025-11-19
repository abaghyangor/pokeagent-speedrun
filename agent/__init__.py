"""
Agent modules for Pokemon Emerald speedrunning agent
"""

from utils.vlm import VLM
from .simple import (
    SimpleAgent,
    get_simple_agent,
    simple_mode_processing_multiprocess,
    configure_simple_agent_defaults,
)
from .react import ReActAgent, create_react_agent


class Agent:
    """
    Unified agent interface that encapsulates all agent logic.
    The client just calls agent.step(game_state) and gets back an action.
    """
    
    def __init__(self, args=None):
        """
        Initialize the agent based on configuration.

        Args:
            args: Command line arguments with agent configuration
        """
        # Extract configuration
        backend = args.backend if args else "gemini"
        model_name = args.model_name if args else "gemini-2.5-flash"

        # Handle scaffold selection (with backward compatibility for --simple)
        if args and hasattr(args, 'scaffold'):
            scaffold = args.scaffold
        elif args and hasattr(args, 'simple') and args.simple:
            scaffold = "simple"
        else:
            scaffold = "fourmodule"

        # Prepare VLM kwargs
        vlm_kwargs = {}
        if args and hasattr(args, 'vertex_id') and args.vertex_id:
            vlm_kwargs['vertex_id'] = args.vertex_id

        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name, **vlm_kwargs)
        print(f"   VLM: {backend}/{model_name}")
        
        # Initialize agent based on scaffold
        self.scaffold = scaffold
        if scaffold == "simple":
            self.agent_impl = get_simple_agent(self.vlm)
            print("   Scaffold: Simple (direct frame->action)")
        else:
            # Collapse both legacy 'react' and 'fourmodule' scaffolds into the
            # new agentic reasoning pipeline.
            self.scaffold = "react"
            self.agent_impl = create_react_agent(vlm_client=self.vlm, verbose=getattr(args, "verbose", False))
            print("   Scaffold: Agentic (Perception→Dialogue→Objectives→Action)")
    
    def step(self, game_state):
        """
        Process a game state and return an action.
        
        Args:
            game_state: Dictionary containing:
                - screenshot: PIL Image
                - game_state: Dict with game memory data
                - visual: Dict with visual observations
                - audio: Dict with audio observations
                - progress: Dict with milestone progress
        
        Returns:
            dict: Contains 'action' and optionally 'reasoning'
        """
        if self.scaffold == "simple":
            return self.agent_impl.step(game_state)

        # Agentic scaffold expects the full structured state
        screenshot = game_state.get('frame') if isinstance(game_state, dict) else None
        state = game_state if isinstance(game_state, dict) else {}
        button = self.agent_impl.step(state, screenshot)
        return {'action': button, 'reasoning': 'Agentic pipeline decision'}


__all__ = [
    'Agent',
    'SimpleAgent',
    'get_simple_agent',
    'simple_mode_processing_multiprocess',
    'configure_simple_agent_defaults',
    'ReActAgent',
    'create_react_agent'
]
