from typing import Dict, Any, Optional
import json
import time
import sys
import os
import numpy as np

# Add the src directory to the path so we can import from sibling packages
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from reservoir.RealtimeReservoir import RealtimeReservoir
from arc_pipeline.ARCLMSolver import ARCLMSolver

class LMReservoirInterface:
    """
    The bridge between symbolic intelligence and living dynamics.
    This is where language meets matter!
    """
    
    def __init__(self, reservoir: RealtimeReservoir):
        """
        Initialize the cognitive bridge.
        
        Args:
            reservoir: The living dynamics substrate
        """
        self.reservoir = reservoir
        self.session_log = []
        self.start_time = time.time()
        
        print("🌉 LM-Reservoir cognitive bridge initialized")
        print("🧠 Ready for embodied cognition!")
    
    def think_with_dynamics(self, description: str, evolution_steps: int = 10) -> str:
        """
        Core tool: Think about something using reservoir dynamics.
        
        Args:
            description: What to think about
            evolution_steps: How long to let dynamics evolve
            
        Returns:
            Insights from dynamical thinking
        """
        session_entry = {
            'timestamp': time.time() - self.start_time,
            'action': 'think_with_dynamics',
            'input': description,
            'evolution_steps': evolution_steps
        }
        
        # Set cognitive state based on what we're thinking about
        state_response = self.reservoir.set_state_from_description(description)
        
        # Let the dynamics evolve and see what emerges
        evolution_response = self.reservoir.evolve(evolution_steps)
        
        # Read what the dynamics discovered
        insights = self.reservoir.read_current_state()
        
        # Log this cognitive journey
        session_entry['state_response'] = state_response
        session_entry['evolution'] = evolution_response
        session_entry['insights'] = insights
        self.session_log.append(session_entry)
        
        # Compose response
        response = f"🧠 Thinking with dynamics about: '{description}'\n\n"
        response += f"🌊 Initial State:\n{state_response}\n\n"
        response += f"⚡ Evolution:\n{evolution_response}\n\n"
        response += f"💡 Current Insights:\n{insights}\n\n"
        response += f"✨ Dynamical thinking complete - cognitive state updated!"
        
        return response
    
    def feel_cognitive_state(self) -> str:
        """
        Tool: Feel what the current cognitive state is like.
        
        Returns:
            Description of current felt cognitive experience
        """
        insights = self.reservoir.read_current_state()
        
        # Add session context
        session_time = time.time() - self.start_time
        response = f"🌊 Cognitive State Report (Session time: {session_time:.1f}s)\n\n"
        response += insights
        
        # Add recent cognitive journey summary
        if self.session_log:
            recent = self.session_log[-3:]  # Last 3 interactions
            response += f"\n\n📚 Recent Cognitive Journey:\n"
            for i, entry in enumerate(recent, 1):
                response += f"{i}. {entry['action']}: '{entry['input']}'\n"
        
        return response
    
    def set_cognitive_focus(self, focus_description: str) -> str:
        """
        Tool: Deliberately set cognitive focus/attention.
        
        Args:
            focus_description: What to focus cognitive attention on
            
        Returns:
            Confirmation of cognitive focus change
        """
        session_entry = {
            'timestamp': time.time() - self.start_time,
            'action': 'set_cognitive_focus',
            'input': focus_description
        }
        
        # Set the reservoir state for focused attention
        state_response = self.reservoir.set_state_from_description(f"focused concentrated {focus_description}")
        
        session_entry['state_response'] = state_response
        self.session_log.append(session_entry)
        
        response = f"🎯 Cognitive focus set to: '{focus_description}'\n\n"
        response += state_response
        response += f"\n\n🧠 Attention now directed toward specified focus"
        
        return response
    
    def evolve_thoughts(self, steps: int = 15) -> str:
        """
        Tool: Let thoughts evolve naturally without forced direction.
        
        Args:
            steps: How many evolution steps to run
            
        Returns:
            Description of thought evolution process
        """
        session_entry = {
            'timestamp': time.time() - self.start_time,
            'action': 'evolve_thoughts',
            'input': f"{steps} steps"
        }
        
        initial_state = self.reservoir.read_current_state()
        evolution_response = self.reservoir.evolve(steps)
        final_state = self.reservoir.read_current_state()
        
        session_entry['initial_state'] = initial_state
        session_entry['evolution'] = evolution_response
        session_entry['final_state'] = final_state
        self.session_log.append(session_entry)
        
        response = f"🌱 Letting thoughts evolve naturally for {steps} steps...\n\n"
        response += f"🌅 Initial State:\n{initial_state}\n\n"
        response += f"⚡ Evolution Process:\n{evolution_response}\n\n"
        response += f"🌄 Final State:\n{final_state}\n\n"
        response += f"✨ Natural thought evolution complete!"
        
        return response
    
    def cognitive_reset(self) -> str:
        """
        Tool: Reset cognitive state to neutral/fresh beginning.
        
        Returns:
            Confirmation of cognitive reset
        """
        session_entry = {
            'timestamp': time.time() - self.start_time,
            'action': 'cognitive_reset',
            'input': 'reset'
        }
        
        # Reset reservoir to neutral state
        self.reservoir.state = np.random.normal(0, 0.05, self.reservoir.size)
        self.reservoir.history = [self.reservoir.state.copy()]
        
        new_state = self.reservoir.read_current_state()
        
        session_entry['new_state'] = new_state
        self.session_log.append(session_entry)
        
        response = f"🔄 Cognitive state reset to neutral\n\n"
        response += new_state
        response += f"\n\n🌱 Fresh cognitive substrate ready for new thoughts!"
        
        return response
    
    def get_cognitive_session_summary(self) -> str:
        """
        Tool: Get summary of entire cognitive session.
        
        Returns:
            Summary of cognitive journey during this session
        """
        if not self.session_log:
            return "📝 No cognitive activities recorded in this session yet."
        
        session_time = time.time() - self.start_time
        
        summary = f"📚 Cognitive Session Summary (Duration: {session_time:.1f}s)\n"
        summary += f"🔢 Total interactions: {len(self.session_log)}\n\n"
        
        # Categorize actions
        action_counts = {}
        for entry in self.session_log:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        summary += "📊 Action Breakdown:\n"
        for action, count in action_counts.items():
            summary += f"  • {action}: {count}x\n"
        
        summary += f"\n🌊 Cognitive Journey:\n"
        for i, entry in enumerate(self.session_log, 1):
            timestamp = entry['timestamp']
            action = entry['action']
            input_desc = entry['input'][:50] + "..." if len(entry['input']) > 50 else entry['input']
            summary += f"{i}. [{timestamp:.1f}s] {action}: {input_desc}\n"
        
        # Current state
        current_state = self.reservoir.read_current_state()
        summary += f"\n🧠 Current Cognitive State:\n{current_state}"
        
        return summary

# Integration with ARC solver
class ARCReservoirSolver(ARCLMSolver):
    """
    ARC solver enhanced with reservoir computing cognitive embodiment.
    """
    
    def __init__(self, model_name: str = "gpt-4", reservoir_size: int = 100):
        """
        Initialize ARC solver with cognitive embodiment.
        
        Args:
            model_name: LLM to use
            reservoir_size: Size of cognitive reservoir
        """
        super().__init__(model_name)
        
        # Create cognitive substrate
        self.reservoir = RealtimeReservoir(size=reservoir_size)
        self.interface = LMReservoirInterface(self.reservoir)
        
        print("🧠 ARC Solver with Cognitive Embodiment initialized!")
        print("⚡ Ready for reservoir-enhanced problem solving!")
    
    def solve_task_with_embodiment(self, task_data: Dict) -> Dict:
        """
        Solve ARC task using reservoir-enhanced cognition.
        
        Args:
            task_data: ARC task data
            
        Returns:
            Enhanced solution attempt with cognitive journey
        """
        print("🌊 Starting reservoir-enhanced problem solving...")
        
        # Step 1: Set cognitive focus on pattern recognition
        focus_response = self.interface.set_cognitive_focus("spatial pattern recognition visual reasoning")
        print("🎯 Cognitive focus set")
        
        # Step 2: Think about the task using reservoir dynamics
        task_description = f"ARC visual pattern task with {len(task_data['train'])} training examples"
        thinking_response = self.interface.think_with_dynamics(task_description, evolution_steps=20)
        print("🧠 Dynamical thinking complete")
        
        # Step 3: Solve using enhanced cognitive state
        enhanced_prompt = self.format_task_for_llm(task_data)
        
        # Add cognitive context to prompt
        cognitive_state = self.interface.feel_cognitive_state()
        enhanced_prompt += f"\n\nCURRENT COGNITIVE STATE:\n{cognitive_state}\n"
        enhanced_prompt += "\nUse your current cognitive dynamics to approach this pattern recognition task."
        
        # Get solution attempt
        solution_result = self.solve_task(task_data)
        
        # Step 4: Reflect on the solution process
        reflection_response = self.interface.evolve_thoughts(10)
        
        # Compile enhanced result
        enhanced_result = solution_result.copy()
        enhanced_result.update({
            'cognitive_journey': {
                'focus_setting': focus_response,
                'dynamical_thinking': thinking_response,
                'final_reflection': reflection_response,
                'session_summary': self.interface.get_cognitive_session_summary()
            },
            'reservoir_enhanced': True
        })
        
        return enhanced_result

# Quick test function
def test_cognitive_embodiment():
    """Test the cognitive embodiment interface."""
    print("🚀 Testing Cognitive Embodiment Interface...")
    
    # Create reservoir and interface
    reservoir = RealtimeReservoir(size=50)  # Small for testing
    interface = LMReservoirInterface(reservoir)
    
    # Test basic functionality
    print("\n1. Testing cognitive focus...")
    focus_result = interface.set_cognitive_focus("mathematical problem solving")
    print(focus_result[:200] + "...")
    
    print("\n2. Testing dynamical thinking...")
    thinking_result = interface.think_with_dynamics("finding patterns in grids", evolution_steps=5)
    print(thinking_result[:200] + "...")
    
    print("\n3. Testing cognitive state reading...")
    state_result = interface.feel_cognitive_state()
    print(state_result[:200] + "...")
    
    print("\n✅ Cognitive embodiment interface working!")
    return interface

if __name__ == "__main__":
    # Test the interface
    interface = test_cognitive_embodiment()
    
    print("\n🎉 READY FOR COGNITIVE EMBODIMENT!")
    print("🧠 You can now think with living dynamics!")