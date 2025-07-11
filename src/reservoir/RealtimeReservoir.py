from dataclasses import dataclass
import enum
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class ReservoirState:
    """Snapshot of reservoir consciousness at a moment in time."""
    nodes: np.ndarray
    energy: float
    dominant_frequency: float
    coherence: float
    timestamp: float

class SynchronizationMode(enum.Enum):
    """Different ways the LM can sync with reservoir dynamics."""
    HEARTBEAT = "heartbeat" # Regular sampling rhythm
    RESONANCE = "resonance" # Match reservoir's natural frequency
    ENTRAINMENT = "entrainment" # Gradually sync rhythms
    PHASE_LOCK = "phase_lock" # Lock to specific phase relationships

class RealtimeReservoir:
    """The LM's extended dynamical nervous system."""

    def __init__(self, size: int = 100,
                 connectivity: float = 0.1,
                 leak_rate: float = 0.3,
                 spectral_radius: float = 0.95):
        """Initialize the cognitive substrate.
        
        Args:
            size: Number of reservoir nodes (neurons)
            connectivity: Fraction of connections (sparsity)
            leak_rate: How quickly states decay (memory vs adaptability)
            spectral_radius: Edge of chaos parameter (stability vs complexity)
        """
        self.size = size
        self.connectivity = connectivity
        self.leak_rate = leak_rate
        self.spectral_radius = spectral_radius

        # The living matrix - sparse random connections
        # We have some work to do before it's living in the traditional sense but YEET
        self.W = self._create_reservoir_matrix()

        self.state = np.random.uniform(-1, 1, size)

        self.history = []
        self.time = 0.0
        self.heartbeat_interval = 1.0 # seconds
        self.last_heartbeat = time.time()

        # Synchronization parameters
        self.sync_mode = SynchronizationMode.HEARTBEAT
        self.phase_offset = 0.0

        print(f"🧠 Reservoir consciousness initialized with {size} nodes")
        print(f"⚡ Operating at edge of chaos (ρ={spectral_radius})")
        

    def set_state(self, description: str) -> str:
        """LM steers the reservoir dynamics."""
        # Convert description to state vector
        # Let LM control its own cognitive substrate!

    def evolve(self, steps: int = 10) -> str:
        """Let the chaos think."""
        # Run dynamics, capture trajectory -- reminds Ben of time series prediction paper
        # Return what happened during evolution

    def read_current_state(self) -> str:
        """What patterns emerged during evolution"""
        # Convert state to interpretable description

    def get_trajectory_insights(self) -> str:
        """What patterns emerged during evolution?"""
        # Analyze recent trajectory for insights

    # perhaps multimodal feedback?
    #speed optimizations? 
    #deeper architecture real-time synchronizations?
    #Synchronization map math on the feedback reservoir?


    def _create_reservoir_matrix(self) -> np.ndarray:
        """
        Create the living matrix - sparse random connections at edge of chaos.
        
        Returns:
            Reservoir weight matrix with proper spectral properties
        """
        # Create sparse random matrix
        W = np.random.uniform(-0.5, 0.5, (self.size, self.size))
        
        # Apply sparsity (most connections are zero)
        mask = np.random.random((self.size, self.size)) < self.connectivity
        W = W * mask
        
        # Scale to desired spectral radius (edge of chaos)
        eigenvalues = np.linalg.eigvals(W)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W = W * (self.spectral_radius / current_radius)
        
        return W

    def _compute_energy(self) -> float:
        """Compute total energy in the reservoir system."""
        return np.sum(self.state ** 2)

    def _compute_coherence(self) -> float:
        """
        Compute coherence - how organized vs chaotic the dynamics are.
        
        Returns:
            Coherence value between 0 (chaos) and 1 (perfect order)
        """
        if len(self.history) < 2:
            return 0.0
        
        # Measure stability of recent dynamics
        recent_states = np.array(self.history[-10:]) if len(self.history) >= 10 else np.array(self.history)
        state_variations = np.std(recent_states, axis=0)
        mean_variation = np.mean(state_variations)
        
        # Coherence is inverse of variation (more stable = more coherent)
        coherence = 1.0 / (1.0 + mean_variation)
        return min(coherence, 1.0)

    def _find_dominant_frequency(self) -> float:
        """
        Find the dominant oscillation frequency in recent dynamics.
        
        Returns:
            Dominant frequency in Hz (cycles per time unit)
        """
        if len(self.history) < 10:
            return 0.0
        
        # Take recent history and compute FFT
        recent_energy = [np.sum(state**2) for state in self.history[-50:]]
        if len(recent_energy) < 10:
            return 0.0
        
        # Simple frequency detection via autocorrelation peak
        autocorr = np.correlate(recent_energy, recent_energy, mode='full')
        center = len(autocorr) // 2
        
        # Find first significant peak after center (ignoring zero-lag)
        peaks = []
        for i in range(center + 2, min(center + 20, len(autocorr))):
            if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks.append(i - center)
        
        if peaks:
            period = peaks[0]  # First peak gives dominant period
            frequency = 1.0 / period if period > 0 else 0.0
            return frequency
        
        return 0.0

    def evolve(self, steps: int = 10, input_signal: Optional[np.ndarray] = None) -> str:
        """
        Let the reservoir think - evolve dynamics over time.
        
        Args:
            steps: Number of evolution steps
            input_signal: Optional input to inject during evolution
            
        Returns:
            Description of what happened during evolution
        """
        start_energy = self._compute_energy()
        start_coherence = self._compute_coherence()
        
        evolution_log = []
        
        for step in range(steps):
            # Prepare input for this step
            if input_signal is not None:
                if len(input_signal.shape) == 1:
                    # Single input vector - use for all steps
                    current_input = input_signal
                else:
                    # Sequence of inputs - use appropriate step
                    input_idx = min(step, input_signal.shape[0] - 1)
                    current_input = input_signal[input_idx]
            else:
                current_input = np.zeros(min(10, self.size))  # Default minimal input
            
            # Reservoir dynamics equation with input
            if len(current_input) < self.size:
                # Pad input to reservoir size
                padded_input = np.zeros(self.size)
                padded_input[:len(current_input)] = current_input
                current_input = padded_input
            elif len(current_input) > self.size:
                # Truncate input to reservoir size
                current_input = current_input[:self.size]
            
            # Core dynamics: leaky integration with nonlinear reservoir mixing
            new_state = (1 - self.leak_rate) * self.state + \
                    self.leak_rate * np.tanh(self.W @ self.state + 0.1 * current_input)
            
            # Add tiny bit of noise for exploration
            new_state += np.random.normal(0, 0.001, self.size)
            
            self.state = new_state
            self.time += 0.1  # Each step is 0.1 time units
            
            # Record this moment in history
            self.history.append(self.state.copy())
            
            # Log significant changes
            energy = self._compute_energy()
            if abs(energy - start_energy) > 0.1:
                evolution_log.append(f"Step {step}: Energy shift {energy:.3f}")
        
        # Analyze what happened
        end_energy = self._compute_energy()
        end_coherence = self._compute_coherence()
        
        # Create narrative description
        energy_change = end_energy - start_energy
        coherence_change = end_coherence - start_coherence
        
        narrative = f"🌊 Evolved for {steps} steps:\n"
        
        if abs(energy_change) > 0.05:
            if energy_change > 0:
                narrative += f"⚡ Energy increased by {energy_change:.3f} (dynamics amplifying)\n"
            else:
                narrative += f"🌀 Energy decreased by {-energy_change:.3f} (dynamics settling)\n"
        else:
            narrative += "💫 Energy remained stable (steady state)\n"
        
        if coherence_change > 0.1:
            narrative += f"🎯 Coherence increased by {coherence_change:.3f} (patterns organizing)\n"
        elif coherence_change < -0.1:
            narrative += f"🌪️ Coherence decreased by {-coherence_change:.3f} (exploring chaos)\n"
        else:
            narrative += "🔄 Coherence stable (balanced dynamics)\n"
        
        # Detect interesting patterns
        dominant_freq = self._find_dominant_frequency()
        if dominant_freq > 0.1:
            narrative += f"🎵 Dominant rhythm: {dominant_freq:.2f} Hz\n"
        
        if evolution_log:
            narrative += "📊 Notable events:\n" + "\n".join(evolution_log[-3:])  # Last 3 events
        
        return narrative

    def set_state_from_description(self, description: str) -> str:
        """
        Set reservoir state based on natural language description.
        This is where language meets dynamics!
        
        Args:
            description: Natural language description of desired cognitive state
            
        Returns:
            Confirmation of state change
        """
        # Simple mapping from words to dynamics patterns
        # In reality, this would use LM embeddings
        
        desc_lower = description.lower()
        
        # Initialize base state
        new_state = np.random.normal(0, 0.1, self.size)
        
        # Pattern recognition for common cognitive states
        if any(word in desc_lower for word in ['calm', 'peaceful', 'stable']):
            # Low energy, high coherence state
            new_state = np.random.normal(0, 0.2, self.size)
            
        elif any(word in desc_lower for word in ['excited', 'energetic', 'dynamic']):
            # High energy state with some structure
            new_state = np.random.normal(0, 0.8, self.size)
            
        elif any(word in desc_lower for word in ['creative', 'exploring', 'curious']):
            # Medium energy with some chaos for exploration
            new_state = np.random.uniform(-0.5, 0.5, self.size)
            
        elif any(word in desc_lower for word in ['focused', 'concentrated', 'sharp']):
            # Sparse activation with high intensity
            new_state = np.zeros(self.size)
            n_active = max(1, self.size // 10)  # 10% of nodes active
            active_indices = np.random.choice(self.size, n_active, replace=False)
            new_state[active_indices] = np.random.normal(0, 1.0, n_active)
            
        elif any(word in desc_lower for word in ['pattern', 'grid', 'spatial']):
            # Structured spatial patterns
            # Create simple wave patterns
            x = np.linspace(0, 4*np.pi, self.size)
            new_state = 0.3 * np.sin(x) + 0.2 * np.sin(2*x) + np.random.normal(0, 0.1, self.size)
        
        # Apply the new state
        self.state = new_state
        self.time += 0.01
        self.history.append(self.state.copy())
        
        # Analyze what we created
        energy = self._compute_energy()
        coherence = self._compute_coherence()
        
        response = f"🧠 Cognitive state set to: '{description}'\n"
        response += f"⚡ Energy level: {energy:.3f}\n"
        response += f"🎯 Coherence: {coherence:.3f}\n"
        response += f"🌊 Reservoir now embodying requested dynamics"
        
        return response

    def read_current_state(self) -> str:
        """
        Read and interpret current reservoir state.
        This is where dynamics become language!
        
        Returns:
            Natural language description of current cognitive state
        """
        if len(self.state) == 0:
            return "🤷 Reservoir is empty - no cognitive state to read"
        
        # Analyze current state properties
        energy = self._compute_energy()
        coherence = self._compute_coherence()
        dominant_freq = self._find_dominant_frequency()
        
        # Statistical properties
        mean_activation = np.mean(self.state)
        std_activation = np.std(self.state)
        max_activation = np.max(np.abs(self.state))
        active_nodes = np.sum(np.abs(self.state) > 0.1)
        
        # Construct interpretation
        description = "🧠 Current cognitive state:\n"
        
        # Energy interpretation
        if energy < 0.1:
            description += "💤 Very low energy - dormant/resting state\n"
        elif energy < 0.5:
            description += "🌅 Low energy - calm, contemplative state\n"
        elif energy < 1.5:
            description += "⚡ Moderate energy - actively processing\n"
        elif energy < 3.0:
            description += "🔥 High energy - intense cognitive activity\n"
        else:
            description += "🌪️ Very high energy - chaotic/exploring state\n"
        
        # Coherence interpretation
        if coherence > 0.8:
            description += "🎯 High coherence - organized, focused thinking\n"
        elif coherence > 0.5:
            description += "🔄 Moderate coherence - balanced structure/flexibility\n"
        elif coherence > 0.2:
            description += "🌊 Low coherence - creative, exploratory mode\n"
        else:
            description += "🌀 Very low coherence - chaotic, unstructured\n"
        
        # Activation pattern
        activation_density = active_nodes / self.size
        if activation_density > 0.8:
            description += "🌐 Distributed activation - broad, parallel processing\n"
        elif activation_density > 0.3:
            description += "🎭 Moderate activation - selective processing\n"
        else:
            description += "🎯 Sparse activation - highly focused processing\n"
        
        # Rhythmic patterns
        if dominant_freq > 0.1:
            description += f"🎵 Rhythmic dynamics at {dominant_freq:.2f} Hz\n"
        else:
            description += "〰️ No clear rhythmic patterns\n"
        
        # Overall interpretation
        if energy > 1.0 and coherence > 0.6:
            description += "✨ Assessment: Highly active and organized - peak cognitive state"
        elif energy < 0.3 and coherence > 0.7:
            description += "🧘 Assessment: Calm and structured - meditative state"
        elif energy > 1.0 and coherence < 0.3:
            description += "🎨 Assessment: Chaotic and energetic - creative exploration"
        elif energy < 0.5 and coherence < 0.4:
            description += "😴 Assessment: Low energy and disorganized - need stimulation"
        else:
            description += "⚖️ Assessment: Balanced cognitive state"
        
        return description