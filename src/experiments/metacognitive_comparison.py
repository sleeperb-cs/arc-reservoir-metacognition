import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from reservoir.RealtimeReservoir import RealtimeReservoir
from arc_pipeline.ARCReservoirSolver import ARCReservoirSolver
from arc_pipeline

def run_metacognitive_expansion_test(data_path: str, max_tasks: int = 3):
    """Test if reservoir interface enhances ARC performance."""
    
    # Load ARC tasks
    from arc_pipeline.arc_loader import ARCLoader
    loader = ARCLoader(data_path)
    tasks = loader.load_tasks("training")
    
    # Test baseline vs enhanced
    baseline_solver = ARCLMSolver()
    enhanced_solver = ARCReservoirSolver()
    
    print("🧪 Testing Metacognitive Expansion on ARC Tasks...")
    
    # Compare performance
    baseline_results = baseline_solver.evaluate_on_tasks(tasks, max_tasks)
    enhanced_results = enhanced_solver.evaluate_on_tasks(tasks, max_tasks) # Need this method
    
    print(f"📊 Baseline accuracy: {baseline_results['summary']['accuracy']:.2%}")
    print(f"📊 Enhanced accuracy: {enhanced_results['summary']['accuracy']:.2%}")