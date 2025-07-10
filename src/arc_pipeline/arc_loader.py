import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class ARCLoader:
    """Load and process ARC dataset with support for different versions."""
    
    def __init__(self, data_path: str, arc_version: int = 1):
        """
        Initialize ARC loader.
        
        Args:
            data_path: Path to ARC data directory
            arc_version: 1 for ARC-AGI 1, 2 for ARC-AGI 2 (future)
        """
        self.data_path = Path(data_path)
        self.arc_version = arc_version
        self.tasks = {}
        
        if arc_version == 2:
            raise NotImplementedError("ARC 2 support coming soon!")
    
    def load_tasks(self, split: str = "training") -> Dict:
        """
        Load all tasks from specified split directory.
        
        Args:
            split: "training" or "evaluation" 
            
        Returns:
            Dictionary of task_id -> task_data
        """
        split_path = self.data_path / split
        
        if not split_path.exists():
            raise FileNotFoundError(f"Could not find directory {split_path}")
            
        if not split_path.is_dir():
            raise NotADirectoryError(f"{split_path} is not a directory")
        
        tasks = {}
        json_files = list(split_path.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {split_path}")
        
        for json_file in json_files:
            task_id = json_file.stem  # filename without .json extension
            
            try:
                with open(json_file, 'r') as f:
                    task_data = json.load(f)
                    tasks[task_id] = task_data
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
                continue
        
        print(f"Loaded {len(tasks)} tasks from {split}")
        self.tasks = tasks
        return tasks
    
    def get_task(self, task_id: str) -> Dict:
        """Get specific task by ID."""
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found. Available: {list(self.tasks.keys())[:5]}...")
        return self.tasks[task_id]
    
    def visualize_grid(self, grid: List[List[int]], title: str = "Grid"):
        """
        Visualize a single grid with proper colors.
        
        Args:
            grid: 2D list of integers (0-9 representing colors)
            title: Title for the plot
        """
        grid_array = np.array(grid)
        
        # ARC color palette (approximate)
        colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
                  '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Create custom colormap
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors[:10])
        
        im = ax.imshow(grid_array, cmap=cmap, vmin=0, vmax=9)
        
        # Add grid lines
        ax.set_xticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)
        
        # Remove tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_title(title)
        plt.tight_layout()
        plt.show()
    
    def visualize_task(self, task_id: str):
        """
        Visualize complete task with all examples.
        
        Args:
            task_id: ID of task to visualize
        """
        task = self.get_task(task_id)
        
        train_examples = task['train']
        test_examples = task['test']
        
        total_examples = len(train_examples) + len(test_examples)
        
        fig, axes = plt.subplots(2, total_examples * 2, 
                                figsize=(4 * total_examples, 8))
        
        if total_examples == 1:
            axes = axes.reshape(2, 2)
        
        # Plot training examples
        for i, example in enumerate(train_examples):
            # Input
            ax_input = axes[0, i * 2] if total_examples > 1 else axes[0, 0]
            ax_input.imshow(np.array(example['input']), cmap='tab10', vmin=0, vmax=9)
            ax_input.set_title(f'Train {i+1} Input')
            ax_input.set_xticks([])
            ax_input.set_yticks([])
            
            # Output
            ax_output = axes[0, i * 2 + 1] if total_examples > 1 else axes[0, 1]
            ax_output.imshow(np.array(example['output']), cmap='tab10', vmin=0, vmax=9)
            ax_output.set_title(f'Train {i+1} Output')
            ax_output.set_xticks([])
            ax_output.set_yticks([])
        
        # Plot test examples
        for i, example in enumerate(test_examples):
            start_col = len(train_examples) * 2 + i * 2
            
            # Input
            ax_input = axes[1, start_col] if total_examples > 1 else axes[1, 0]
            ax_input.imshow(np.array(example['input']), cmap='tab10', vmin=0, vmax=9)
            ax_input.set_title(f'Test {i+1} Input')
            ax_input.set_xticks([])
            ax_input.set_yticks([])
            
            # Output (if available)
            if 'output' in example:
                ax_output = axes[1, start_col + 1] if total_examples > 1 else axes[1, 1]
                ax_output.imshow(np.array(example['output']), cmap='tab10', vmin=0, vmax=9)
                ax_output.set_title(f'Test {i+1} Output')
                ax_output.set_xticks([])
                ax_output.set_yticks([])
            
        plt.suptitle(f'Task: {task_id}')
        plt.tight_layout()
        plt.show()
    
    def get_dataset_stats(self) -> Dict:
        """Get basic statistics about the loaded dataset."""
        if not self.tasks:
            print("No tasks loaded. Call load_tasks() first.")
            return {}
        
        stats = {
            'total_tasks': len(self.tasks),
            'grid_sizes': [],
            'color_usage': {},
            'train_examples_per_task': [],
            'test_examples_per_task': []
        }
        
        for task_id, task in self.tasks.items():
            # Count examples
            stats['train_examples_per_task'].append(len(task['train']))
            stats['test_examples_per_task'].append(len(task['test']))
            
            # Analyze all grids in task
            for example in task['train'] + task['test']:
                for grid_type in ['input', 'output']:
                    if grid_type in example:
                        grid = example[grid_type]
                        h, w = len(grid), len(grid[0])
                        stats['grid_sizes'].append((h, w))
                        
                        # Count colors
                        for row in grid:
                            for color in row:
                                stats['color_usage'][color] = stats['color_usage'].get(color, 0) + 1
        
        return stats


# Convenience function for quick exploration
def quick_explore(data_path: str):
    """Quick dataset exploration."""
    loader = ARCLoader(data_path)
    tasks = loader.load_tasks("training")
    
    # Show first task
    first_task_id = list(tasks.keys())[0]
    print(f"Visualizing first task: {first_task_id}")
    loader.visualize_task(first_task_id)
    
    # Show stats
    stats = loader.get_dataset_stats()
    print(f"\nDataset Statistics:")
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Grid sizes range: {min(stats['grid_sizes'])} to {max(stats['grid_sizes'])}")
    print(f"Most common colors: {sorted(stats['color_usage'].items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    return loader

# if __name__ == "__main__":
