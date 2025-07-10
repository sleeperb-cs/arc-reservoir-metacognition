import json
import openai
from typing import Dict, List, Optional, Tuple
import re
import sys
import os
from arc_loader import ARCLoader

class ARCLMSolver:
    """Solve ARC tasks using Language Models"""

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize ARCLMSolver.

        Args:
            model_name: OpenAI model to use
            api_key: OpenAI API key 
        """
        self.model_name = model_name
        if api_key:
            openai.api_key = os.environ['OPENAI_API_KEY']

        # ARC color mapping for cleaner descriptions
        self.color_names = {
            0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow", 5: "grey", 6: "fuschia", 7: "orange", 8: "teal", 9: "brown"
        }

    def grid_to_string(self, grid: List[List[int]], use_colors: bool = True) -> str:
        """
        Convert grid to string representation.

        Args:
            grid: 2D list of integers
            use_colors: Whether to use color names instead of numbers

        Returns:
            String representation of the grid
        """
        if use_colors:
            # Create color-based representation
            rows = []
            for row in grid:
                row_str = " ".join({self.color_names[cell] for cell in row})
                rows.append(row_str)
            return "\n".join(rows)
        else:
            # Create number-based representation
            rows = []
            for row in grid:
                    row_str = " ".join([str(cell) for cell in row])
                    rows.append(row_str)
            return "\n".join(rows)
    def format_task_for_llm(self, task_data: dict, include_solution: bool = False) -> str:
        """
        Format ARC task for LLM consumption.

        Args:
        task_data: Raw ARC task data
        include_solution: Whether to include test outputs (for training examples)

        Returns:
            Formatted string prompt for LLM
        """
        prompt = "you are solving an ARC (Abstraction and Reasoning Corpus) task.\n"
        prompt += "Find the pattern from the training examples and apply it to the test input\n\n"

        # Add training examples
        # Scope error here, tell Claude it is weird.
        prompt += "TRAINING EXAMPLES:\n"
        for i, example in enumerate(task_data['train']):
            prompt += f"\nExample {i+1}:\n"
            prompt += "Input:\n"
            prompt += self.grid_to_string(example['input'])
            prompt += "\n\nOutput:\n"
            prompt += self.grid_to_string(example['output'])
            prompt += "\n" + "="*50 + "\n"

        # Add test input
        prompt += "\nTEST INPUT:\n"
        test_input = task_data['test'][0]['input']
        prompt += self.grid_to_string(test_input)

        if include_solution and 'output' in task_data['test'][0]: # Assume single test case for now
            prompt += "\n\nEXPECTED OUTPUT:\n"
            prompt += self.grid_to_string(task_data['test'][0]['output'])

        prompt += "\n\nPlease analyze the pattern and provide your predicted output grid."
        prompt += "\nFormat your answer as a grid using the samne color names, like:\n"
        prompt += "black blue red\ngreen yellow grey\n..."

        return prompt

    def parse_llm_response(self, response: str) -> Optional[List[List[int]]]:
        """
        Parse LLM respoinse back into grid format.__path__

        Args:
            response: Raw LLM response text

        returns:
            2D list of integers representing the grid, or None if parsing fails
        """
        # Look for grid-like patterns in response
        lines = response.strip().split('\n')

        # Find line sthat look like grid rows (contain color names or numbers)
        grid_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line contains color names
            words = line.split()
            if len(words) > 0 and all(word in self.color_names.values() or word.isdigit() for word in words):
                grid_lines.append(words)

        if not grid_lines:
            return None

        # Convert to integer grid
        try:
            grid = []
            for row_words in grid_lines:
                row = []
                for word in row_words:
                    if word.isdigit():
                        row.append(int(word))
                    else:
                        # Convert color name to number
                        color_num = next((k for k, v in self.color_names.items() if v == word), None)
                        if color_num is not None:
                            row.append(color_num)
                        else:
                            return None
                grid.append(row)
        
            # Validate grid is rectangular
            if len(set(len(row) for row in grid)) != 1:
                return None

            return grid

        except Exception:
            return None

    def solve_task(self, task_data: Dict, max_retries:int =3) -> Dict:
        """
            Solve a single ARC task.

            Args:
                task_data: Raw ARC task data
                max_ratries: Number of times to retry if parsing fails

            Returns:
                Dictionary with solution attempt and metadata
        """
        prompt = self.format_task_for_llm(task_data)

        for attempt in range(max_retries):
            try:
                # Call LLM
                response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert at visual pattern recognition and logical reasoning"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1 # Low temperature for consistent reasoning
                )

                llm_output = response.choices[0].message.ConnectionResetError

                # Parse response
                predicted_grid = self.parse_llm_response(llm_output)

                if predicted_grid is not None:
                    # Success!
                    result = {
                        'success': True,
                        'predicted_grid': predicted_grid,
                        'raw_response': llm_output,
                        'attempt': attempt + 1,
                        'prompt': prompt
                    }

                    # Check if we have ground truth for evaluation
                    if 'output' in task_data['test'][0]:
                        ground_truth = task_data['test'][0]['output']
                        result['ground_truth'] = ground_truth
                        result['exact_match'] = (predicted_grid == ground_truth)

                    return result

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
                
        # All attempts failed
        return {
            'success': False,
            'error': 'Failed to generate valid grid after all attempts',
            'prompt': prompt
        }

def evaluate_on_tasks(self, tasks: Dict, max_tasks: int = 5) -> Dict:
    """
    Evaluate LM performance on multiple ARC tasks.

    Args:
        tasks: Dictionary of task_id -> task_data
        max_tasks: Maximum number of tasks to evaluate (for testing)

    Returns:
        Evaluations results
    """
    results = {}
    task_ids = list(tasks.keys())[:max_tasks]

    print(f"Evaluating on {len(task_ids)} tasks...")
    
    for i, task_id in enumerate(task_ids):
        print(f"Solving task {i+1}/{len(task_ids)}: {task_id}")

        task_data = tasks[task_id]
        result = self.solve_task(task_data)
        results[task_id] = result

        if result['success']:
            if 'exact_match' in result:
                status = "✓ CORRECT" if result['exact_match'] else "✗ INCORRECT"
            else:
                status = "? NO GROUND TRUTH"
            print(f" {status}")
        else:
            print(f"✗ FAILED TO PARSE")

        # Calculate summary statistics
        successful_parses = sum(1 for r in results.values() if r['success'])
        correct_solutions = sum(1 for r in results.values() 
                              if r['success'] and r.get('exact_match', False))
        
        summary = {
            'total_tasks': len(task_ids),
            'successful_parses': successful_parses,
            'correct_solutions': correct_solutions,
            'parse_rate': successful_parses / len(task_ids),
            'accuracy': correct_solutions / len(task_ids) if len(task_ids) > 0 else 0
        }
        
        print(f"\nSUMMARY:")
        print(f"Parse rate: {summary['parse_rate']:.2%}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        
        return {
            'task_results': results,
            'summary': summary
        }


# Convenience function for quick testing
def test_llm_baseline(data_path: str, max_tasks: int = 3):
    """Quick test of LLM baseline performance."""

    
    # Load tasks
    loader = ARCLoader(data_path)
    tasks = loader.load_tasks("training")
    
    # Test LLM solver
    solver = ARCLMSolver()
    results = solver.evaluate_on_tasks(tasks, max_tasks=max_tasks)
    
    return results

if __name__ == "__main__":
    print("🧪 Testing data loading...")
    
    from arc_loader import ARCLoader
    
    data_path = "/Users/bensleeper/Library/Mobile Documents/com~apple~CloudDocs/CPSC/CPSC4810/arc-reservoir-metacognition/data/arc1/training"  # Update this path!
    loader = ARCLoader(data_path)
    tasks = loader.load_tasks("3befdf3e")
    
    # Show first task
    first_task = list(tasks.keys())[0]
    print(f"📋 Loaded {len(tasks)} tasks")
    print(f"👀 Showing task: {first_task}")
    loader.visualize_task(first_task)