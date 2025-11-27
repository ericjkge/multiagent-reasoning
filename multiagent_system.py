from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
import prompts
from tasks import Game24Task, BaseTask
import re

# Load environment variables
load_dotenv()

# Uses "GEMINI_API_KEY" in .env
client = genai.Client()

# Thought class (thought string and evaluation score)
class Thought(BaseModel):
    thought: str
    score: float

class TreeOfThoughts:
    def __init__(self, task: BaseTask, model_name: str = "gemini-2.5-flash"):
        self.task = task # Unused for now
        self.model_name = model_name

    def solve(self, initial_problem: str, k: int = 3, b: int = 5, d: int = 3):
        """
        Solves the problem using BFS.

        k: Number of proposed thoughts per state
        b: Branching factor (top states to keep)
        d: Max depth (steps)
        """
        # Validate Input
        if not self.task.validate_input(initial_problem):
            return "Invalid input for task."

        # Empty "tree" for tracking paths (strings representing full move histories)
        current_paths = [""] 
        
        # Loop for "d" steps
        for step in range(d):
            print(f"Step {step+1}/{d}, current paths: {len(current_paths)}")
            
            # 1. Generate k new thoughts per path
            candidates = []
            for path in current_paths:
                proposals = self._propose(initial_problem, path, k)
                
                for p in proposals:
                    new_path = (path + "\n" + p).strip() # Old path + new thought (NOTE: redundant path copying)
                    candidates.append(new_path)

            if not candidates:
                break

            # 2. Evaluate each new thought
            evaluated_thoughts = []
            for path in candidates:
                last_step = path.split('\n')[-1] # Evaluate last step in each path (i.e. new thought)
                score = self._evaluate(initial_problem, path, last_step)
                evaluated_thoughts.append(Thought(thought=path, score=score))

            # 3. Select top "b" new thoughts (prune others)
            evaluated_thoughts.sort(key=lambda x: x.score, reverse=True) # Sort by descending score
            selected = evaluated_thoughts[:b] # Keep top b
            current_paths = [t.thought for t in selected]
            
            if selected:
                print(f"Top score: {selected[0].score}")

        # Best solution has highest-scoring newest thought
        return current_paths[0] if current_paths else "No solution found"

    # Internal function for proposing new thoughts
    def _propose(self, problem: str, history: str, k: int) -> list[str]:
        prompt = prompts.propose_prompt.format(
            input=problem,
            history=history if history else "Empty",
            k=k
        )
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=prompts.system_prompt
                )
            )
            # Split by newline and filter empty lines
            return [line.strip() for line in response.text.split('\n') if line.strip()]
        except Exception as e:
            print(f"Error in propose: {e}")
            return []

    # Internal function for evaluating new thoughts
    def _evaluate(self, problem: str, history: str, candidate: str) -> float:
        prompt = prompts.value_prompt.format(
            input=problem,
            history=history,
            candidate=candidate
        )
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=prompts.system_prompt
                )
            )

            # Match 0.x, 1.0, and 1 in text output
            match = re.search(r'Score:\s*(0\.\d+|1\.0|1)', response.text)
            if match:
                return float(match.group(1))
            return 0.1 # Default low score if parse fails
        except Exception as e:
            print(f"Error in evaluate: {e}")
            return 0.0

if __name__ == "__main__":
    # Initialize task and solver
    task = Game24Task()
    tot = TreeOfThoughts(task)
    
    # Example problem
    problem = "2 2 6 8"
    print(f"Solving: {problem}")
    
    # Run Tree of Thoughts
    best_path = tot.solve(problem, k=3, b=5, d=3)
    
    # Output results
    print("\n=== BEST PATH ===")
    print(best_path)
    
    with open("output.txt", "w") as f:
        f.write(f"Problem: {problem}\n\n")
        f.write("=== BEST PATH ===\n")
        f.write(best_path)