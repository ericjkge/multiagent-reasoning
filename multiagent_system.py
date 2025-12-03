from dotenv import load_dotenv
from pydantic import BaseModel
import prompts
from tasks import Game24Task, BaseTask
from models import GeminiLLM, BaseLLM
import re

# Load environment variables
load_dotenv()

# Thought class (thought string and evaluation score)
class Thought(BaseModel):
    thought: str
    score: float

class TreeOfThoughts:
    def __init__(self, task: BaseTask, llm: BaseLLM):
        self.task = task
        self.llm = llm

    def solve(self, initial_problem: str, k: int = 3, b: int = 5, d: int = 3, log_file=None):
        """
        Solves the problem using BFS.

        k: Number of proposed thoughts per state
        b: Branching factor (top states to keep)
        d: Max depth (steps)
        log_file: Open file handle for logging (optional)
        """
        # Validate input
        if not self.task.validate_input(initial_problem):
            return "Invalid input for task."

        # Generate full task prompt
        task_prompt = self.task.get_prompt(initial_problem)

        # Empty "tree" for tracking paths (strings representing full move histories)
        current_paths = [""] 
        total_tokens = 0
        
        # Loop for "d" steps
        for step in range(d):
            print(f"Step {step+1}/{d}, current paths: {len(current_paths)}")
            
            # 1. Generate k new thoughts per path
            candidates = []
            for path in current_paths:
                proposals, tokens = self._propose(task_prompt, path, k)
                total_tokens += tokens
                
                for p in proposals:
                    new_path = (path + "\n" + p).strip() # Old path + new thought (NOTE: redundant path copying)
                    candidates.append(new_path)

            if not candidates:
                break

            # 2. Evaluate each new thought
            evaluated_thoughts = []
            for path in candidates:
                last_step = path.split('\n')[-1] # Evaluate last step in each path (i.e. new thought)
                score, tokens = self._evaluate(task_prompt, path, last_step)
                total_tokens += tokens
                evaluated_thoughts.append(Thought(thought=path, score=score))

            # 3. Select top "b" new thoughts (prune others)
            evaluated_thoughts.sort(key=lambda x: x.score, reverse=True) # Sort by descending score
            selected = evaluated_thoughts[:b] # Keep top b
            current_paths = [t.thought for t in selected]
            
            # Log top thoughts
            if log_file:
                log_file.write(f"\n--- Step {step+1} Top Thoughts ---\n")
                for i, t in enumerate(selected):
                    log_file.write(f"Rank {i+1} (Score: {t.score}):\n{t.thought}\n{'-'*20}\n")
                    print(f"Rank {i+1} (Score: {t.score}):\n{t.thought}\n{'-'*20}\n")

            if selected:
                print(f"Top score: {selected[0].score}")

        print(f"Total tokens used: {total_tokens}")
        # Best solution has highest-scoring newest thought
        return current_paths[0] if current_paths else "No solution found"

    # Internal function for proposing new thoughts
    def _propose(self, problem: str, history: str, k: int) -> tuple[list[str], int]:
        prompt = prompts.propose_prompt.format(
            input=problem,
            history=history if history else "Empty",
            k=k
        )
        
        response_text, token_count = self.llm.generate(prompt, system_prompt=prompts.system_prompt)
        
        # Split by newline and filter empty lines
        return [line.strip() for line in response_text.split('\n') if line.strip()], token_count

    # Internal function for evaluating new thoughts
    def _evaluate(self, problem: str, history: str, candidate: str) -> tuple[float, int]:
        prompt = prompts.value_prompt.format(
            input=problem,
            history=history,
            candidate=candidate
        )
        
        response_text, token_count = self.llm.generate(prompt, system_prompt=prompts.system_prompt)

        # Match 0.x, 1.0, and 1 in text output
        match = re.search(r'Score:\s*(0\.\d+|1\.0|1)', response_text)
        if match:
            return float(match.group(1)), token_count
        return 0.1, token_count # Default low score if parse fails

if __name__ == "__main__":
    # Initialize task and solver
    task = Game24Task()
    llm = GeminiLLM()
    tot = TreeOfThoughts(task, llm)
    
    # Example problem
    problem = "2 2 6 8"
    print(f"Solving: {problem}")
    
    # Run Tree of Thoughts
    with open("output.txt", "w") as f:
        f.write(f"Problem: {problem}\n")
        best_path = tot.solve(problem, k=3, b=5, d=3, log_file=f)
        
        f.write("\n=== BEST PATH ===\n")
        f.write(best_path)
    
    # Output results
    print("\n=== BEST PATH ===")
    print(best_path)