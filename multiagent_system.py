import asyncio
from dotenv import load_dotenv
import prompts
from tasks import Game24Task, BaseTask
from models import GeminiLLM, BaseLLM
import re

# Load environment variables
load_dotenv()

class TreeNode:
    def __init__(self, content: str, parent: 'TreeNode' = None, score: float = 0.0):
        self.content = content      # Step string (e.g. "2+2=4")
        self.parent = parent
        self.children = []
        self.score = score          # Evaluation score

    # Convert history to string (walk from node to root)        
    def get_history(self) -> str:
        history = []
        curr = self
        while curr and curr.content: # Stop if content is empty (root dummy)
            history.append(curr.content)
            curr = curr.parent
        return "\n".join(reversed(history))
    
    def add_child(self, node: 'TreeNode'):
        self.children.append(node)

class TreeOfThoughts:
    def __init__(self, task: BaseTask, llm: BaseLLM):
        self.task = task
        self.llm = llm

    async def solve(self, initial_problem: str, k: int = 3, b: int = 5, d: int = 3, log_file=None):
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

        # Root node (empty content)
        current_nodes = [TreeNode(content="")] 
        total_tokens = 0
        
        # Loop for "d" steps
        for step in range(d):
            print(f"Step {step+1}/{d}, current nodes: {len(current_nodes)}")
            
            # 1. Generate k new thoughts per path (NOTE: * unpacks list created by list comprehension)
            proposal_results = await asyncio.gather(
                *[self._propose(task_prompt, node.get_history(), k) for node in current_nodes]
            )
            
            # Use zip to pair parents with new thoughts cleanly
            candidates = []
            for parent_node, (proposals, tokens) in zip(current_nodes, proposal_results):
                total_tokens += tokens
                for p in proposals:
                    # New node linked to parent
                    new_node = TreeNode(content=p, parent=parent_node)
                    candidates.append(new_node)

            # 2. Evaluate each new thought
            eval_tasks = []
            for node in candidates:
                # History for evaluation excludes new step
                history = node.parent.get_history()
                eval_tasks.append(self._evaluate(task_prompt, history, node.content))
            
            eval_results = await asyncio.gather(*eval_tasks)
            
            # Update scores
            for node, (score, tokens) in zip(candidates, eval_results):
                total_tokens += tokens
                node.score = score
                node.parent.add_child(node)

            # 3. Select top "b" new nodes (prune others)
            candidates.sort(key=lambda x: x.score, reverse=True) # Sort by descending score
            selected_nodes = candidates[:b] # Keep top b
            current_nodes = selected_nodes
            
            # Log top thoughts
            if log_file:
                log_file.write(f"\n--- Step {step+1} Top Thoughts ---\n")
                for i, node in enumerate(selected_nodes):
                    log_file.write(f"Rank {i+1} (Score: {node.score}):\n{node.content}\n{'-'*20}\n")
                    print(f"Rank {i+1} (Score: {node.score}):\n{node.content}\n{'-'*20}\n")

            if selected_nodes:
                print(f"Top score: {selected_nodes[0].score}")

        print(f"Total tokens used: {total_tokens}")
        # Best solution has highest-scoring newest thought
        return current_nodes[0].get_history() if current_nodes else "No solution found"

    # Internal function for proposing new thoughts
    async def _propose(self, problem: str, history: str, k: int) -> tuple[list[str], int]:
        prompt = prompts.propose_prompt.format(
            input=problem,
            history=history if history else "Empty",
            k=k
        )
        
        response_text, token_count = await self.llm.agenerate(prompt, system_prompt=prompts.system_prompt)
        
        # Split by newline and filter empty lines
        return [line.strip() for line in response_text.split('\n') if line.strip()], token_count

    # Internal function for evaluating new thoughts
    async def _evaluate(self, problem: str, history: str, candidate: str) -> tuple[float, int]:
        prompt = prompts.value_prompt.format(
            input=problem,
            history=history,
            candidate=candidate
        )
        
        response_text, token_count = await self.llm.agenerate(prompt, system_prompt=prompts.system_prompt)

        # Match 0.x, 1.0, and 1 in text output
        match = re.search(r'Score:\s*(0\.\d+|1\.0|1)', response_text)
        if match:
            return float(match.group(1)), token_count
        return 0.1, token_count # Default low score if parse fails

async def main():
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
        best_path = await tot.solve(problem, k=3, b=5, d=3, log_file=f)
        
        f.write("\n=== BEST PATH ===\n")
        f.write(best_path)
    
    # Output results
    print("\n=== BEST PATH ===")
    print(best_path)

if __name__ == "__main__":
    asyncio.run(main())