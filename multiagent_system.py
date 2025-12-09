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
        self.visits = 0             # Visit count (for future MCTS/UCB/etc.)

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

        # Root node (empty content)
        current_nodes = [TreeNode(content="")] 
        total_tokens = 0
        
        # Loop for "d" steps
        for step in range(d):
            print(f"Step {step+1}/{d}, current nodes: {len(current_nodes)}")
            
            # 1. Generate k new thoughts per path (NOTE: * unpacks list created by list comprehension)
            proposal_results = await asyncio.gather(
                *[self._propose(initial_problem, node.content, k) for node in current_nodes]
            )
            
            # Use zip to pair parents with new thoughts cleanly
            candidates = []
            for parent_node, (proposals, tokens) in zip(current_nodes, proposal_results):
                total_tokens += tokens
                for content in proposals:
                    candidates.append((parent_node, content))

            # 2. Evaluate each next step
            eval_tasks = [self._evaluate(content) for (_, content) in candidates]
            eval_results = await asyncio.gather(*eval_tasks)
            
            # Pair candidates with scores
            scored_candidates = []
            for (parent_node, content), (score, tokens) in zip(candidates, eval_results):
                total_tokens += tokens
                scored_candidates.append((score, parent_node, content))

            # 3. Select top "b" next steps and create TreeNodes
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            selected = scored_candidates[:b]
            
            current_nodes = []
            for score, parent_node, content in selected:
                new_node = TreeNode(content=content, parent=parent_node, score=score)
                parent_node.add_child(new_node)
                current_nodes.append(new_node)
            
            # Log top thoughts
            if log_file:
                log_file.write(f"\n--- Step {step+1} Top Thoughts ---\n")
                for i, node in enumerate(current_nodes):
                    log_file.write(f"Rank {i+1} (Score: {node.score}):\n{node.content}\n{'-'*20}\n")
                    print(f"Rank {i+1} (Score: {node.score}):\n{node.content}\n{'-'*20}\n")

            if current_nodes:
                print(f"Top score: {current_nodes[0].score}")

        print(f"Total tokens used: {total_tokens}")
        # Best solution has highest-scoring newest thought
        return current_nodes[0].get_history() if current_nodes else "No solution found"

    # Internal helper for parsing "left" numbers (e.g. '2 + 8 = 10 (left: 8 10 14)' -> '8 10 14')
    def _extract_remaining(self, step: str) -> str:
        match = re.search(r'\(left:\s*([^\)]+)\)', step)
        return match.group(1).strip() if match else ""

    # Internal function for proposing new thoughts
    async def _propose(self, problem: str, parent_content: str, k: int) -> tuple[list[str], int]:
        # Extract state from parent's "left" numbers, or use original problem
        if parent_content:
            remaining = self._extract_remaining(parent_content)
        else:
            remaining = problem
        
        prompt = prompts.propose_prompt.format(input=remaining)
        response_text, token_count = await self.llm.agenerate(prompt, system_prompt=prompts.system_prompt)
        
        # Split by newline and filter empty lines
        proposals = [line.strip() for line in response_text.split('\n') if line.strip()]
        return proposals, token_count

    # Internal function for evaluating new thoughts
    async def _evaluate(self, candidate: str) -> tuple[float, int]:
        # Extract "left" numbers from candidate step
        remaining = self._extract_remaining(candidate)
        if not remaining:
            return 0.001, 0  # Invalid format
        
        prompt = prompts.value_prompt.format(input=remaining)
        response_text, token_count = await self.llm.agenerate(prompt, system_prompt=prompts.system_prompt)

        # Map sure/likely/impossible to scores (NOTE: values copied from Princeton ToT)
        response_lower = response_text.lower()
        if 'sure' in response_lower:
            return 20, token_count
        elif 'likely' in response_lower:
            return 1, token_count
        else:  # impossible or unrecognized
            return 0.001, token_count

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