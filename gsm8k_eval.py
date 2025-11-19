from datasets import load_dataset
from multiagent_system import multiagent_solve, single_agent_solve
import re

ds = load_dataset("openai/gsm8k", "main")["test"]

def extract_answer(answer_text: str) -> int:
    """Extract answer from GSM8K format (#### 123)"""
    match = re.search(r'####\s*(-?\d+)', answer_text)
    return int(match.group(1).replace(",", "")) if match else None

# Evaluate on first 10 problems
single_correct = 0
multi_correct = 0

for i in range(10):
    problem = ds[i]
    question = problem["question"]
    correct = extract_answer(problem["answer"])
    
    # Single agent
    single = single_agent_solve(question)
    single_correct += (single.answer == correct)
    
    # Multiagent
    multi, _ = multiagent_solve(question, rounds=2, agents=2)
    multi_correct += (multi.answer == correct)
    
    print(f"{i+1}. Correct: {correct} | Single: {single.answer} | Multi: {multi.answer}")

print(f"\nSingle: {single_correct}/10 | Multi: {multi_correct}/10")