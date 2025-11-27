# Multiagent Reasoning

A **Tree of Thoughts** framework for multi-agent reasoning using Breadth-First Search (BFS).

## Quick Start
1. Set `GEMINI_API_KEY` in `.env`
2. `pip install -r requirements.txt`
3. Run ToT: `python3 multiagent_system.py`
4. Run Baseline (CoT): `python3 baseline.py`

## Files
- `multiagent_system.py`: Main ToT class (BFS search)
- `tasks.py`: Task logic (e.g. Game of 24)
- `prompts.py`: Prompt templates
- `baseline.py`: Zero-shot CoT baseline

## Config
Modify `k` (width), `b` (breadth), and `d` (depth) in `multiagent_system.py` to tune the search.