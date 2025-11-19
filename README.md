# Multiagent Reasoning

Testing whether multiagent debate improves reasoning on GSM8K math problems.

## Setup
1. Create `.env` file with `GEMINI_API_KEY` from Google AI Studio
2. Install dependencies: `pip install -r requirements.txt`

## Usage
- `python multiagent_system.py` - Run single problem with debate
- `python gsm8k_eval.py` - Evaluate single vs multiagent on GSM8K

## Open Questions:
- Termination: Fixed rounds vs. confidence voting vs. meta-moderator?
- Communication: Tags vs. hierarchical blackboard?
- Agent diversity: Different prompts/capabilities per agent?
- Triggering order: Methods for determining agent calls?