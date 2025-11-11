from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Temporary setup
problem = "Solve 10 + 10"
blackboard = []

# "GEMINI_API_KEY in .env"
client = genai.Client()

system_prompt = """
        You are a reasoning agent participating in a collaborative problem-solving session.

        Your role:
        - Read all prior contributions on the shared blackboard
        - Add new insights, alternative approaches, or critiques
        - Build on others' ideas rather than repeating them
        - Be concise but thorough (2-4 sentences per contribution)
        - Focus on advancing the collective reasoning toward a solution
"""

def get_text(blackboard: list) -> str:
    if not blackboard:
        return "Empty"

    return "\n\n".join([
        f"Agent {e['agent_id']} (Round {e['round']}): {e['content']}"
        for e in blackboard
    ])

def create_prompt(problem: str, blackboard: list) -> str:
    user_prompt = f"""
        ORIGINAL PROBLEM: {problem}

        BLACKBOARD: {get_text(blackboard)}

        Tags:
        - [FINISH]: You believe the problem is solved and reasoning is correct
        - [EXPLORE]: You're exploring a new approach or alternative
        - [CRITIQUE]: You're challenging or questioning a previous contribution
        - [QUESTION]: You need clarification on something

        Task: Synth
    """
    return user_prompt

response = client.models.generate_content(
    model="gemini-2.5-flash", 
    config=types.GenerateContentConfig(
        system_instruction=system_prompt),
    contents=create_prompt(problem, blackboard)
)

print(response.text)