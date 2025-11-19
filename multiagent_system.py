from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment vars and dataset
load_dotenv()

# Temporary setup
problem = "Solve 10 + 10"
blackboard = []
rounds = 3
agents = 2

# Uses "GEMINI_API_KEY in .env"
client = genai.Client()

system_prompt = """
        You are a reasoning agent participating in a collaborative problem-solving session.

        Task: Given the original problem and blackboard context, add a response based on one of the following tags.

        Tags:
        - [FINISH]: You believe the problem is solved and reasoning is correct
        - [EXPLORE]: You're exploring a new approach or alternative
        - [CRITIQUE]: You're challenging or questioning a previous contribution
        - [QUESTION]: You need clarification on something
"""

# Create independent agent contexts
# List of lists of dicts (sublists are conversation histories for each agent)
agent_contexts = [
    [
        types.Content(
            role="user",
            parts=[
                types.Part(
                    text=f"Try to solve {problem}. Choose from your given tags to determine which type of contribution would be most meaningful."
                )
            ],
        )
    ]
    for _ in range(agents)
]


# Extract text from blackboard (list of dicts)
def get_text(blackboard: list) -> str:
    if not blackboard:
        return "Empty"

    return "\n\n".join([e["content"] for e in blackboard])


for round in range(rounds):
    for i, agent_context in enumerate(agent_contexts):

        # Only add blackboard context after the first round
        if round > 0:
            blackboard_summary = get_text(blackboard)
            update_message = types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=f"""
                Here's what has been discussed so far: {blackboard_summary}
                """
                    )
                ],
            )
            agent_context.append(update_message)

        # Generate response with agent memory
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=agent_context,
        )

        # Add response to agent memory
        agent_context.append(
            types.Content(role="model", parts=[types.Part(text=response.text)])
        )

        # Add response to blackboard
        blackboard.append({"agent_id": i, "round": round, "content": response.text})








# Write output to file instead of printing
output_file = "output.txt"

with open(output_file, "w") as f:
    f.write("=== FINAL BLACKBOARD ===\n\n")

    for entry in blackboard:
        f.write(f"Agent {entry['agent_id']} (Round {entry['round']}):\n")
        f.write(f"{entry['content']}\n")
        f.write("-" * 80 + "\n")

    f.write("=== AGENT MEMORIES ===\n\n")

    for i, context in enumerate(agent_contexts):
        f.write(f"AGENT {i} - Total messages: {len(context)}\n")

        for j, msg in enumerate(context):
            f.write(f"[Message {j}] Role: {msg.role}\n")
            f.write(f"{msg.parts[0].text}\n")
            f.write("-" * 80 + "\n")

print(f"Output written to {output_file}")


"""
Structured outputs: https://ai.google.dev/gemini-api/docs/structured-output?example=recipe
Thought signatures: https://ai.google.dev/gemini-api/docs/thought-signatures
Also function calling, 
"""
