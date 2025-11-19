from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Uses "GEMINI_API_KEY" in .env
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


# Define structured output schema for synthesizer
class Synthesized(BaseModel):
    answer: int = Field(description="The final numerical answer to the problem")
    reasoning: str = Field(
        description="Brief explanation of why this is the correct answer based on the discussion"
    )


# Extract text from blackboard (list of dicts)
def get_text(blackboard: list) -> str:
    if not blackboard:
        return "Empty"

    return "\n\n".join([e["content"] for e in blackboard])


def multiagent_solve(
    problem: str, rounds: int, agents: int
) -> tuple[Synthesized, list]:
    """
    Run multiagent debate and return synthesized answer.

    Args:
        problem: The problem to solve
        rounds: Number of debate rounds
        agents: Number of agents

    Returns:
        Synthesized object with answer and reasoning
    """
    blackboard = []

    # Create independent agent contexts
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

    # Synthesize final answer
    synthesizer_prompt = f"""
        You are a synthesizer agent responsible for determining the final output of a collaborative problem-solving session.
        Problem: {problem}

        Discussion: {get_text(blackboard)}

        Task: Based on all the reasoning above, provide the final numerical answer and your reasoning.
    """

    synthesized = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=synthesizer_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=Synthesized.model_json_schema(),
        ),
    )

    final_answer = Synthesized.model_validate_json(synthesized.text)
    return final_answer, blackboard


def single_agent_solve(problem: str) -> Synthesized:
    """
    Run single agent baseline (no debate).

    Args:
        problem: The problem to solve

    Returns:
        Synthesized object with answer and reasoning
    """
    single_agent_prompt = f"""
        You are a helpful math problem solver.
        
        Problem: {problem}
        
        Task: Solve this problem step by step and provide your final numerical answer with reasoning.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=single_agent_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=Synthesized.model_json_schema(),
        ),
    )

    result = Synthesized.model_validate_json(response.text)
    return result


# Main execution (for standalone testing)
if __name__ == "__main__":
    problem = "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

    final_answer, blackboard = multiagent_solve(problem, rounds=2, agents=1)

    # Write blackboard to output file
    output_file = "output.txt"

    with open(output_file, "w") as f:
        f.write(f"=== FINAL ANSWER: {final_answer.answer} === \n\n")
        f.write(f"Reasoning: {final_answer.reasoning}\n\n")

        f.write("=== FINAL BLACKBOARD ===\n\n")

        for entry in blackboard:
            f.write(f"Agent {entry['agent_id']} (Round {entry['round']}):\n")
            f.write(f"{entry['content']}\n")
            f.write("-" * 80 + "\n")
