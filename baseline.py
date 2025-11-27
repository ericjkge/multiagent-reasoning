from dotenv import load_dotenv
from google import genai
from google.genai import types
import prompts

# Load environment variables
load_dotenv()

# Uses "GEMINI_API_KEY" in .env
client = genai.Client()

def solve_cot(problem: str, model_name: str = "gemini-2.5-flash") -> str:
    prompt = prompts.cot_prompt.format(input=problem)
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Error in CoT solve: {e}")
        return "Error"