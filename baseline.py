from dotenv import load_dotenv
import prompts
from models import GeminiLLM

# Load environment variables
load_dotenv()

def solve_cot(problem: str, model_name: str = "gemini-2.5-flash") -> str:
    prompt = prompts.cot_prompt.format(input=problem)
    
    llm = GeminiLLM(model_name=model_name)
    
    response_text, _ = llm.generate(prompt)
    return response_text