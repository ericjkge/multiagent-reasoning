from abc import ABC, abstractmethod
import re

class BaseTask(ABC):
    """
    Abstract base class for all tasks.
    """
    
    @abstractmethod
    def validate_input(self, input_str: str) -> bool:
        """
        Check if the initial problem input is valid.
        """
        pass

    @abstractmethod
    def check_solution(self, expression: str, numbers: list[str]) -> bool:
        """
        Check if a final answer is correct.
        """
        pass

    @abstractmethod
    def get_prompt(self, input_str: str) -> str:
        """
        Returns the task-specific prompt/instructions based on the input.
        """
        pass


class Game24Task(BaseTask):
    def __init__(self):
        self.target = 24

    def validate_input(self, input_str: str) -> bool:
        return len(input_str.strip().split()) == 4

    def check_solution(self, expression: str, numbers: list[str] = None) -> bool:
        try:
            return abs(eval(expression) - self.target) < 1e-6
        except:
            return False

    def get_prompt(self, input_str: str) -> str:
        return (
            f"Use the numbers {input_str} to create a valid arithmetic expression that equals {self.target}.\n"
            "At each step, select two numbers to operate on (+, -, *, /).\n"
            "Format your response exactly like this example:\n"
            "4 + 8 = 12 (left: 2 6 12)\n"
            "Goal: Reach exactly 24."
        )

    def get_left(self, thought: str) -> str:
        """
        Extracts the 'left' numbers from the thought string.
        Expected format example: "4 + 8 = 12 (left: 6 12)"
        """
        match = re.search(r'\(left: (.*?)\)', thought)
        if match:
            return match.group(1)
        return ""
