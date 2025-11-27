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

    def get_left(self, thought: str) -> str:
        """
        Extracts the 'left' numbers from the thought string.
        Expected format example: "4 + 8 = 12 (left: 6 12)"
        """
        match = re.search(r'\(left: (.*?)\)', thought)
        if match:
            return match.group(1)
        return ""
