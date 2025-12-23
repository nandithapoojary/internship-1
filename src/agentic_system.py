import re
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# --- 1. Core Interfaces ---

class Tool(ABC):
    """Abstract base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
        
    @property
    @abstractmethod
    def description(self) -> str:
        pass
        
    @abstractmethod
    def run(self, **kwargs) -> str:
        pass

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generates text based on the prompt."""
        pass

# --- 2. Concrete Implementations (Tools & LLM) ---

class CalculatorTool(Tool):
    """A simple calculator tool."""
    name = "Calculator"
    description = "Useful for answering math questions. Input should be a valid mathematical expression string."
    
    def run(self, expression: str) -> str:
        try:
            # WARNING: eval is dangerous in production! Using it here for simplicity.
            # In a real app, use a safer parser like `numexpr` or valid logic.
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error executing calculation: {e}"

class WeatherTool(Tool):
    """A mock weather tool."""
    name = "Weather"
    description = "Get current weather for a city. Input should be the city name."
    
    def run(self, city: str) -> str:
        # Mock response
        if "london" in city.lower():
            return "15°C, Cloudy"
        elif "delhi" in city.lower():
            return "32°C, Sunny"
        elif "nyc" in city.lower():
             return "20°C, Rain"
        return "Unknown city"

class MockLLM(LLMProvider):
    """
    A Mock LLM that simulates an agent's reasoning process for demo purposes.
    It follows a fixed script based on the prompt to demonstrate the ReAct loop.
    """
    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Simple heuristic to simulate an agent "Thinking"
        if "Weather" in prompt and "Delhi" in prompt and "Action:" not in prompt:
            return 'Thought: The user is asking about the weather in Delhi. I should use the Weather tool.\nAction: Weather\nAction Input: "Delhi"'
        
        if "Observation: 32°C, Sunny" in prompt and "Final Answer" not in prompt:
             return "Thought: I have the weather. Now I can answer.\nFinal Answer: It is 32°C and Sunny in Delhi."

        if "2 + 2" in prompt:
             if "Action:" not in prompt:
                 return 'Thought: I need to calculate 2 + 2.\nAction: Calculator\nAction Input: "2 + 2"'
             if "Observation: 4" in prompt:
                 return "Thought: I have the result.\nFinal Answer: The answer is 4."
        
        return "Thought: I am confused. \nFinal Answer: I don't know how to handle this."

# --- 3. The Agent ---

class ReActAgent:
    """
    A minimal Agent that implements the ReAct (Reasoning + Acting) pattern.
    Loop: Thought -> Action -> Action Input -> Observation -> Thought ...
    """
    def __init__(self, llm: LLMProvider, tools: List[Tool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in tools])

    def run(self, query: str, max_steps: int = 5) -> str:
        system_prompt = f"""
Answer the following questions as best you can. You have access to the following tools:

{self.tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{', '.join(self.tools.keys())}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {query}
"""
        history = system_prompt
        print(f"--- Agent started for: '{query}' ---")
        
        for step in range(max_steps):
            # 1. Ask LLM to think/act
            output = self.llm.generate(history, stop=["Observation:"])
            print(output) # Print the LLM's thought/action
            
            history += output + "\n"
            
            # 2. Check for Final Answer
            if "Final Answer:" in output:
                return output.split("Final Answer:")[-1].strip()
            
            # 3. Parse Action
            match = re.search(r"Action: (.*?)\nAction Input: (.*)", output, re.DOTALL)
            if not match:
                print("Error: Could not parse action. Ending.")
                break
                
            action_name = match.group(1).strip()
            action_input = match.group(2).strip().strip('"') # Remove quotes if present
            
            # 4. Execute Action
            if action_name in self.tools:
                print(f" * Executing Tool: {action_name} with '{action_input}'")
                try:
                    observation = self.tools[action_name].run(action_input)
                except Exception as e:
                    observation = f"Error: {str(e)}"
            else:
                observation = f"Error: Tool '{action_name}' not found."
            
            print(f"Observation: {observation}")
            history += f"Observation: {observation}\n"

        return "Agent stopped (max steps reached or error)."

# --- 4. Main Execution ---

if __name__ == "__main__":
    # Setup
    mock_llm = MockLLM()
    tools = [CalculatorTool(), WeatherTool()]
    agent = ReActAgent(mock_llm, tools)
    
    # Run 1
    print("\n>>> TEST 1: Delhi Weather")
    result = agent.run("What is the weather in Delhi?")
    print(f"\nFinal Result: {result}")

    # Run 2
    print("\n>>> TEST 2: Math")
    result = agent.run("What is 2 + 2?")
    print(f"\nFinal Result: {result}")
