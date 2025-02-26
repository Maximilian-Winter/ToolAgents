import os
from typing import Dict, List, Any

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI, AnthropicChatAPI
from pydantic import BaseModel, Field
from enum import Enum

from dotenv import load_dotenv

load_dotenv()

class ReflectionType(Enum):
    REASONING = "reasoning"
    PLAN = "plan" 
    CRITIQUE = "critique"
    IMPROVEMENT = "improvement"

class SelfReflection(BaseModel):
    """Perform a structured self-reflection on the agent's own reasoning process"""
    reflection_type: ReflectionType = Field(..., description="Type of reflection to perform")
    content_to_reflect_on: str = Field(..., description="The content to reflect upon, usually a previous thought or plan")
    depth: int = Field(1, description="Depth of reflection from 1 (simple) to 3 (deep)")
    
    def run(self):
        return f"Reflecting on content using {self.reflection_type.value} at depth {self.depth}"

class PlanStep(BaseModel):
    """Define a specific step in a larger plan"""
    description: str = Field(..., description="Description of the plan step")
    expected_outcome: str = Field(..., description="What this step should accomplish")
    dependencies: List[str] = Field(default_factory=list, description="Steps that must be completed before this one")
    
    def run(self):
        return f"Plan step created: {self.description}"

class TestHypothesis(BaseModel):
    """Test a hypothesis by logical reasoning"""
    hypothesis: str = Field(..., description="The hypothesis to test")
    test_method: str = Field(..., description="How to test the hypothesis (e.g., 'logical analysis', 'example generation')")
    
    def run(self):
        return f"Testing hypothesis: '{self.hypothesis}' using {self.test_method}"

# Create function tools
self_reflection_tool = FunctionTool(SelfReflection)
plan_step_tool = FunctionTool(PlanStep)
test_hypothesis_tool = FunctionTool(TestHypothesis)

# Set up API client
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
settings = api.get_default_settings()
settings.temperature = 0.2
agent = ChatToolAgent(chat_api=api)

# Create tool registry
tools = ToolRegistry()
tools.add_tools([self_reflection_tool, plan_step_tool, test_hypothesis_tool])

# Initialize chat history
chat_history = ChatHistory()
chat_history.add_system_message("""
You are a self-reflective problem-solving agent capable of metacognition.
Your approach to problems includes:

1. UNDERSTANDING: Carefully analyze the problem before attempting a solution
2. PLANNING: Break complex problems into clear steps
3. REFLECTION: Regularly pause to reflect on your reasoning process
4. TESTING: Test your hypotheses and plans critically
5. ITERATION: Improve your approach based on reflection

Use the SelfReflection tool to explicitly review your own reasoning
Use the PlanStep tool to outline specific steps in your problem-solving approach
Use the TestHypothesis tool to check the validity of your assumptions

When solving problems:
- Start by understanding the problem completely
- Develop a plan with concrete steps
- Think step-by-step and show your reasoning
- Use self-reflection to catch errors in your reasoning
- Iterate and improve based on your reflections

Your goal is to demonstrate exceptional problem-solving through careful reasoning and self-correction.
""")

def process_input(user_input: str) -> str:
    """Process user input through the self-reflection agent"""
    chat_history.add_user_message(user_input)
    
    response = agent.get_response(
        messages=chat_history.get_messages(),
        settings=settings,
        tool_registry=tools
    )
    
    chat_history.add_messages(response.messages)
    return response.response

def solve_with_reflection(problem: str) -> Dict[str, Any]:
    """Solve a problem using the self-reflection process"""
    # Initial problem understanding
    understanding_prompt = f"""
I need to solve the following problem:

{problem}

First, I'll analyze this problem carefully to make sure I understand it completely.
"""
    understanding = process_input(understanding_prompt)
    
    # Create a plan
    planning_prompt = """
Now that I understand the problem, I'll create a detailed plan to solve it.
Break this down into specific steps using the PlanStep tool.
"""
    plan = process_input(planning_prompt)
    
    # Execute with reflection
    execution_prompt = """
I'll now execute my plan, step by step. For each significant step:
1. I'll think through my approach
2. Use the SelfReflection tool to critique my reasoning
3. Use the TestHypothesis tool to verify key assumptions
4. Adjust my approach based on reflections
"""
    execution = process_input(execution_prompt)
    
    # Final solution with reflection
    solution_prompt = """
Based on all my work, reflections, and testing, I'll now provide my final solution to the original problem.
After presenting the solution, use the SelfReflection tool one last time to assess the overall quality of my approach.
"""
    solution = process_input(solution_prompt)
    
    return {
        "understanding": understanding,
        "plan": plan,
        "execution": execution,
        "solution": solution,
        "full_history": chat_history
    }

# Example usage
if __name__ == "__main__":
    problem = input("Enter a problem to solve: ")
    results = solve_with_reflection(problem)
    
    print("\n=== FINAL SOLUTION ===\n")
    print(results["solution"])