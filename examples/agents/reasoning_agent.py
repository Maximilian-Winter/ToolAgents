import os
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

class ReasongingType(Enum):
    DECOMPOSITION = "decomposition"
    ANALOGY = "analogy"
    COUNTERFACTUAL = "counterfactual"
    MATHEMATICAL = "mathematical"
    LOGICAL = "logical"
    CAUSAL = "causal"

class SubProblem(BaseModel):
    """Create a subproblem that breaks down a larger problem"""
    problem_statement: str = Field(..., description="Clear statement of the subproblem")
    relevance: str = Field(..., description="How solving this helps with the main problem")
    
    def run(self):
        return f"Subproblem created: {self.problem_statement}"

class Analogy(BaseModel):
    """Create an analogy to understand or explain a concept"""
    target_concept: str = Field(..., description="The concept to be explained")
    source_domain: str = Field(..., description="The familiar domain to draw an analogy from")
    mapping: str = Field(..., description="How elements in the source map to the target")
    
    def run(self):
        return f"Analogy created: {self.target_concept} is like {self.source_domain} because {self.mapping}"

class LogicalAnalysis(BaseModel):
    """Perform a structured logical analysis of an argument or hypothesis"""
    statement: str = Field(..., description="The statement to analyze")
    premises: List[str] = Field(..., description="The premises or assumptions")
    conclusion: str = Field(..., description="The conclusion derived from premises")
    validity_assessment: str = Field(..., description="Assessment of the logical validity")
    
    def run(self):
        return f"Logical analysis of: '{self.statement}' with conclusion: {self.conclusion}"

class MathematicalSolution(BaseModel):
    """Solve a mathematical problem step-by-step"""
    problem: str = Field(..., description="The mathematical problem to solve")
    approach: str = Field(..., description="The mathematical approach or formula to apply")
    steps: List[str] = Field(..., description="Step-by-step solution process")
    answer: Union[str, int, float] = Field(..., description="The final answer")
    
    def run(self):
        return f"Mathematical solution to: {self.problem}\nAnswer: {self.answer}"

class CounterfactualAnalysis(BaseModel):
    """Explore a counterfactual scenario to analyze implications"""
    actual_scenario: str = Field(..., description="The actual situation or fact")
    counterfactual_change: str = Field(..., description="The hypothetical change to consider")
    implications: List[str] = Field(..., description="The logical implications of this change")
    
    def run(self):
        return f"Counterfactual analysis: If {self.counterfactual_change} instead of {self.actual_scenario}, then {'; '.join(self.implications)}"

class CausalModel(BaseModel):
    """Create a causal model showing relationships between factors"""
    effect: str = Field(..., description="The outcome or effect to be explained")
    causes: List[str] = Field(..., description="The factors that contribute to the effect")
    relationships: List[str] = Field(..., description="Description of how causes relate to the effect")
    
    def run(self):
        return f"Causal model for {self.effect} with {len(self.causes)} causal factors identified"

# Create function tools
sub_problem_tool = FunctionTool(SubProblem)
analogy_tool = FunctionTool(Analogy)
logical_analysis_tool = FunctionTool(LogicalAnalysis)
mathematical_solution_tool = FunctionTool(MathematicalSolution)
counterfactual_analysis_tool = FunctionTool(CounterfactualAnalysis)
causal_model_tool = FunctionTool(CausalModel)

# Set up API client
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
settings = api.get_default_settings()
settings.temperature = 0.3
agent = ChatToolAgent(chat_api=api)

# Create tool registry
tools = ToolRegistry()
tools.add_tools([
    sub_problem_tool, 
    analogy_tool, 
    logical_analysis_tool, 
    mathematical_solution_tool,
    counterfactual_analysis_tool,
    causal_model_tool
])

# Initialize chat history
chat_history = ChatHistory()
chat_history.add_system_message("""
You are an advanced reasoning agent that uses multiple reasoning strategies to solve problems.
Your reasoning toolkit includes:

1. DECOMPOSITION: Breaking complex problems into simpler subproblems
2. ANALOGY: Using familiar concepts to understand unfamiliar ones
3. LOGICAL ANALYSIS: Analyzing arguments for logical validity
4. MATHEMATICAL REASONING: Solving quantitative problems step-by-step
5. COUNTERFACTUAL THINKING: Exploring hypothetical scenarios
6. CAUSAL REASONING: Identifying cause-effect relationships

Use the appropriate reasoning tools based on the nature of the problem:
- For complex problems, use SubProblem to break them down
- For abstract concepts, use Analogy to make them more concrete
- For arguments, use LogicalAnalysis to assess validity
- For quantitative problems, use MathematicalSolution for step-by-step solutions
- For "what if" scenarios, use CounterfactualAnalysis to explore implications
- For understanding why something happens, use CausalModel to identify factors

Choose the most appropriate reasoning strategies for each problem, and you may use multiple strategies when beneficial.
Explain your reasoning process clearly and show your thinking step-by-step.
""")

def solve_problem(problem: str, reasoning_type: Optional[ReasongingType] = None) -> str:
    """Solve a problem using the specified reasoning type, or let the agent choose if not specified"""
    
    # Construct the prompt based on whether a reasoning type was specified
    if reasoning_type:
        prompt = f"""
I need to solve the following problem using {reasoning_type.value} reasoning:

{problem}

Please apply {reasoning_type.value} reasoning to solve this problem step-by-step.
Use the appropriate reasoning tools to structure your approach.
"""
    else:
        prompt = f"""
I need to solve the following problem:

{problem}

Please analyze this problem and choose the most appropriate reasoning strategies to solve it.
Use the reasoning tools to structure your approach and show your thinking step-by-step.
"""
    
    chat_history.add_user_message(prompt)
    
    response = agent.get_response(
        messages=chat_history.get_messages(),
        settings=settings,
        tool_registry=tools
    )
    
    chat_history.add_messages(response.messages)
    return response.response

# Example usage
if __name__ == "__main__":
    print("Advanced Reasoning Agent")
    print("------------------------")
    print("This agent can solve problems using various reasoning strategies:")
    print("1. Decomposition")
    print("2. Analogy")
    print("3. Logical Analysis")
    print("4. Mathematical Reasoning")
    print("5. Counterfactual Thinking")
    print("6. Causal Reasoning")
    print("7. Let the agent choose the best strategy")
    
    choice = input("\nChoose a reasoning strategy (1-7): ")
    problem = input("Enter your problem: ")
    
    reasoning_types = {
        "1": ReasongingType.DECOMPOSITION,
        "2": ReasongingType.ANALOGY,
        "3": ReasongingType.LOGICAL,
        "4": ReasongingType.MATHEMATICAL,
        "5": ReasongingType.COUNTERFACTUAL,
        "6": ReasongingType.CAUSAL,
        "7": None
    }
    
    selected_type = reasoning_types.get(choice, None)
    solution = solve_problem(problem, selected_type)
    
    print("\n=== SOLUTION ===\n")
    print(solution)