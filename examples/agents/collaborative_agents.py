import os
from typing import List, Dict, Any

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI, AnthropicChatAPI
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

class AgentResponse(BaseModel):
    """Response from an agent for a certain task"""
    task: str = Field(..., description="The task that was assigned to the agent")
    response: str = Field(..., description="The agent's response to the task")

class CollaborativeTask(BaseModel):
    """Define a task for collaborative agents"""
    task_description: str = Field(..., description="Detailed description of the task to be performed")
    agent_role: str = Field(..., description="The specialized role this agent should take (e.g., 'critic', 'researcher', 'implementer')")
    
    def run(self):
        return f"Task assigned: {self.task_description} to agent with role: {self.agent_role}"

class AgentFeedback(BaseModel):
    """Provide feedback on another agent's work"""
    agent_response: str = Field(..., description="The response from another agent to review")
    feedback_type: str = Field(..., description="Type of feedback (e.g., 'critique', 'suggestion', 'question')")
    
    def run(self):
        return f"Reviewing response with {self.feedback_type} feedback"

class TaskSynthesis(BaseModel):
    """Synthesize work from multiple agents"""
    agent_responses: List[str] = Field(..., description="List of responses from different agents")
    
    def run(self):
        return f"Synthesizing {len(self.agent_responses)} agent responses"

# Create the function tools
collaborative_task_tool = FunctionTool(CollaborativeTask)
agent_feedback_tool = FunctionTool(AgentFeedback)
task_synthesis_tool = FunctionTool(TaskSynthesis)

# Set up OpenAI API client for the coordinator agent
coordinator_api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
coordinator_settings = coordinator_api.get_default_settings()
coordinator_settings.temperature = 0.4
coordinator_agent = ChatToolAgent(chat_api=coordinator_api)

# Set up Anthropic API client for the specialized agents
specialist_api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-haiku-20240307")
specialist_settings = specialist_api.get_default_settings()
specialist_settings.temperature = 0.7
specialist_agent = ChatToolAgent(chat_api=specialist_api)

# Create tool registries
coordinator_tools = ToolRegistry()
coordinator_tools.add_tools([collaborative_task_tool, task_synthesis_tool])

specialist_tools = ToolRegistry()
specialist_tools.add_tools([agent_feedback_tool])

# Initialize chat histories
coordinator_history = ChatHistory()
coordinator_history.add_system_message("""
You are a Coordinator Agent responsible for managing a team of specialized AI agents.
Your job is to:
1. Break down complex tasks into subtasks
2. Assign these subtasks to specialized agents with specific roles
3. Synthesize their work into a cohesive final output
4. Ensure the team's work addresses the original request comprehensively

Use the CollaborativeTask tool to assign tasks to specialized agents.
Use the TaskSynthesis tool to combine their work.

Be clear, specific, and strategic in your task assignments.
""")

# Dictionary to hold histories for specialist agents in different roles
specialist_histories = {}

def initialize_specialist(role: str) -> ChatHistory:
    """Initialize a chat history for a specialist agent with a specific role"""
    history = ChatHistory()
    history.add_system_message(f"""
You are a specialized {role} agent. Your job is to:
1. Complete the assigned tasks related to your specialization
2. Provide detailed, high-quality responses within your domain of expertise
3. Use the AgentFeedback tool to critique or provide suggestions on other agents' work

Focus on bringing your unique perspective as a {role} to every task.
""")
    return history

def run_coordinator(user_input: str) -> Dict[str, Any]:
    """Run the coordinator agent to break down the task"""
    coordinator_history.add_user_message(user_input)
    
    response = coordinator_agent.get_response(
        messages=coordinator_history.get_messages(),
        settings=coordinator_settings,
        tool_registry=coordinator_tools
    )
    
    coordinator_history.add_messages(response.messages)
    return response.response, response.messages

def run_specialist(role: str, task: str) -> str:
    """Run a specialist agent on a specific task"""
    # Initialize history for this role if it doesn't exist
    if role not in specialist_histories:
        specialist_histories[role] = initialize_specialist(role)
    
    # Add the task to the specialist's history
    specialist_histories[role].add_user_message(task)
    
    # Get response from the specialist
    response = specialist_agent.get_response(
        messages=specialist_histories[role].get_messages(),
        settings=specialist_settings,
        tool_registry=specialist_tools
    )
    
    # Update the specialist's history
    specialist_histories[role].add_messages(response.messages)
    return response.response

def collaborative_workflow(user_request: str) -> str:
    """Execute a full collaborative workflow given a user request"""
    # Step 1: Coordinator breaks down the task
    coordinator_output, messages = run_coordinator(
        f"Please break down this request into specialized tasks for our agent team: {user_request}"
    )
    
    # Extract tasks for specialists
    tasks = {}
    for msg in messages:
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call["name"] == "CollaborativeTask":
                    args = tool_call["arguments"]
                    tasks[args["agent_role"]] = args["task_description"]
    
    # Step 2: Run each specialist agent on their assigned task
    specialist_outputs = {}
    for role, task in tasks.items():
        specialist_outputs[role] = run_specialist(role, task)
    
    # Step 3: Have the coordinator synthesize the results
    synthesis_input = f"""
Here are the outputs from our specialist agents:

{chr(10).join([f'--- {role} ---{chr(10)}{output}{chr(10)}' for role, output in specialist_outputs.items()])}

Please synthesize these outputs into a comprehensive response to the original request: "{user_request}"
"""
    final_output, _ = run_coordinator(synthesis_input)
    
    return final_output

# Example usage
if __name__ == "__main__":
    user_request = input("Enter your request: ")
    final_response = collaborative_workflow(user_request)
    print("\n=== FINAL RESPONSE ===\n")
    print(final_response)