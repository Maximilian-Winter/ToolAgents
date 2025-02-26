import os
import time
from enum import Enum
from typing import List, Dict, Any, Optional
import requests

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.mcp_server import MCPServer, MCPServerConfig, create_and_run_mcp_server
from ToolAgents.mcp_tool import MCPToolRegistry, MCPTool, MCPToolDefinition

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# ---------- Step 1: Define tools to expose via the MCP server ----------

class WeatherUnit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

class WeatherInfo(BaseModel):
    """Get current weather information for a location"""
    location: str = Field(..., description="City and state/country, e.g., 'San Francisco, CA'")
    unit: Optional[WeatherUnit] = Field(WeatherUnit.CELSIUS, description="Temperature unit")
    
    def run(self):
        """Get mock weather data for the location"""
        # In a real implementation, this would call a weather API
        mock_temps = {
            "New York": 22,
            "San Francisco": 18,
            "London": 15,
            "Tokyo": 25,
            "Sydney": 28
        }
        
        # Find the closest matching city
        closest_city = None
        for city in mock_temps:
            if city.lower() in self.location.lower():
                closest_city = city
                break
        
        if not closest_city:
            closest_city = list(mock_temps.keys())[hash(self.location) % len(mock_temps)]
        
        temp = mock_temps[closest_city]
        
        # Convert to Fahrenheit if requested
        if self.unit == WeatherUnit.FAHRENHEIT:
            temp = (temp * 9/5) + 32
            unit_str = "°F"
        else:
            unit_str = "°C"
        
        return {
            "location": self.location,
            "temperature": temp,
            "unit": unit_str,
            "condition": "Sunny",
            "humidity": 65,
            "wind_speed": 10
        }

class Calculator(BaseModel):
    """Perform a mathematical calculation"""
    expression: str = Field(..., description="Mathematical expression to evaluate")
    
    def run(self):
        """Evaluate the mathematical expression"""
        # Warning: Using eval is generally not safe for production use
        # This is just for demonstration purposes
        try:
            # Make sure the expression only contains mathematical operations
            safe_chars = set("0123456789+-*/().%^ ")
            if not all(c in safe_chars for c in self.expression):
                return {"error": "Expression contains invalid characters"}
            
            # Replace unsafe operations
            expression = self.expression.replace("^", "**")
            
            result = eval(expression)
            return {
                "expression": self.expression,
                "result": result
            }
        except Exception as e:
            return {"error": f"Error calculating result: {str(e)}"}

class NoteDB:
    """Simple in-memory database for notes"""
    def __init__(self):
        self.notes = {}
        self.next_id = 1
    
    def add_note(self, content):
        note_id = str(self.next_id)
        self.notes[note_id] = content
        self.next_id += 1
        return note_id
    
    def get_note(self, note_id):
        return self.notes.get(note_id)
    
    def update_note(self, note_id, content):
        if note_id in self.notes:
            self.notes[note_id] = content
            return True
        return False
    
    def delete_note(self, note_id):
        if note_id in self.notes:
            del self.notes[note_id]
            return True
        return False
    
    def list_notes(self):
        return {id: content for id, content in self.notes.items()}

# Global note database instance
note_db = NoteDB()

class AddNote(BaseModel):
    """Add a new note to the database"""
    content: str = Field(..., description="Content of the note")
    
    def run(self):
        note_id = note_db.add_note(self.content)
        return {
            "id": note_id,
            "content": self.content,
            "status": "created"
        }

class GetNote(BaseModel):
    """Retrieve a note from the database"""
    note_id: str = Field(..., description="ID of the note to retrieve")
    
    def run(self):
        content = note_db.get_note(self.note_id)
        if content is None:
            return {"error": f"Note with ID {self.note_id} not found"}
        return {
            "id": self.note_id,
            "content": content
        }

class ListNotes(BaseModel):
    """List all notes in the database"""
    
    def run(self):
        notes = note_db.list_notes()
        return {
            "notes": [{"id": id, "content": content} for id, content in notes.items()],
            "count": len(notes)
        }

# Create function tools
weather_tool = FunctionTool(WeatherInfo)
calculator_tool = FunctionTool(Calculator)
add_note_tool = FunctionTool(AddNote)
get_note_tool = FunctionTool(GetNote)
list_notes_tool = FunctionTool(ListNotes)

# Create a tool registry for the server
server_tool_registry = ToolRegistry()
server_tool_registry.add_tools([
    weather_tool,
    calculator_tool,
    add_note_tool,
    get_note_tool,
    list_notes_tool
])

# ---------- Step 2: Create and start the MCP server ----------

def start_mcp_server():
    """Start the MCP server with our tools"""
    config = MCPServerConfig(
        host="localhost",
        port=8000,
        prefix="/mcp",
        debug_mode=True
    )
    
    # Create and run the server in a background thread
    server = create_and_run_mcp_server(server_tool_registry, config)
    print(f"MCP server started at http://{config.host}:{config.port}{config.prefix}")
    
    return server, config

# ---------- Step 3: Create an agent that uses the MCP tools ----------

def initialize_mcp_client_agent(server_config):
    """Initialize an agent that uses tools from the MCP server"""
    # Set up API client
    api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
    settings = api.get_default_settings()
    settings.temperature = 0.2
    agent = ChatToolAgent(chat_api=api)
    
    # Create a tool registry for the client
    client_tool_registry = ToolRegistry()
    
    # Create MCPToolDefinition objects for each tool
    # In a real scenario, you would discover these from the server
    server_url = f"http://{server_config.host}:{server_config.port}{server_config.prefix}"
    
    # Discover tools from the MCP server
    print(f"Discovering tools from MCP server at {server_url}...")
    discovered_tools = MCPToolRegistry.discover_tools_from_server(server_url)
    print(f"Discovered {len(discovered_tools)} tools")
    
    # Add the discovered tools to the client registry
    MCPToolRegistry.add_mcp_tools_to_registry(client_tool_registry, discovered_tools)
    
    # Initialize chat history
    chat_history = ChatHistory()
    chat_history.add_system_message("""
    You are an assistant with access to various tools from a Model Context Protocol server.
    When a user asks for information that requires using a tool, use the appropriate tool to get
    the information and respond based on the result.

    Please help the user with their requests by using these tools when appropriate.
    """)
    
    return agent, settings, client_tool_registry, chat_history

# ---------- Step 4: Run a conversation with the agent using MCP tools ----------

def run_conversation(agent, settings, tool_registry, chat_history):
    """Run an interactive conversation with the agent"""
    print("\n=== MCP Agent Conversation ===")
    print("Type 'exit' to end the conversation")
    print("Type 'tools' to see available tools")
    
    while True:
        user_input = input("\nYou > ")
        
        if user_input.lower() == "exit":
            break
            
        elif user_input.lower() == "tools":
            print("\nAvailable tools:")
            for name, tool in tool_registry.tools.items():
                openai_tool = tool.to_openai_tool()
                print(f"- {name}: {openai_tool['function']['description']}")
            continue
            
        # Add the user's message to the chat history
        chat_history.add_user_message(user_input)
        
        # Get a response from the agent
        print("\nAgent is thinking...")
        response = agent.get_streaming_response(
            messages=chat_history.get_messages(),
            settings=settings,
            tool_registry=tool_registry
        )
        
        # Process the streaming response
        print("Agent > ", end="")
        completed_response = None
        
        for chunk in response:
            print(chunk.chunk, end="", flush=True)
            
            # If this is the final chunk, save the completed response
            if chunk.finished:
                completed_response = chunk.finished_response
        
        print()  # Add a newline after the response
        
        # Add the agent's messages to the chat history
        if completed_response:
            chat_history.add_messages(completed_response.messages)

# ---------- Main ----------

if __name__ == "__main__":
    print("Model Context Protocol (MCP) Full Example")
    print("----------------------------------------")
    print("This example shows how to create an MCP server and an agent that uses MCP tools.")
    
    try:
        # Start the MCP server
        server, server_config = start_mcp_server()
        
        # Wait for the server to start
        print("Waiting for server to start...")
        time.sleep(2)
        
        # Initialize the agent
        agent, settings, tool_registry, chat_history = initialize_mcp_client_agent(server_config)
        
        # Run the conversation
        run_conversation(agent, settings, tool_registry, chat_history)
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    print("\nThank you for using the MCP Full Example!")