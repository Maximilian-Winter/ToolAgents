import logging
import os
import time
from enum import Enum
from typing import List, Dict, Any, Optional
import requests

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.utilities.mcp_server import (
    MCPServer,
    MCPServerConfig,
    create_and_run_mcp_server,
)
from ToolAgents.mcp_tool import MCPToolRegistry, MCPTool, MCPToolDefinition
from pydantic import BaseModel, Field


# Define some example tools to expose via the MCP server
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class WeatherUnit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class WeatherInfo(BaseModel):
    """Get current weather information for a location"""

    location: str = Field(
        ..., description="City and state/country, e.g., 'San Francisco, CA'"
    )
    unit: Optional[WeatherUnit] = Field(
        WeatherUnit.CELSIUS, description="Temperature unit"
    )

    def run(self):
        """Get mock weather data for the location"""
        # In a real implementation, this would call a weather API
        mock_temps = {
            "New York": 22,
            "San Francisco": 18,
            "London": 15,
            "Tokyo": 25,
            "Sydney": 28,
        }

        # Find the closest matching city
        closest_city = None
        for city in mock_temps:
            if city.lower() in self.location.lower():
                closest_city = city
                break

        if not closest_city:
            closest_city = list(mock_temps.keys())[
                hash(self.location) % len(mock_temps)
            ]

        temp = mock_temps[closest_city]

        # Convert to Fahrenheit if requested
        if self.unit == WeatherUnit.FAHRENHEIT:
            temp = (temp * 9 / 5) + 32
            unit_str = "°F"
        else:
            unit_str = "°C"

        return {
            "location": self.location,
            "temperature": temp,
            "unit": unit_str,
            "condition": "Sunny",
            "humidity": 65,
            "wind_speed": 10,
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
            return {"expression": self.expression, "result": result}
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
        return {"id": note_id, "content": self.content, "status": "created"}


class GetNote(BaseModel):
    """Retrieve a note from the database"""

    note_id: str = Field(..., description="ID of the note to retrieve")

    def run(self):
        content = note_db.get_note(self.note_id)
        if content is None:
            return {"error": f"Note with ID {self.note_id} not found"}
        return {"id": self.note_id, "content": content}


class UpdateNote(BaseModel):
    """Update an existing note"""

    note_id: str = Field(..., description="ID of the note to update")
    content: str = Field(..., description="New content for the note")

    def run(self):
        success = note_db.update_note(self.note_id, self.content)
        if not success:
            return {"error": f"Note with ID {self.note_id} not found"}
        return {"id": self.note_id, "content": self.content, "status": "updated"}


class DeleteNote(BaseModel):
    """Delete a note from the database"""

    note_id: str = Field(..., description="ID of the note to delete")

    def run(self):
        success = note_db.delete_note(self.note_id)
        if not success:
            return {"error": f"Note with ID {self.note_id} not found"}
        return {"id": self.note_id, "status": "deleted"}


class ListNotes(BaseModel):
    """List all notes in the database"""

    def run(self):
        notes = note_db.list_notes()
        return {
            "notes": [{"id": id, "content": content} for id, content in notes.items()],
            "count": len(notes),
        }


# Create function tools
weather_tool = FunctionTool(WeatherInfo)
calculator_tool = FunctionTool(Calculator)
add_note_tool = FunctionTool(AddNote)
get_note_tool = FunctionTool(GetNote)
update_note_tool = FunctionTool(UpdateNote)
delete_note_tool = FunctionTool(DeleteNote)
list_notes_tool = FunctionTool(ListNotes)

# Create a tool registry
tool_registry = ToolRegistry()
tool_registry.add_tools(
    [
        weather_tool,
        calculator_tool,
        add_note_tool,
        get_note_tool,
        update_note_tool,
        delete_note_tool,
        list_notes_tool,
    ]
)


def start_mcp_server():
    # Configure and start the MCP server
    config = MCPServerConfig(
        host="localhost", port=8000, prefix="/mcp", debug_mode=True
    )

    # Create and run the server in a background thread
    server = create_and_run_mcp_server(tool_registry, config)
    print(f"MCP server started at http://{config.host}:{config.port}{config.prefix}")

    return server


def test_mcp_client():
    """Test client for the MCP server"""
    base_url = "http://localhost:8000/mcp"

    # List available tools
    print("\n=== Available Tools ===")
    response = requests.get(f"{base_url}/tools")
    tools = response.json()

    for tool in tools:
        print(f"- {tool['name']}: {tool['description']}")

    # Test weather tool
    print("\n=== Testing Weather Tool ===")
    response = requests.post(
        f"{base_url}/tools/WeatherInfo",
        json={"parameters": {"location": "New York", "unit": "fahrenheit"}},
    )
    print(f"Response: {response.json()}")

    # Test calculator tool
    print("\n=== Testing Calculator Tool ===")
    response = requests.post(
        f"{base_url}/tools/Calculator",
        json={"parameters": {"expression": "25 * 4 + 10"}},
    )
    print(f"Response: {response.json()}")

    # Test note tools
    print("\n=== Testing Note Tools ===")

    # Add a note
    print("\nAdding a note...")
    response = requests.post(
        f"{base_url}/tools/AddNote",
        json={"parameters": {"content": "Remember to buy milk"}},
    )
    add_result = response.json()
    note_id = add_result["result"]["id"]
    print(f"Added note with ID: {note_id}")

    # Get the note
    print("\nRetrieving the note...")
    response = requests.post(
        f"{base_url}/tools/GetNote", json={"parameters": {"note_id": note_id}}
    )
    print(f"Retrieved note: {response.json()}")

    # Update the note
    print("\nUpdating the note...")
    response = requests.post(
        f"{base_url}/tools/UpdateNote",
        json={
            "parameters": {
                "note_id": note_id,
                "content": "Remember to buy milk and eggs",
            }
        },
    )
    print(f"Updated note: {response.json()}")

    # List all notes
    print("\nListing all notes...")
    response = requests.post(f"{base_url}/tools/ListNotes", json={"parameters": {}})
    print(f"All notes: {response.json()}")

    # Delete the note
    print("\nDeleting the note...")
    response = requests.post(
        f"{base_url}/tools/DeleteNote", json={"parameters": {"note_id": note_id}}
    )
    print(f"Deleted note: {response.json()}")

    # List all notes again
    print("\nListing all notes after deletion...")
    response = requests.post(f"{base_url}/tools/ListNotes", json={"parameters": {}})
    print(f"All notes: {response.json()}")

    print("\nMCP client test completed successfully!")


if __name__ == "__main__":
    # Start the MCP server
    server = start_mcp_server()

    # Wait for the server to start
    print("Waiting for server to start...")
    time.sleep(2)

    try:
        # Run the client test
        test_mcp_client()

        # Keep the server running until interrupted
        print("\nServer is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError: {str(e)}")
