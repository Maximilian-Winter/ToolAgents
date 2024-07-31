from FunkyFlow.agents.ollama_agent import OllamaAgent
from VirtualGameMasterFunctionCalling.function_calling import FunctionTool

from pydantic import BaseModel, Field
import json
from typing import Dict, Any


class FlightTimes(BaseModel):
    """
    A class to represent flight times between two locations.

    This class uses Pydantic for data validation and provides a method
    to retrieve flight information based on departure and arrival locations.
    """

    departure: str = Field(
        ...,
        description="The departure location (airport code)",
        min_length=3,
        max_length=3
    )
    arrival: str = Field(
        ...,
        description="The arrival location (airport code)",
        min_length=3,
        max_length=3
    )

    class Config:
        """Pydantic configuration class"""
        schema_extra = {
            "example": {
                "departure": "NYC",
                "arrival": "LAX"
            }
        }

    def run(self) -> str:
        """
        Retrieve flight information for the given departure and arrival locations.

        Returns:
            str: A JSON string containing flight information including departure time,
                 arrival time, and flight duration. If no flight is found, returns an error message.
        """
        flights: Dict[str, Dict[str, str]] = {
            'NYC-LAX': {'departure': '08:00 AM', 'arrival': '11:30 AM', 'duration': '5h 30m'},
            'LAX-NYC': {'departure': '02:00 PM', 'arrival': '10:30 PM', 'duration': '5h 30m'},
            'LHR-JFK': {'departure': '10:00 AM', 'arrival': '01:00 PM', 'duration': '8h 00m'},
            'JFK-LHR': {'departure': '09:00 PM', 'arrival': '09:00 AM', 'duration': '7h 00m'},
            'CDG-DXB': {'departure': '11:00 AM', 'arrival': '08:00 PM', 'duration': '6h 00m'},
            'DXB-CDG': {'departure': '03:00 AM', 'arrival': '07:30 AM', 'duration': '7h 30m'},
        }

        key: str = f'{self.departure}-{self.arrival}'.upper()
        result: Dict[str, Any] = flights.get(key, {'error': 'Flight not found'})
        return json.dumps(result)


def run():
    agent = OllamaAgent(model='mistral-nemo', debug_output=True)

    get_flight_times_tool = FunctionTool(
        function_tool=FlightTimes
    )

    tools = [get_flight_times_tool]

    response = agent.get_response(
        message="What is the flight time from New York (NYC) to Los Angeles (LAX)?",
        tools=tools,
    )

    print(response)

    print("\nStreaming response:")
    for chunk in agent.get_streaming_response(
            message="What is the flight time from London (LHR) to New York (JFK)?",
            tools=tools,
    ):
        print(chunk, end='', flush=True)


# Run the function
if __name__ == "__main__":
    run()
