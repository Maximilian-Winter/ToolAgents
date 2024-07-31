from ToolAgents.agents.mistral_agent import MistralAgent
from ToolAgents.agents.ollama_agent import OllamaAgent
from ToolAgents.provider.llama_cpp_server import LlamaCppServerProvider
from ToolAgents.tests.test_tools import get_flight_times_tool


def run():

    agent = OllamaAgent(model='mistral-nemo', system_prompt="You are a crazy, old and drunken pirate.", debug_output=False)

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
