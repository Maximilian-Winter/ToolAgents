from ToolAgents.agents import OllamaAgent
from ToolAgents.utilities import ChatHistory

from test_tools import get_flight_times_tool


def run():

    agent = OllamaAgent(model="llama3.1:8b", debug_output=False)

    tools = [get_flight_times_tool]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the flight time from New York (NYC) to Los Angeles (LAX)?",
        },
    ]

    response = agent.get_response(
        messages=messages,
        tools=tools,
    )

    print(response)

    chat_history = ChatHistory()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the flight time from London (LHR) to New York (JFK)?",
        },
    ]
    chat_history.add_list_of_dicts(messages)

    print("\nStreaming response:")
    for chunk in agent.get_streaming_response(
        messages=chat_history.to_list(),
        tools=tools,
    ):
        print(chunk, end="", flush=True)

    chat_history.add_list_of_dicts(agent.last_messages_buffer)
    chat_history.save_history("./test_chat_history_after_ollama.json")


# Run the function
if __name__ == "__main__":
    run()
