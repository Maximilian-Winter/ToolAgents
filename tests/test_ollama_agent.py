from ToolAgents.agents import OllamaAgent
from ToolAgents.data_models.chat_history import ChatHistory

from test_tools import get_flight_times_tool


def run():
    agent = OllamaAgent(model="llama3.1:8b", log_output=False)
    tools = [get_flight_times_tool]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What is the flight time from New York (NYC) to Los Angeles (LAX)?",
        },
    ]

    response = agent.get_response(messages=messages, tools=tools)
    print(response)

    chat_history = ChatHistory()
    chat_history.add_messages_from_dictionaries(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "What is the flight time from London (LHR) to New York (JFK)?",
            },
        ]
    )

    print("\nStreaming response:")
    for chunk in agent.get_streaming_response(
        messages=chat_history.get_messages(),
        tools=tools,
    ):
        print(chunk, end="", flush=True)

    chat_history.add_messages(agent.last_messages_buffer)
    chat_history.save_to_json("./test_chat_history_after_ollama.json")


if __name__ == "__main__":
    run()
