from ToolAgents.agents import LlamaAgent
from ToolAgents.provider import LlamaCppSamplingSettings, LlamaCppServerProvider
from ToolAgents.provider import VLLMServerSamplingSettings, \
    VLLMServerProvider
from ToolAgents.utilities import ChatHistory
from code_executer import PythonCodeExecutor, system_message_code_agent

provider = LlamaCppServerProvider("http://127.0.0.1:8080/")

agent = LlamaAgent(llm_provider=provider, debug_output=True)

settings = LlamaCppSamplingSettings()
settings.temperature = 0.3
settings.top_p = 1.0
settings.top_k = 0
settings.min_p = 0.0
settings.max_tokens = 4096
settings.stop = ["</s>", "<|eom_id|>", "<|eot_id|>", "assistant", "<|start_header_id|>assistant<|end_header_id|>"]

chat_history = ChatHistory()
chat_history.add_system_message(system_message_code_agent)

python_code_executor = PythonCodeExecutor()


def run_code_agent(user):
    chat_history.add_user_message(user)
    result_gen = agent.get_streaming_response(
        messages=chat_history.to_list(),
        sampling_settings=settings, tools=None)

    full_response = ""
    for tok in result_gen:
        print(tok, end="", flush=True)
        full_response += tok
    print()
    while True:
        chat_history.add_assistant_message(full_response)
        if "```python_interpreter" in full_response:
            chat_history.add_tool_message(python_code_executor.run(full_response))
            result_gen = agent.get_streaming_response(
                messages=chat_history.to_list(),
                sampling_settings=settings, tools=None)
            full_response = ""
            for tok in result_gen:
                print(tok, end="", flush=True)
                full_response += tok
            print()
        else:
            break


prompt = r"""Create a graph of x^2 + 5 with your Python Code Interpreter and save it as an image."""
prompt2 = r"""Create an interesting and engaging random 3d scatter plot with your Python Code Interpreter and save it as an image."""
prompt3 = r"""Analyze and visualize the dataset "./input.csv" with your Python code interpreter as a interesting and visually appealing scatterplot matrix."""

run_code_agent(prompt)
run_code_agent(prompt2)
run_code_agent(prompt3)

chat_history.save_history("./test_chat_history_after_llama.json")
