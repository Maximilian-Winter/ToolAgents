from ToolAgents.messages.chat_history import ChatHistory, AdvancedChatFormatter

chat_history = ChatHistory()
chat_history.load_history("./test_chat_history.json")

llama_31_system_template = "<|start_header_id|>system<|end_header_id|>\n\n{content}\n<|eot_id|>"
llama_31_assistant_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{content}\n<|eot_id|>"
llama_31_user_template = "<|start_header_id|>user<|end_header_id|>\n\n{content}\n<|eot_id|>"

llama_31_chat_formatter = AdvancedChatFormatter({
    "system": llama_31_system_template,
    "user": llama_31_user_template,
    "assistant": llama_31_assistant_template,
})

result = llama_31_chat_formatter.format_messages(chat_history.to_list())

print(result)
