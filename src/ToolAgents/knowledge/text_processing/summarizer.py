from typing import Any

from ToolAgents.agents import ChatToolAgent
from ToolAgents.function_tool import PostProcessor
from ToolAgents.messages import ChatMessage, MessageTemplate
from ToolAgents.provider import ProviderSettings


def summarize_list_of_strings(agent: ChatToolAgent, settings: ProviderSettings, strings: list[str]):
    template = MessageTemplate.from_string("""You are an expert document analyst and summarizer. Your task is to create a comprehensive summary of multiple documents. Here are the documents you need to analyze and summarize:

<documents>
{{DOCUMENTS}}
</documents>

Please follow these steps to create your summary:

1. Carefully read and analyze all the documents provided above.
2. Identify the main themes, key arguments, and important facts across all documents.
3. Look for connections, similarities, and differences between the documents.
4. Synthesize the information into a unified summary.

Before writing your final summary, break down your thought process inside <document_analysis> tags. Consider the following:
- Create a brief outline for each document, listing key points and themes.
- Identify and list common themes across all documents.
- Compare and contrast the main arguments from each document.
- List any significant data or statistics from the documents.
- What conclusions can be drawn from the collective information?

It's OK for this section to be quite long. Remember that your final summary should be approximately 10% of the total length of all input documents combined, but no shorter than 250 words and no longer than 1000 words.

After your analysis, create a summary that meets these criteria:
- Content: Comprehensive coverage of all major points from the original documents, avoiding unnecessary repetition.
- Organization: Logical and coherent structure with appropriate transitions between ideas.
- Language: Clear and concise, avoiding jargon unless essential to understanding the content.
- Tone: Objective, accurately representing the information without personal opinions or interpretations.

Structure your summary as follows:
1. Introduction: Briefly outline the main topic or purpose of the documents.
2. Body: Organize the main content by themes or topics.
3. Conclusion: Concisely tie together the key points.

Do not include citations or references to specific documents in your summary.

Please provide your final summary within <summary> tags.

Example output structure (replace with actual content):

<document_analysis>
Document 1 Outline:
- Key point 1
- Key point 2
- Main theme

Document 2 Outline:
- Key point 1
- Key point 2
- Main theme

Common Themes:
1. [Theme 1 description]
2. [Theme 2 description]

Argument Comparison:
- Document 1 argues [X], while Document 2 contends [Y]
- Both documents agree on [Z]

Significant Data/Statistics:
- Statistic 1 from Document [X]
- Statistic 2 from Document [Y]

Conclusions:
- [Conclusion 1 based on collective information]
- [Conclusion 2 based on collective information]
</document_analysis>

<summary>
[Introduction paragraph outlining the main topic]

[Body paragraph 1 discussing Theme 1]

[Body paragraph 2 discussing Theme 2]

[Body paragraph 3 discussing key arguments and important facts]

[Conclusion paragraph tying together the key points]
</summary>""")
    results = []
    for doc in strings:
        messages = [
            ChatMessage.create_system_message("You are a helpful assistant."),
            ChatMessage.create_user_message(template.generate_message_content(DOCUMENTS=doc))
        ]
        print(messages[1].model_dump_json(indent=2))
        chat_response = agent.get_response(
            messages=messages,
            settings=settings)
        print(chat_response.response.strip())
        results.append(chat_response.response.strip())
    return results



class SummarizingFunctionToolPostProcessor(PostProcessor):
    def __init__(self, summarizing_agent: ChatToolAgent, settings: ProviderSettings):
        super().__init__()
        self.summarizing_agent = summarizing_agent
        self.settings = settings

    def process(self, result: Any) -> Any:
        return summarize_list_of_strings(self.summarizing_agent, self.settings, [f"{result}"])[0]