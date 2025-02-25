import os

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.knowledge.agent_tools.web_search_tool import WebSearchTool
from ToolAgents.messages import ChatMessage

from ToolAgents.provider import AnthropicChatAPI, OpenAIChatAPI
from ToolAgents.knowledge.web_search.implementations.googlesearch import GoogleWebSearchProvider
from ToolAgents.knowledge.web_crawler.implementations.camoufox_crawler import CamoufoxWebCrawler
from dotenv import load_dotenv


load_dotenv()

# Local OpenAI like API, like vllm or llama-cpp-server
# Groq API
api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")
settings = api.get_default_settings()
settings.temperature = 0.45
# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api, debug_output=True)
web_crawler = CamoufoxWebCrawler()
web_search_provider = GoogleWebSearchProvider()


summary_api = OpenAIChatAPI(api_key="xxx", base_url="http://127.0.0.1:8080/v1", model="xxx")
summary_settings = summary_api.get_default_settings()
summary_settings.temperature = 0.35
summary_api.set_default_settings(summary_settings)

web_search_tool = WebSearchTool(web_crawler=web_crawler, web_provider=web_search_provider, summarizing_api=summary_api)

tool_registry = ToolRegistry()

tool_registry.add_tool(web_search_tool.get_tool())


messages = [
    ChatMessage.create_user_message("""You are an expert research engineer tasked with conducting comprehensive research based on user requests. Your goal is to create an in-depth, well-structured document that thoroughly explores the given topic. Here's the user's request:

<user_request>
Advancements in reasoning of large language models.
</user_request>

Follow these steps to complete your task:

1. Initial Research Planning:
Before beginning your web searches, use <research_planning> tags to outline your research strategy. Include the following:
- List 5-7 key aspects of the topic to investigate
- Propose 3-5 potential sources of information
- Formulate 3-5 specific questions that need to be answered
- Identify 2-3 possible challenges in the research process

2. Web Search Process:
Conduct a series of web searches to gather information using your tools. Start with a broad search based on the user's request, then perform more specific searches as needed.

Continue this process until you have gathered sufficient information to create a comprehensive document.

3. Information Synthesis:
As you collect information, synthesize it within your <research_planning> tags. Include:
- List 3-5 key findings from each search
- Identify 2-3 connections between different pieces of information
- Note any conflicting viewpoints or data
- Emerging patterns or themes
- How the information relates to the main topic

4. Document Creation:
Based on your research and synthesis, create a comprehensive document that explores the topic in depth. Your document should:
- Provide thorough analysis
- Present information in a logical, well-structured manner
- Include relevant examples, case studies, or anecdotes
- Address potential counterarguments or alternative viewpoints
- Cite sources for key information or claims

5. Output Format:
Present your findings using the following structure:

<research_document>
<title>[An appropriate title for your research]</title>

<executive_summary>
[A brief overview of the main findings and key points]
</executive_summary>

<main_content>
[Your comprehensive research, organized into logical sections with appropriate headings]
</main_content>

<conclusions>
[Summary of main takeaways, implications, and potential future directions]
</conclusions>

<sources>
[List of main sources used in your research]
</sources>
</research_document>

Example output structure (note: this is a generic example; your actual content will be much more detailed):

<research_document>
<title>Comprehensive Analysis of [Topic]</title>

<executive_summary>
This research explores [topic], covering its historical context, current developments, and future implications. Key findings include [brief points].
</executive_summary>

<main_content>
1. Historical Context
   [Detailed information]

2. Current Developments
   [Detailed information]

3. Future Implications
   [Detailed information]

4. Debates and Perspectives
   [Detailed information]

5. Relevant Statistics and Data
   [Detailed information]

6. Expert Opinions and Research Findings
   [Detailed information]
</main_content>

<conclusions>
In conclusion, [summary of main points]. This research suggests [implications]. Future studies could focus on [potential directions].
</conclusions>

<sources>
1. [Source 1]
2. [Source 2]
3. [Source 3]
...
</sources>
</research_document>

Remember to maintain a neutral and objective tone throughout your document. Your goal is to provide a thorough and balanced exploration of the topic based on the user's request.

Begin your research process now, starting with your initial research planning.""")
]


chat_response = agent.get_response(
    messages=messages,
    settings=settings, tool_registry=tool_registry)

print(chat_response.response.strip())

print('\n'.join([msg.model_dump_json(indent=4) for msg in chat_response.messages]))