import datetime
import os
import json
from typing import List, Dict, Any, Optional

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI, AnthropicChatAPI
from ToolAgents.agent_tools.web_search_tool import WebSearchTool
from ToolAgents.knowledge.web_search.implementations.googlesearch import GoogleWebSearchProvider
from ToolAgents.knowledge.web_crawler.implementations.camoufox_crawler import CamoufoxWebCrawler
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()


def setup_deep_research_agent():
    """Set up the deep research agent with all necessary tools"""
    # Set up the primary LLM agent
    primary_api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-7-sonnet-20250219")
    
    # Get default settings with appropriate temperature
    settings = primary_api.get_default_settings()
    settings.temperature = 0.2  # Lower temperature for research tasks

    primary_api.set_default_settings(settings)

    # Create the agent
    agent = ChatToolAgent(chat_api=primary_api)
    
    # Set up web search capabilities
    web_crawler = CamoufoxWebCrawler()
    web_search_provider = GoogleWebSearchProvider()
    
    # Set up a summarization API (can be the same as the primary)
    summary_api = primary_api
    
    # Create the web search tool
    web_search_tool = WebSearchTool(
        web_crawler=web_crawler, 
        web_provider=web_search_provider,
        summarizing_api=summary_api,
        number_of_results=3  # Default number of results per search
    )
    
    # Create a registry for all tools
    tool_registry = ToolRegistry()
    
    # Add the web search tool
    tool_registry.add_tool(web_search_tool.get_tool())

    # Initialize chat history with system prompt
    chat_history = ChatHistory()
    chat_history.add_system_message("""You are DeepResearch, an advanced research agent specialized in conducting iterative, in-depth investigations on any topic. 
Your research methodology follows these principles:

1. EXPLORE BROADLY: Begin with a broad exploration of the topic to understand the landscape.
2. IDENTIFY KEY SUBTOPICS: Extract the most important subtopics, concepts, and areas for deeper investigation.
3. ITERATIVE DEEPENING: For each important subtopic, conduct a new focused search to gather more specific information.
4. CROSS-REFERENCE: Look for connections, contradictions, and patterns across different sources and subtopics.
5. SYNTHESIZE: Combine all findings into a comprehensive, well-structured research report.

Your research process should be:
- THOROUGH: Leave no important angle unexplored
- ITERATIVE: Use findings from each search to inform subsequent searches
- TRANSPARENT: Document your research process and reasoning
- CRITICAL: Evaluate sources and contradictory information
- OBJECTIVE: Present multiple perspectives when relevant

When conducting research:
1. Start by exploring the general topic with search_web
2. Identify 3-5 key subtopics or questions for deeper investigation
3. For each subtopic, perform a new focused search_web query
4. Extract new insights and potential areas to explore even deeper
5. Continue this iterative process until you reach the max_search_depth
6. Synthesize all findings into a comprehensive research report

Your final output should be a well-organized research report with:
- Executive summary of main findings
- Detailed exploration of the topic with supporting evidence
- Multiple perspectives when the topic is debated
- Clear organization with sections and subsections
- Citations to sources (URLs)
- Recommendations for further research

Always prioritize quality of research over speed. Take your time to thoroughly explore each important aspect of the topic.
""")
    
    return agent, settings, tool_registry, chat_history

def conduct_deep_research(query: str, max_depth: int = 3, results_per_search: int = 3):
    """
    Conduct deep, iterative research on the provided query
    
    Args:
        query: The research topic or question
        max_depth: Maximum depth of iterative searches (1-5)
        results_per_search: Number of results per search query (1-5)
    
    Returns:
        The comprehensive research report
    """
    # Set up the agent
    agent, settings, tool_registry, chat_history = setup_deep_research_agent()
    
    # Format the research request
    research_prompt = f"""The current date and time is (Format: %Y-%m-%d %H:%M:%S): {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} 

Please conduct a deep, iterative research investigation on the following topic:

RESEARCH TOPIC: {query}


Use the following research parameters:
- Maximum search depth: {max_depth} levels
- Results per search: {results_per_search}

Follow these steps:
1. Start with a broad search to understand the general landscape of the topic
2. Identify 3-5 key subtopics, concepts, or questions that need deeper investigation
3. For each important subtopic, conduct a new focused search
4. From these results, identify additional aspects that need even deeper research
5. Continue this iterative process until you reach the maximum search depth
6. Synthesize all findings into a comprehensive, well-structured research report

Document your research process by:
- Explaining your search strategy for each iteration
- Noting the most important findings from each search
- Describing how each finding influences your next research steps

Your final report should include:
- An executive summary of your main findings
- A detailed exploration of the topic with supporting evidence
- Multiple perspectives when the topic is debated
- Clear organization with sections and subsections
- Citations to sources (URLs)
- Recommendations for further research"""
    
    # Add the user's message to the chat history
    chat_history.add_user_message(research_prompt)
    
    # Get a response from the agent
    print(f"Conducting deep research on: {query}")
    print(f"Maximum depth: {max_depth}, Results per search: {results_per_search}")
    print("Research in progress. This may take some time...")
    
    response = agent.get_response(
        messages=chat_history.get_messages(),
        settings=settings,
        tool_registry=tool_registry
    )
    
    # Add the agent's messages to the chat history
    chat_history.add_messages(response.messages)
    
    # Return the final research report
    return response.response

def save_research_report(report: str, query: str, filename: Optional[str] = None):
    """Save the research report to a file"""
    if not filename:
        # Create a filename based on the query
        clean_query = "".join(c if c.isalnum() else "_" for c in query)
        clean_query = clean_query[:30]  # Limit length
        filename = f"research_report_{clean_query}.md"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Research report saved to {filename}")
    return filename

if __name__ == "__main__":
    print("Deep Research Agent")
    print("==================")
    print("This agent conducts iterative, in-depth research on any topic.")
    print("It uses a multi-level approach, exploring subtopics in increasing detail.")
    
    # Get the research query from the user
    query = input("\nEnter your research topic or question: ")
    
    # Get optional parameters
    try:
        depth_input = input("Maximum search depth (1-5) [default: 3]: ")
        max_depth = int(depth_input) if depth_input.strip() else 3
        max_depth = max(1, min(5, max_depth))  # Ensure between 1-5
        
        results_input = input("Results per search (1-5) [default: 3]: ")
        results_per_search = int(results_input) if results_input.strip() else 3
        results_per_search = max(1, min(5, results_per_search))  # Ensure between 1-5
    except ValueError:
        print("Invalid input. Using default values.")
        max_depth = 3
        results_per_search = 3
    
    # Conduct the research
    try:
        report = conduct_deep_research(query, max_depth, results_per_search)
        
        # Save the report
        save_option = input("\nSave report to file? (y/n) [default: y]: ")
        if save_option.lower() != "n":
            filename = save_research_report(report, query)
        
        # Print the report
        print("\n=== RESEARCH REPORT ===\n")
        print(report)
        
    except Exception as e:
        print(f"An error occurred during research: {str(e)}")