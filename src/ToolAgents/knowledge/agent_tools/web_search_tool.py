from ToolAgents import FunctionTool
from ToolAgents.agents import ChatToolAgent
from ToolAgents.knowledge.utilities.summarizer import summarize_list_of_strings
from ToolAgents.knowledge.web_crawler import WebCrawler
from ToolAgents.knowledge.web_search import WebSearchProvider
from ToolAgents.provider import ChatAPIProvider


class WebSearchTool:
    def __init__(self, web_crawler: WebCrawler, web_provider: WebSearchProvider, number_of_results: int = 3, summarizing_api: ChatAPIProvider = None):
        self.web_crawler = web_crawler
        self.web_provider = web_provider
        self.summarizing_api = summarizing_api
        if self.summarizing_api:
            self.summarizing_agent = ChatToolAgent(chat_api=summarizing_api)
            self.summarizing_settings = summarizing_api.get_default_settings()
        self.number_of_results = number_of_results

    def search_web(self, query: str):
        """
        Performs a web search like a google search and returns a list of the results.
        Args:
            query (str): The search query.
        Returns:
            list: The results as a list.
        """
        urls = self.web_provider.search_web(query, self.number_of_results)
        all_content = self.web_crawler.get_website_content_from_urls(urls)
        if self.summarizing_api:
            all_content = summarize_list_of_strings(self.summarizing_agent, self.summarizing_settings, all_content)

        final_results = ""
        for url, content in zip(urls, all_content):
            final_results += f"Website: '{url}'\nContent:\n{content}\n\n---\n\n"
        return final_results.strip()

    def get_tool(self):
        return FunctionTool(self.search_web)