from duckduckgo_search import DDGS

from ToolAgents.knowledge.web_search.web_search import WebSearchProvider


class DDGWebSearchProvider(WebSearchProvider):

    def search_web(self, search_query: str, num_results: int):
        """
        Search the web, like a google search query. Returns a list of result URLs
        Args:
            search_query (str): Search query.
            num_results (int): Number of results to return.
        Returns:
            List[str]: List of URLs with search results.
        """
        results = DDGS().text(search_query, region='wt-wt', safesearch='off', max_results=num_results)
        return [res["href"] for res in results]
