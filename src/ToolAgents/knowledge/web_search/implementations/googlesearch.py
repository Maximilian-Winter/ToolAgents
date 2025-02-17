from googlesearch import search

from ToolAgents.knowledge.web_search.web_search import WebSearchProvider


class GoogleWebSearchProvider(WebSearchProvider):
    def search_web(self, search_query: str, num_results: int):
        """
        Search the web, like a google search query. Returns a list of result URLs
        Args:
            search_query (str): Search query.
            num_results (int): Number of results to return.
        Returns:
            List[str]: List of URLs with search results.
        """
        try:
            # Only return the top 5 results for simplicity
            return list(search(search_query, num_results=num_results))
        except Exception as e:
            return f"An error occurred during Google search: {str(e)}"
