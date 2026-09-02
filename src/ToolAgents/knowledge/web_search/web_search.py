import abc


class WebSearchProvider(abc.ABC):
    """A source of web search results.

    Implementations return URLs only; fetching and extracting the page
    contents is a :class:`~ToolAgents.knowledge.web_crawler.web_crawler.WebCrawler`'s
    job.
    """

    @abc.abstractmethod
    def search_web(self, search_query: str, num_results: int) -> list[str]:
        """Search the web and return the result URLs.

        Args:
            search_query: The query to search for.
            num_results: Maximum number of URLs to return.

        Returns:
            list[str]: URLs of the results.
        """
        pass
