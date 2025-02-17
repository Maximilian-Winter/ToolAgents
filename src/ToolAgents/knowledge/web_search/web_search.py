import abc

class WebSearchProvider(abc.ABC):
    @abc.abstractmethod
    def search_web(self, query: str, number_of_results: int) -> list[str]:
        """Searches the web and returns a list of urls of the result"""
        pass

