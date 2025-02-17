import abc


class WebCrawler(abc.ABC):
    @abc.abstractmethod
    def get_website_content_from_url(self, url: str) -> str:
        """Get the website content from an url."""
        pass

    @abc.abstractmethod
    def get_website_content_from_urls(self, urls: list[str]) -> list[str]:
        """Get the website content from urls."""
        pass