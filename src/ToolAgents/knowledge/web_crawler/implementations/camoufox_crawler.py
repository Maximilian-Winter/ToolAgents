from camoufox.sync_api import Camoufox

from ToolAgents.knowledge.web_crawler.html2markdown import HTML2Markdown
from ToolAgents.knowledge.web_crawler.web_crawler import WebCrawler


class CamoufoxWebCrawler(WebCrawler):
    def get_website_content_from_url(self, url: str) -> str:
        """
        Get website content from a URL using Selenium and BeautifulSoup for improved content extraction and filtering.

        Args:
            url (str): URL to get website content from.

        Returns:
            str: Extracted content including title, main text, and tables.
        """

        with Camoufox(headless=True) as browser:
            page = browser.new_page()
            page.goto(url, timeout=300000)
            content = page.content()
            page.close()
        return HTML2Markdown().convert(content)

    def get_website_content_from_urls(self, urls: list[str]) -> list[str]:
        results = []
        with Camoufox(headless=True) as browser:
            for url in urls:
                page = browser.new_page()
                page.goto(url, timeout=300000)
                content = page.content()
                results.append(HTML2Markdown().convert(content))
                page.close()
        return results