import markdownify
from camoufox.sync_api import Camoufox



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

        with Camoufox() as browser:
            page = browser.new_page()
            page.goto(url, timeout=300000)
            content = page.content()
            page.close()
        return markdownify.markdownify(content)

    def get_website_content_from_urls(self, urls: list[str]) -> list[str]:
        results = []
        with Camoufox() as browser:
            for url in urls:
                page = browser.new_page()
                page.goto(url, timeout=300000)
                content = page.content()
                results.append(markdownify.markdownify(content))
                page.close()
        return results