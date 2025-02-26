import json

from trafilatura import fetch_url, extract

from ToolAgents.knowledge.web_crawler.web_crawler import WebCrawler


class TrafilaturaWebCrawler(WebCrawler):
    def get_website_content_from_url(self, url: str) -> str:
        """
        Get website content from a URL using Selenium and BeautifulSoup for improved content extraction and filtering.

        Args:
            url (str): URL to get website content from.

        Returns:
            str: Extracted content including title, main text, and tables.
        """

        try:
            downloaded = fetch_url(url)

            result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url)

            if result:
                result = json.loads(result)
                return f'=========== Website Title: {result["title"]} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{result["raw_text"]}\n\n=========== Website Content End ===========\n\n'
            else:
                return ""
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def get_website_content_from_urls(self, urls: list[str]) -> list[str]:
        results = []
        for url in urls:
            try:
                downloaded = fetch_url(url)

                result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url)

                if result:
                    result = json.loads(result)
                    results.append(f'=========== Website Title: {result["title"]} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{result["raw_text"]}\n\n=========== Website Content End ===========\n\n')
                else:
                    results.append("NO RESULT")
            except Exception as e:
                results.append(f"An error occurred: {str(e)}")
        return results