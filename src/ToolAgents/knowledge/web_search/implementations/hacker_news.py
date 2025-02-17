import httpx

from ToolAgents.knowledge.web_search.web_search import WebSearchProvider


class HackernewsWebSearchProvider(WebSearchProvider):

    def search_web(self, search_query: str, num_results: int):
        """
        Search the web, like a google search query. Returns a list of result URLs
        Args:
            search_query (str): Search query.
            num_results (int): Number of results to return.
        Returns:
            List[str]: List of URLs with search results.
        """
        # Fetch top story IDs
        response = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json")
        response.raise_for_status()
        story_ids = response.json()

        # Fetch story details
        stories = []
        for story_id in story_ids[:num_results]:
            story_response = httpx.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
            story_response.raise_for_status()
            story = story_response.json()
            story["username"] = story["by"]
            stories.append(story["url"])

        return stories
