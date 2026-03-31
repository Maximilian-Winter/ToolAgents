"""
Agora KB Backend — StorageBackend adapter for the Agora Knowledge Base.

Connects NavigableMemory to an Agora project's KB via its REST API.

Usage:
    from dao_framework.backends.agora import AgoraBackend
    from dao_framework.navigable_memory import NavigableMemory

    backend = AgoraBackend(project_slug="my-project")
    memory = NavigableMemory(backend)

Requires httpx: pip install httpx
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from navigable_memory import Document

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    httpx = None


class AgoraBackend:
    """StorageBackend implementation for the Agora Knowledge Base REST API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8321",
        project_slug: str = "default",
        author: str = "navigable-memory",
        timeout: float = 30.0,
    ):
        if httpx is None:
            raise ImportError("AgoraBackend requires httpx: pip install httpx")

        self.base_url = base_url.rstrip("/")
        self.project_slug = project_slug
        self.author = author
        self._client = httpx.Client(timeout=timeout)

    @property
    def kb_url(self) -> str:
        return f"{self.base_url}/api/projects/{self.project_slug}/kb"

    def read(self, path: str) -> Optional[Document]:
        try:
            resp = self._client.get(f"{self.kb_url}/{path}")
            if resp.status_code == 200:
                d = resp.json()
                return Document(
                    path=d.get("path", path),
                    title=d.get("title", "Untitled"),
                    content=d.get("content", ""),
                    tags=_parse_tags(d.get("tags", "")),
                    updated_at=d.get("updated_at"),
                )
            return None
        except httpx.HTTPError as e:
            logger.error("Agora read '%s': %s", path, e)
            return None

    def write(self, path: str, title: str, content: str,
              tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None) -> bool:
        payload = {
            "path": path,
            "title": title,
            "content": content,
            "tags": ",".join(tags) if tags else "",
            "author": self.author,
        }
        try:
            resp = self._client.post(self.kb_url, json=payload)
            return resp.status_code in (200, 201)
        except httpx.HTTPError as e:
            logger.error("Agora write '%s': %s", path, e)
            return False

    def list(self, prefix: str = "") -> List[Document]:
        params = {"prefix": prefix} if prefix else {}
        try:
            resp = self._client.get(self.kb_url, params=params)
            if resp.status_code == 200:
                return [
                    Document(
                        path=d.get("path", ""),
                        title=d.get("title", ""),
                        content=d.get("content", ""),
                        tags=_parse_tags(d.get("tags", "")),
                    )
                    for d in resp.json()
                ]
            return []
        except httpx.HTTPError as e:
            logger.error("Agora list: %s", e)
            return []

    def search(self, query: str) -> List[Document]:
        try:
            resp = self._client.get(f"{self.kb_url}/search", params={"q": query})
            if resp.status_code == 200:
                return [
                    Document(
                        path=d.get("path", ""),
                        title=d.get("title", ""),
                        content=d.get("snippet", d.get("content", "")),
                        tags=_parse_tags(d.get("tags", "")),
                    )
                    for d in resp.json()
                ]
            return []
        except httpx.HTTPError as e:
            logger.error("Agora search: %s", e)
            return []

    def delete(self, path: str) -> bool:
        try:
            resp = self._client.delete(f"{self.kb_url}/{path}")
            return resp.status_code in (200, 204)
        except httpx.HTTPError as e:
            logger.error("Agora delete '%s': %s", path, e)
            return False

    def close(self):
        self._client.close()


def _parse_tags(tags_str: str) -> List[str]:
    if not tags_str:
        return []
    return [t.strip() for t in tags_str.split(",") if t.strip()]
