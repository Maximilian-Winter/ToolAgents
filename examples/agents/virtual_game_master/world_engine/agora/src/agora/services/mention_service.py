"""Mention parsing and storage service for kb: and #N cross-references."""

import logging
import re

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from agora.db.models.mention import Mention

logger = logging.getLogger("agora")

# Patterns for mention extraction
KB_PATTERN = re.compile(r"kb:([^\s]+)")
ISSUE_PATTERN = re.compile(r"(?<![&#/\w])#(\d+)")

# Pattern to strip code blocks before parsing
CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```|`[^`]+`")


def extract_mentions(text: str) -> list[tuple[str, str]]:
    """Extract mentions from text content.

    Returns list of (mention_type, target) tuples:
    - ("kb", "path/to/doc.md")
    - ("kb", "path/to/doc.md#Section")
    - ("issue", "7")
    """
    # Strip code blocks to avoid false positives
    cleaned = CODE_FENCE_PATTERN.sub("", text)

    mentions: list[tuple[str, str]] = []

    for match in KB_PATTERN.finditer(cleaned):
        mentions.append(("kb", match.group(1)))

    for match in ISSUE_PATTERN.finditer(cleaned):
        mentions.append(("issue", match.group(1)))

    return mentions


async def store_mentions(
    project_id: int,
    source_type: str,
    source_id: int,
    text: str,
    db: AsyncSession,
) -> None:
    """Extract mentions from text and store them. Replaces any existing mentions for this source."""
    try:
        # Delete existing mentions for this source (handles edits)
        await db.execute(
            delete(Mention).where(
                Mention.project_id == project_id,
                Mention.source_type == source_type,
                Mention.source_id == source_id,
            )
        )

        # Extract and insert new mentions
        mentions = extract_mentions(text)
        for mention_type, target in mentions:
            mention = Mention(
                project_id=project_id,
                source_type=source_type,
                source_id=source_id,
                mention_type=mention_type,
                target_path=target if mention_type == "kb" else None,
                target_issue_number=int(target) if mention_type == "issue" else None,
            )
            db.add(mention)

        # Don't commit here — let the caller commit as part of their transaction
    except Exception:
        logger.exception("Failed to store mentions for %s/%s", source_type, source_id)


async def update_mention_paths(
    project_id: int,
    old_path: str,
    new_path: str,
    db: AsyncSession,
) -> None:
    """Update all mentions referencing old_path to point to new_path. Called on document move."""
    await db.execute(
        update(Mention)
        .where(
            Mention.project_id == project_id,
            Mention.mention_type == "kb",
            Mention.target_path == old_path,
        )
        .values(target_path=new_path)
    )
