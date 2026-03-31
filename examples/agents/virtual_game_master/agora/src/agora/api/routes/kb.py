"""Knowledge base document CRUD, search, and tree endpoints."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import JSONResponse

from agora.db.engine import get_db
from agora.db.models.kb_document import KBDocument
from agora.schemas.kb import (
    KBDocumentCreate,
    KBDocumentMove,
    KBDocumentOut,
    KBDocumentSummary,
    KBSearchResult,
    KBTreeNode,
)
from agora.api.deps import require_project
from agora.services.kb_service import (
    build_tree,
    extract_section,
    fts_delete,
    fts_insert,
    fts_update,
    tags_contain,
)
from agora.services.mention_service import update_mention_paths

router = APIRouter(prefix="/api/projects/{project_slug}/kb", tags=["Knowledge Base"])


# ── Fixed routes MUST come before {path:path} catch-all ──────────


@router.get("/search", response_model=list[KBSearchResult])
async def search_documents(
    project_slug: str,
    q: str = Query(..., min_length=1),
    tag: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        text("""
            SELECT kd.path, kd.title, kd.tags,
                   snippet(kb_document_fts, 1, '<mark>', '</mark>', '...', 32) as snippet,
                   bm25(kb_document_fts) as rank
            FROM kb_document_fts
            JOIN kb_documents kd ON kd.id = kb_document_fts.rowid
            WHERE kb_document_fts MATCH :query AND kd.project_id = :project_id
            ORDER BY rank
            LIMIT :limit
        """),
        {"query": q, "project_id": project.id, "limit": limit},
    )
    rows = result.all()
    results = []
    for row in rows:
        if tag and not tags_contain(row.tags, tag):
            continue
        results.append(KBSearchResult(path=row.path, title=row.title, snippet=row.snippet, rank=row.rank))
    return results


@router.get("/tree", response_model=list[KBTreeNode])
async def get_document_tree(
    project_slug: str,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(KBDocument.path, KBDocument.title)
        .where(KBDocument.project_id == project.id)
        .order_by(KBDocument.path)
    )
    docs = [{"path": row.path, "title": row.title} for row in result.all()]
    return build_tree(docs)


# ── CRUD routes ──────────────────────────────────────────────────


@router.post("", status_code=201)
async def create_or_replace_document(
    project_slug: str,
    body: KBDocumentCreate,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    title = body.title or body.path.rsplit("/", 1)[-1]

    # Check if document already exists (upsert)
    result = await db.execute(
        select(KBDocument).where(
            KBDocument.project_id == project.id,
            KBDocument.path == body.path,
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        old_title, old_content, old_tags = existing.title, existing.content, existing.tags
        existing.title = title
        existing.tags = body.tags
        existing.content = body.content
        existing.updated_by = body.author
        await db.flush()
        await fts_update(db, existing.id, old_title, old_content, old_tags or "", title, body.content, body.tags or "")
        await db.commit()
        await db.refresh(existing)
        return JSONResponse(
            status_code=200,
            content=KBDocumentOut.model_validate(existing).model_dump(mode="json"),
        )
    else:
        doc = KBDocument(
            project_id=project.id,
            path=body.path,
            title=title,
            tags=body.tags,
            content=body.content,
            created_by=body.author,
            updated_by=body.author,
        )
        db.add(doc)
        await db.flush()
        await fts_insert(db, doc.id, title, body.content, body.tags or "")
        await db.commit()
        await db.refresh(doc)
        return JSONResponse(
            status_code=201,
            content=KBDocumentOut.model_validate(doc).model_dump(mode="json"),
        )


@router.get("", response_model=list[KBDocumentSummary])
async def list_documents(
    project_slug: str,
    prefix: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    stmt = select(KBDocument).where(KBDocument.project_id == project.id)
    if prefix:
        stmt = stmt.where(KBDocument.path.startswith(prefix))
    stmt = stmt.order_by(KBDocument.path)
    result = await db.execute(stmt)
    docs = result.scalars().all()
    if tag:
        docs = [d for d in docs if tags_contain(d.tags, tag)]
    return docs


@router.get("/{path:path}", response_model=KBDocumentOut)
async def read_document(
    project_slug: str,
    path: str,
    section: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(KBDocument).where(
            KBDocument.project_id == project.id,
            KBDocument.path == path,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, f"Document '{path}' not found")

    if section:
        section_content = extract_section(doc.content, section)
        if section_content is not None:
            # Return doc with only the section content
            out = KBDocumentOut.model_validate(doc)
            out.content = section_content
            return out
        # Section not found — return full document (spec: "return full document with empty section_match")

    return doc


@router.delete("/{path:path}", status_code=204)
async def delete_document(
    project_slug: str,
    path: str,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(KBDocument).where(
            KBDocument.project_id == project.id,
            KBDocument.path == path,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, f"Document '{path}' not found")
    await fts_delete(db, doc.id, doc.title, doc.content, doc.tags or "")
    await db.delete(doc)
    await db.commit()


@router.patch("/{path:path}/move", response_model=KBDocumentOut)
async def move_document(
    project_slug: str,
    path: str,
    body: KBDocumentMove,
    db: AsyncSession = Depends(get_db),
):
    project = await require_project(project_slug, db)
    result = await db.execute(
        select(KBDocument).where(
            KBDocument.project_id == project.id,
            KBDocument.path == path,
        )
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, f"Document '{path}' not found")

    # Check target doesn't already exist
    conflict = await db.execute(
        select(KBDocument).where(
            KBDocument.project_id == project.id,
            KBDocument.path == body.new_path,
        )
    )
    if conflict.scalar_one_or_none():
        raise HTTPException(409, f"Document '{body.new_path}' already exists")

    doc.path = body.new_path
    await update_mention_paths(project.id, path, body.new_path, db)
    await db.commit()
    await db.refresh(doc)
    return doc
