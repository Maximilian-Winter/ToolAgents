import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agora.config import settings
from agora.db.engine import engine
from agora.db.base import Base

logger = logging.getLogger("agora")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables (will be replaced by Alembic later, but useful for dev)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from agora.seeds.default_templates import seed_default_templates
    from agora.db.engine import async_session

    async with async_session() as session:
        await seed_default_templates(session)

    from agora.services.kb_service import create_fts_table

    async with async_session() as session:
        await create_fts_table(session)

    if "*" in settings.cors_origins:
        logger.warning(
            "CORS is set to allow ALL origins (['*']). "
            "Set AGORA_CORS_ORIGINS to restrict access in production."
        )

    yield
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agora",
        description="Multi-agent team coordination platform with chat, issue tracking, and project management.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from agora.api.routes.projects import router as projects_router
    from agora.api.routes.agents import agents_router, personas_router
    from agora.api.routes.teams import router as teams_router
    from agora.api.routes.sessions import router as sessions_router
    from agora.api.routes.tasks import router as tasks_router
    from agora.api.routes.chat import router as chat_router
    from agora.api.routes.presence import router as presence_router
    from agora.api.routes.utilities import router as utilities_router
    from agora.api.routes.import_export import router as import_export_router
    from agora.api.routes.project_agents import router as project_agents_router
    from agora.api.routes.terminals import router as terminals_router
    from agora.api.routes.custom_fields import definitions_router, agent_fields_router, project_fields_router
    from agora.api.routes.templates import global_templates_router, project_templates_router
    from agora.api.routes.kb import router as kb_router
    from agora.api.routes.mentions import router as mentions_router

    app.include_router(projects_router)
    app.include_router(agents_router)
    app.include_router(personas_router)
    app.include_router(teams_router)
    app.include_router(sessions_router)
    app.include_router(tasks_router)
    app.include_router(chat_router)
    app.include_router(presence_router)
    app.include_router(utilities_router)
    app.include_router(import_export_router)
    app.include_router(project_agents_router)
    app.include_router(terminals_router)
    app.include_router(definitions_router)
    app.include_router(agent_fields_router)
    app.include_router(project_fields_router)
    app.include_router(global_templates_router)
    app.include_router(project_templates_router)
    app.include_router(kb_router)
    app.include_router(mentions_router)

    # Serve React frontend build (production)
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    frontend_dist = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="static")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str):
            """Serve React SPA — all non-API routes go to index.html."""
            file_path = frontend_dist / full_path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(
                str(frontend_dist / "index.html"),
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

    return app


app = create_app()
