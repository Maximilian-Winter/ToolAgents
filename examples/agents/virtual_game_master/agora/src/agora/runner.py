import uvicorn

from agora.config import settings


def main():
    uvicorn.run(
        "agora.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
