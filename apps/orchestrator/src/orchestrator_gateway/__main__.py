"""Entrypoint for running the orchestrator via `python -m orchestrator_gateway`."""
from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "orchestrator_gateway.main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
