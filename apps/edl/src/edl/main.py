from __future__ import annotations

from fastapi import FastAPI

from edl.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="Dullahan EDL")
    app.include_router(router)
    return app


app = create_app()
