from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from dullahan_inference.config import InferenceConfig
from dullahan_inference.ollama import OllamaClient, OllamaError, OllamaProcess
from dullahan_inference.plan import ResolvedInferencePlan


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=512, ge=1)
    temperature: float | None = Field(default=None, ge=0)


def create_ollama_app(config: InferenceConfig, plan: ResolvedInferencePlan) -> FastAPI:
    client = OllamaClient(config, plan)
    process = OllamaProcess(config, plan)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        process.start()
        try:
            yield
        finally:
            process.stop()

    app = FastAPI(title="Dullahan Local Inference", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "plan": plan.model_dump(mode="json")}

    @app.get("/v1/models")
    def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": config.ollama.model, "object": "model", "owned_by": "ollama"}],
        }

    @app.post("/v1/completions")
    def completions(request: CompletionRequest) -> dict:
        try:
            result = client.complete(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )
        except OllamaError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model or config.ollama.model,
            "choices": [{"index": 0, "text": result.text, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    return app
