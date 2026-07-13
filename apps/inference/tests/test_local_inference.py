from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import yaml
from agent_runtime.agent import AgentRuntime
from agent_runtime.config import AgentRuntimeConfig
from agent_runtime.models import AgentRunRequest
from agent_runtime.planning.provider import OpenAICompatiblePlannerProvider, PlannerRequest
from cal.config import CalConfig
from dullahan_inference.config import InferenceConfig
from dullahan_inference.device import detect_device
from dullahan_inference.plan import resolve_inference_plan
from dullahan_shared.embeddings import OpenAICompatibleEmbeddingModel, cosine_similarity
from dullahan_shared.schemas.context import ContextBundle, ContextDocument, ContextSource
from dullahan_shared.schemas.execution import ExecutionLimits
from dullahan_shared.schemas.expert import ExpertProfile
from dullahan_shared.schemas.query import QueryEnvelope
from dullahan_shared.tokenization import InferenceTokenCounter
from edl.api.schemas import DispatchRequest
from edl.dispatch.attention_router import ExpertAttentionScore, ExpertRoute
from edl.execution.expert_runner import ExpertRunner
from edl.execution.model_provider import OpenAICompatibleHttpProvider
from edl.execution.prompt import ExpertPromptBuilder


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.fixture(scope="module")
def cpu_inference_base_url(tmp_path_factory) -> str:
    if os.getenv("DULLAHAN_RUN_LOCAL_INFERENCE") != "1":
        pytest.skip("set DULLAHAN_RUN_LOCAL_INFERENCE=1 to run a local model")

    port = _free_port()
    config = InferenceConfig(
        provider="ollama",
        device="cpu",
        ollama={"model": os.getenv("DULLAHAN_TEST_MODEL", "qwen3:8b")},
        embeddings={
            "model": os.getenv("DULLAHAN_TEST_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
            "dimensions": 1024,
        },
        server={"host": "127.0.0.1", "advertised_host": "127.0.0.1", "port": port},
    )
    plan = resolve_inference_plan(config, inventory=detect_device(config.device))
    assert plan.memory_fit, plan.notes
    config_path = tmp_path_factory.mktemp("cpu-inference") / "inference.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source_root), environment.get("PYTHONPATH")) if value
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "dullahan_inference.cli",
            "serve",
            "--config",
            str(config_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"inference server exited early: {process.stderr.read()}")
        try:
            with urlopen(health_url, timeout=1):
                break
        except URLError:
            time.sleep(0.1)
    else:
        process.terminate()
        pytest.fail("inference server did not become healthy")

    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        process.terminate()
        process.wait(timeout=10)


# Verifies that local Ollama generates a basic response.
@pytest.mark.local_inference
def test_local_ollama_generates_a_basic_response(cpu_inference_base_url) -> None:
    request = Request(
        f"{cpu_inference_base_url}/completions",
        data=json.dumps(
            {
                "model": "local-planner",
                "prompt": "Reply with exactly: inference-ok",
                "max_tokens": 32,
                "temperature": 0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        payload = json.loads(response.read())

    assert payload["choices"][0]["text"]
    assert payload["usage"]["completion_tokens"] > 0


# Verifies real CPU semantic embeddings separate related and unrelated text.
@pytest.mark.local_inference
def test_real_embedding_model_produces_semantic_similarity(cpu_inference_base_url) -> None:
    model = OpenAICompatibleEmbeddingModel(
        base_url=cpu_inference_base_url,
        model=os.getenv("DULLAHAN_TEST_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        dimensions=1024,
        timeout_seconds=180,
    )

    context, related, unrelated = model.embed_many(
        [
            "CAL retrieves context for an agent query.",
            "The context retrieval layer finds relevant documents.",
            "A banana bread recipe needs ripe fruit and flour.",
        ]
    )

    assert cosine_similarity(context, related) > cosine_similarity(context, unrelated)
    assert cosine_similarity(context, related) > 0.5


# Verifies the tokenizer endpoint reports the generation model's real native prompt usage.
@pytest.mark.local_inference
def test_real_token_counter_matches_ollama_prompt_usage(cpu_inference_base_url) -> None:
    prompt = "Token accounting must use the serving model tokenizer."
    counter = InferenceTokenCounter(
        base_url=cpu_inference_base_url,
        model="Qwen/Qwen3-8B",
        timeout_seconds=180,
    )
    request = Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(
            {
                "model": os.getenv("DULLAHAN_TEST_MODEL", "qwen3:8b"),
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "options": {"num_predict": 1, "num_gpu": 0},
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        payload = json.loads(response.read())

    assert counter.count(prompt) == payload["prompt_eval_count"]


# Verifies the full local runtime uses real planning, embeddings, tokenization, and experts.
@pytest.mark.local_inference
def test_real_runtime_uses_semantic_context_and_native_token_budget(
    cpu_inference_base_url,
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("EDL_MODEL_BASE_URL", cpu_inference_base_url)
    monkeypatch.setenv("EDL_MODEL_TIMEOUT_SECONDS", "180")
    embedding_model = OpenAICompatibleEmbeddingModel(
        base_url=cpu_inference_base_url,
        model=os.getenv("DULLAHAN_TEST_EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
        dimensions=1024,
        timeout_seconds=180,
    )
    token_counter = InferenceTokenCounter(
        base_url=cpu_inference_base_url,
        model="Qwen/Qwen3-8B",
        timeout_seconds=180,
    )
    repo_root = Path(__file__).resolve().parents[3]
    runtime = AgentRuntime.local(
        AgentRuntimeConfig(
            repo_root=repo_root,
            limits=ExecutionLimits(
                max_depth=1,
                max_breadth_per_agent=1,
                max_total_agent_instances=2,
                timeout_seconds_per_instance=180,
            ),
            planner_base_url=cpu_inference_base_url,
            planner_timeout_seconds=180,
            synthesis_base_url=cpu_inference_base_url,
            synthesis_timeout_seconds=180,
            synthesis_max_tokens=192,
        ),
        embedding_model=embedding_model,
        token_counter=token_counter,
        cal_config=CalConfig(
            repo_root=repo_root,
            world_state_index_path=tmp_path / "semantic-world-state.json",
            world_top_k=2,
            parent_top_k=1,
            token_budget=256,
            inference_base_url=cpu_inference_base_url,
            inference_timeout_seconds=180,
        ),
    )

    result = runtime.run(
        AgentRunRequest(
            query="Explain how CAL retrieves relevant context before EDL selects an expert."
        )
    )

    assert result.expert_responses
    assert result.contexts
    assert result.contexts[0].documents
    assert result.contexts[0].metadata["tokenizer_model"] == "Qwen/Qwen3-8B"
    assert result.contexts[0].metadata["selected_token_count"] <= 256
    assert result.final_response
    assert not result.final_response.startswith("Root query:")
    normalized_final = result.final_response.lower()
    assert "context" in normalized_final
    assert "expert" in normalized_final or "route" in normalized_final
    synthesis_spans = [span for span in result.spans if span.name == "agent.synthesis"]
    assert len(synthesis_spans) == 1
    assert synthesis_spans[0].attributes["provider"] == "openai-compatible-synthesis"
    assert synthesis_spans[0].attributes["completion_token_count"] > 0


# Verifies that real planner generates query specific subqueries.
@pytest.mark.local_inference
def test_real_planner_generates_query_specific_subqueries(cpu_inference_base_url) -> None:
    result = OpenAICompatiblePlannerProvider(
        base_url=cpu_inference_base_url,
        model="local-planner",
        timeout_seconds=180,
    ).plan(
        PlannerRequest(
            parent_query=QueryEnvelope(
                sender_id="user",
                query_id="query:root",
                query=(
                    "Plan a safe migration from local file storage to PostgreSQL "
                    "with checksum verification and rollback."
                ),
            ),
            max_breadth=3,
        )
    )

    generic_placeholders = {
        "What context should CAL retrieve?",
        "Which expert should EDL select?",
        "What knowledge graph concepts are relevant?",
    }
    joined = " ".join(result.subqueries).lower()

    assert result.provider == "openai-compatible-planner"
    assert 1 <= len(result.subqueries) <= 3
    assert len(set(result.subqueries)) == len(result.subqueries)
    assert not generic_placeholders.intersection(result.subqueries)
    assert {"migration", "postgresql", "checksum", "rollback", "storage"} & set(
        joined.replace("?", "").replace(".", "").split()
    )


# Verifies that real CPU inference answers with the ordered facts supplied in expert context.
@pytest.mark.local_inference
def test_real_expert_produces_grounded_response(cpu_inference_base_url) -> None:
    expert = ExpertProfile(
        id="expert:migration",
        cluster_id="cluster:migration",
        role_context="You are a database migration specialist.",
        model="local-slm-test",
    )
    route = ExpertRoute(
        expert=expert,
        score=1.0,
        probability=1.0,
        distribution=[ExpertAttentionScore(expert_id=expert.id, raw_score=1.0, probability=1.0)],
    )
    request = DispatchRequest(
        sender_id="query:root",
        query_id="query:migration",
        subquery="What exact migration sequence does the supplied runbook require?",
        context=ContextBundle(
            query_id="query:migration",
            documents=[
                ContextDocument(
                    id="doc:runbook",
                    source=ContextSource.WORLD_STATE,
                    text=(
                        "The required sequence is: enable dual-write, backfill PostgreSQL, "
                        "verify checksums, switch reads, then retain the local files for rollback."
                    ),
                )
            ],
        ),
    )

    response = ExpertRunner(
        prompt_builder=ExpertPromptBuilder(),
        model_provider=OpenAICompatibleHttpProvider(
            base_url=cpu_inference_base_url,
            timeout_seconds=180,
        ),
        max_tokens=128,
    ).run(request, expert, route)
    normalized = response.response.lower().replace("-", "")

    assert response.routing_metadata["model_provider"] == "openai-compatible-http"
    assert "model local-slm" not in normalized
    assert (
        sum(
            term in normalized
            for term in ("dualwrite", "backfill", "checksum", "switch reads", "rollback")
        )
        >= 3
    )
