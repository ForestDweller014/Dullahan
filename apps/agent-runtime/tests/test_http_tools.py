from __future__ import annotations

import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

import agent_runtime.tools.http as http_module
from agent_runtime.tools.http import HttpToolError
from agent_runtime.tools.http_cal import HttpCalTool
from agent_runtime.tools.http_edl import HttpEdlTool
from dullahan_shared.schemas.context import ContextBundle, ContextDocument, ContextSource
from dullahan_shared.schemas.query import QueryEnvelope


class FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHttpResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


# Verifies that HTTP CAL tool posts augment request.
def test_http_cal_tool_posts_augment_request(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeHttpResponse(
            {
                "subquery": "How should CAL retrieve context?",
                "context": {
                    "query_id": "query:root",
                    "documents": [
                        {
                            "id": "doc:cal",
                            "source": "graph_node",
                            "text": "CAL retrieves context.",
                            "score": 0.7,
                            "metadata": {},
                        }
                    ],
                    "token_budget": 2048,
                },
            }
        )

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)

    result = HttpCalTool("http://cal.local").send(
        subquery=QueryEnvelope(
            sender_id="query:root",
            query_id="query:child",
            query="How should CAL retrieve context?",
        ),
        parent_context=ContextBundle(query_id="query:root"),
    )

    assert result.context.documents[0].id == "doc:cal"
    assert requests[0]["url"] == "http://cal.local/augment"
    assert requests[0]["payload"]["sender_id"] == "query:root"
    assert requests[0]["payload"]["query_id"] == "query:child"


# Verifies that HTTP CAL tool posts batch augment request.
def test_http_cal_tool_posts_batch_augment_request(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeHttpResponse(
            {
                "responses": [
                    {
                        "subquery": "How should CAL retrieve context?",
                        "context": {
                            "query_id": "query:root",
                            "documents": [],
                            "token_budget": 2048,
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)

    result = HttpCalTool("http://cal.local").send_batch(
        [
            (
                QueryEnvelope(
                    sender_id="query:root",
                    query_id="query:child",
                    query="How should CAL retrieve context?",
                ),
                ContextBundle(query_id="query:root"),
            )
        ]
    )

    assert result[0].subquery == "How should CAL retrieve context?"
    assert requests[0]["url"] == "http://cal.local/augment/batch"
    assert requests[0]["payload"]["requests"][0]["sender_id"] == "query:root"
    assert requests[0]["payload"]["requests"][0]["query_id"] == "query:child"


# Verifies that HTTP EDL tool posts dispatch request.
def test_http_edl_tool_posts_dispatch_request(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeHttpResponse(
            {
                "response": {
                    "sender_id": "query:root",
                    "query_id": "query:child",
                    "subquery": "Which expert should EDL select?",
                    "expert_id": "expert:expert_dispatch",
                    "response": "EDL selected the dispatch expert.",
                    "confidence": 0.8,
                    "cited_context_document_ids": ["doc:edl"],
                }
            }
        )

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)

    result = HttpEdlTool("http://edl.local").send(
        subquery=QueryEnvelope(
            sender_id="query:root",
            query_id="query:child",
            query="Which expert should EDL select?",
        ),
        context=ContextBundle(
            query_id="query:child",
            documents=[
                ContextDocument(
                    id="doc:edl",
                    source=ContextSource.GRAPH_NODE,
                    text="EDL selects experts.",
                )
            ],
        ),
    )

    assert result.expert_id == "expert:expert_dispatch"
    assert requests[0]["url"] == "http://edl.local/dispatch"
    assert requests[0]["payload"]["context"]["documents"][0]["id"] == "doc:edl"


# Verifies that HTTP EDL tool posts batch dispatch request.
def test_http_edl_tool_posts_batch_dispatch_request(monkeypatch) -> None:
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(
            {
                "url": request.full_url,
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        return FakeHttpResponse(
            {
                "responses": [
                    {
                        "sender_id": "query:root",
                        "query_id": "query:child",
                        "subquery": "Which expert should EDL select?",
                        "expert_id": "expert:expert_dispatch",
                        "response": "EDL selected the dispatch expert.",
                        "confidence": 0.8,
                        "cited_context_document_ids": [],
                    }
                ]
            }
        )

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)

    result = HttpEdlTool("http://edl.local").send_batch(
        [
            (
                QueryEnvelope(
                    sender_id="query:root",
                    query_id="query:child",
                    query="Which expert should EDL select?",
                ),
                ContextBundle(query_id="query:child"),
            )
        ]
    )

    assert result[0].expert_id == "expert:expert_dispatch"
    assert requests[0]["url"] == "http://edl.local/dispatch/batch"
    assert requests[0]["payload"]["requests"][0]["query_id"] == "query:child"


# Verifies that an upstream HTTP error is converted into HttpToolError with its details.
def test_http_tool_raises_for_http_error(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(
            url=request.full_url,
            code=404,
            msg="not found",
            hdrs=None,
            fp=BytesIO(b"missing"),
        )

    monkeypatch.setattr(http_module, "urlopen", fake_urlopen)

    with pytest.raises(HttpToolError, match="HTTP 404"):
        HttpCalTool("http://cal.local").send(
            subquery=QueryEnvelope(
                sender_id="query:root",
                query_id="query:child",
                query="Missing route",
            ),
            parent_context=ContextBundle(query_id="query:root"),
        )
