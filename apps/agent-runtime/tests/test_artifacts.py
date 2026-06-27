import json
from pathlib import Path

import yaml

from agent_runtime.agent import AgentRuntime
from agent_runtime.config import AgentRuntimeConfig
from agent_runtime.models import AgentRunRequest
from dullahan_shared.schemas.execution import ExecutionLimits


ROOT = Path(__file__).resolve().parents[3]


def test_agent_runtime_persists_execution_artifacts() -> None:
    runtime = AgentRuntime.local(
        AgentRuntimeConfig(
            repo_root=ROOT,
            limits=ExecutionLimits(max_depth=1, max_breadth_per_agent=1),
        )
    )

    result = runtime.run(
        AgentRunRequest(
            query="How should context move through YAML and Markdown?",
            persist_artifacts=True,
        )
    )

    assert result.artifact_dir is not None
    artifact_dir = Path(result.artifact_dir)
    assert (artifact_dir / "queries.yaml").exists()
    assert (artifact_dir / "contexts.yaml").exists()
    assert (artifact_dir / "responses.yaml").exists()
    assert (artifact_dir / "trace.yaml").exists()
    assert (artifact_dir / "manifest.yaml").exists()
    assert (artifact_dir / "action_graph.json").exists()
    assert (artifact_dir / "action_graph.mmd").exists()
    assert (artifact_dir / "final_response.md").exists()

    trace = yaml.safe_load((artifact_dir / "trace.yaml").read_text(encoding="utf-8"))
    queries = yaml.safe_load((artifact_dir / "queries.yaml").read_text(encoding="utf-8"))
    contexts = yaml.safe_load((artifact_dir / "contexts.yaml").read_text(encoding="utf-8"))
    manifest = yaml.safe_load((artifact_dir / "manifest.yaml").read_text(encoding="utf-8"))
    action_graph = json.loads((artifact_dir / "action_graph.json").read_text(encoding="utf-8"))

    assert trace["trace_id"] == result.trace_id
    assert queries["root_query"]["query"] == result.root_query.query
    assert contexts["contexts"]
    assert contexts["contexts"][0]["query_id"]
    assert manifest["root_query_id"] == result.root_query.query_id
    assert manifest["instance_count"] == 2
    assert len(manifest["instances"]) == 2
    assert action_graph["schema"] == "dullahan.action_graph.v1"
    assert action_graph["root_query_id"] == result.root_query.query_id
    assert len(action_graph["nodes"]) == 2
    assert len(action_graph["edges"]) == 1
    assert action_graph["edges"][0]["source"] == result.root_query.query_id
    assert action_graph["edges"][0]["target"] == result.subqueries[0].query_id
    assert action_graph["edges"][0]["query"] == result.subqueries[0].query

    subquery_node = next(
        node for node in action_graph["nodes"] if node["id"] == result.subqueries[0].query_id
    )
    assert subquery_node["query"]["query"] == result.subqueries[0].query
    assert subquery_node["context"]["query_id"] == result.subqueries[0].query_id
    assert subquery_node["response"]["query_id"] == result.subqueries[0].query_id
    assert subquery_node["responses"][0]["query_id"] == result.subqueries[0].query_id
    assert "flowchart TD" in (artifact_dir / "action_graph.mmd").read_text(encoding="utf-8")

    subquery_instance = next(
        instance
        for instance in manifest["instances"]
        if instance["query_id"] == result.subqueries[0].query_id
    )
    subquery_dir = artifact_dir / subquery_instance["path"]

    assert (subquery_dir / "query.yaml").exists()
    assert (subquery_dir / "context.yaml").exists()
    assert (subquery_dir / "responses.yaml").exists()
    assert (subquery_dir / "summary.md").exists()

    subquery_context = yaml.safe_load((subquery_dir / "context.yaml").read_text(encoding="utf-8"))
    subquery_response = yaml.safe_load(
        (subquery_dir / "responses.yaml").read_text(encoding="utf-8")
    )

    assert subquery_context["context"]["query_id"] == result.subqueries[0].query_id
    assert subquery_response["responses"][0]["query_id"] == result.subqueries[0].query_id
    assert "Query Instance" in (subquery_dir / "summary.md").read_text(encoding="utf-8")


def test_agent_runtime_does_not_persist_artifacts_by_default() -> None:
    runtime = AgentRuntime.local(
        AgentRuntimeConfig(
            repo_root=ROOT,
            limits=ExecutionLimits(max_depth=0, max_breadth_per_agent=1),
        )
    )

    result = runtime.run(AgentRunRequest(query="No artifact run"))

    assert result.artifact_dir is None
