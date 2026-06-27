from __future__ import annotations

from pathlib import Path

import yaml

from agent_runtime.models import AgentRunResult


class ExecutionArtifactStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    def write_run(self, result: AgentRunResult) -> Path:
        run_dir = self.root_dir / _safe_path_id(result.trace_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_yaml(
            run_dir / "queries.yaml",
            {
                "root_query": result.root_query.model_dump(mode="json"),
                "subqueries": [query.model_dump(mode="json") for query in result.subqueries],
            },
        )
        self._write_yaml(
            run_dir / "responses.yaml",
            {
                "responses": [
                    response.model_dump(mode="json") for response in result.expert_responses
                ]
            },
        )
        self._write_yaml(
            run_dir / "contexts.yaml",
            {
                "contexts": [
                    context.model_dump(mode="json") for context in result.contexts
                ]
            },
        )
        self._write_yaml(
            run_dir / "trace.yaml",
            {
                "trace_id": result.trace_id,
                "spans": [span.model_dump(mode="json") for span in result.spans],
            },
        )
        self._write_query_instances(run_dir, result)
        self._write_markdown(run_dir / "final_response.md", "Final Response", result.final_response)
        return run_dir

    def _write_query_instances(self, run_dir: Path, result: AgentRunResult) -> None:
        instances_dir = run_dir / "instances"
        instances_dir.mkdir(exist_ok=True)

        contexts_by_query_id = {context.query_id: context for context in result.contexts}
        responses_by_query_id = {}
        for response in result.expert_responses:
            responses_by_query_id.setdefault(response.query_id, []).append(response)
        children_by_query_id = {}
        for subquery in result.subqueries:
            children_by_query_id.setdefault(subquery.sender_id, []).append(subquery.query_id)

        manifest_instances = []
        for query in [result.root_query, *result.subqueries]:
            safe_query_id = _safe_path_id(query.query_id)
            query_dir = instances_dir / safe_query_id
            query_dir.mkdir(exist_ok=True)

            context = contexts_by_query_id.get(query.query_id)
            responses = responses_by_query_id.get(query.query_id, [])
            child_query_ids = children_by_query_id.get(query.query_id, [])

            self._write_yaml(query_dir / "query.yaml", {"query": query.model_dump(mode="json")})
            if context is not None:
                self._write_yaml(
                    query_dir / "context.yaml",
                    {"context": context.model_dump(mode="json")},
                )
            if responses:
                self._write_yaml(
                    query_dir / "responses.yaml",
                    {
                        "responses": [
                            response.model_dump(mode="json") for response in responses
                        ]
                    },
                )
            self._write_markdown(
                query_dir / "summary.md",
                f"Query Instance {query.query_id}",
                self._instance_summary(
                    query_id=query.query_id,
                    query=query.query,
                    depth=query.depth,
                    sender_id=query.sender_id,
                    context_document_count=(
                        len(context.documents) if context is not None else 0
                    ),
                    expert_ids=[response.expert_id for response in responses],
                    child_query_ids=child_query_ids,
                ),
            )
            manifest_instances.append(
                {
                    "query_id": query.query_id,
                    "sender_id": query.sender_id,
                    "depth": query.depth,
                    "path": f"instances/{safe_query_id}",
                    "context_file": (
                        f"instances/{safe_query_id}/context.yaml"
                        if context is not None
                        else None
                    ),
                    "response_file": (
                        f"instances/{safe_query_id}/responses.yaml" if responses else None
                    ),
                    "child_query_ids": child_query_ids,
                }
            )

        self._write_yaml(
            run_dir / "manifest.yaml",
            {
                "trace_id": result.trace_id,
                "root_query_id": result.root_query.query_id,
                "instance_count": len(manifest_instances),
                "instances": manifest_instances,
            },
        )

    def _write_yaml(self, path: Path, data: dict) -> None:
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False)

    def _write_markdown(self, path: Path, title: str, text: str) -> None:
        path.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")

    def _instance_summary(
        self,
        *,
        query_id: str,
        query: str,
        depth: int,
        sender_id: str,
        context_document_count: int,
        expert_ids: list[str],
        child_query_ids: list[str],
    ) -> str:
        expert_text = ", ".join(expert_ids) if expert_ids else "none"
        child_text = ", ".join(child_query_ids) if child_query_ids else "none"
        return "\n".join(
            [
                f"- Query ID: `{query_id}`",
                f"- Sender ID: `{sender_id}`",
                f"- Depth: `{depth}`",
                f"- Context documents: `{context_document_count}`",
                f"- Experts: `{expert_text}`",
                f"- Child queries: `{child_text}`",
                "",
                "## Query",
                "",
                query,
            ]
        )


def _safe_path_id(value: str) -> str:
    return value.replace(":", "__").replace("/", "_")
