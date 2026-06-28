# Dullahan

Dullahan is an experimental app for hierarchical agent execution over graph
memory. It turns a domain corpus into graph-backed memory, partitions that graph
into expert-owned clusters, routes subqueries to specialized SLM experts, and
persists the full execution as inspectable YAML, Markdown, and an
`action_graph.json` trace.

Dullahan is the trace producer. Specter is the separate validation layer that can
consume Dullahan's `action_graph.json` and run prosecutor/defender/judge debate,
TransformerLens localization, and activation-hook steering.

## What Dullahan Does

```text
domain data -> graph memory -> expert clusters -> hierarchical execution -> action_graph.json
```

| Area | Purpose |
| --- | --- |
| `apps/agent-runtime` | Recursive execution loop, CLI, tracing, and artifacts. |
| `apps/cal` | Context Augmentation Layer for bounded context retrieval. |
| `apps/edl` | Expert Dispatch Layer for selecting and running expert instances. |
| `apps/graph-builder` | Imports graphified data, partitions graph clusters, and generates experts. |
| `apps/mcp-servers` | MCP tool surfaces for CAL and EDL. |
| `packages/kg` | Knowledge graph model, YAML graph storage, and K-partitioning. |
| `packages/world-state` | Local vector index over graph-backed Markdown documents. |
| `packages/shared` | Shared schemas, IDs, embeddings, and retrieval helpers. |

## Install

```bash
python -m pip install -e ".[dev]"
```

Installed commands:

```bash
dullahan-agent
dullahan-cal
dullahan-edl
dullahan-mcp-cal
dullahan-mcp-edl
dullahan-graphify
```

## Build Graph Memory

```bash
dullahan-graphify ./path/to/data --k 8
```

This writes:

```text
memory/graph/graph.yaml
memory/graph/clusters.yaml
memory/graph/experts.yaml
memory/documents/nodes/
memory/documents/clusters/
memory/world_state/indexes/local.json
```

If you already have a graphify output:

```bash
dullahan-graphify ./path/to/data \
  --from-graphify-json ./graphify-out/graph.json \
  --k 6
```

## Run An Agent Trace

```bash
dullahan-agent "Assess the deployment risk of this GPU inference architecture" \
  --max-depth 2 \
  --max-breadth 3 \
  --max-total-instances 12 \
  --persist-artifacts
```

Persisted runs are written under:

```text
memory/executions/<trace_id>/
  action_graph.json
  action_graph.mmd
  queries.yaml
  contexts.yaml
  responses.yaml
  trace.yaml
  manifest.yaml
  final_response.md
  instances/<query_id>/
```

The `action_graph.json` file is the handoff artifact for Specter.

## Validate With Specter

From the separate Specter repo:

```bash
specter-courtroom /path/to/Dullahan/memory/executions/<trace_id>/action_graph.json \
  --repo-root /path/to/Specter \
  --persist
```

Specter then owns:

```bash
specter-localize-feedback
specter-apply-feedback
specter-run-feedback-hooks
```

This keeps Dullahan focused on graph-backed agent execution and Specter focused
on validation, feedback localization, and inference steering.

## Run CAL And EDL As Services

```bash
docker compose up cal edl
```

Then run the agent over HTTP:

```bash
dullahan-agent "Evaluate whether this deployment path is robust" \
  --transport http \
  --cal-url http://127.0.0.1:8100 \
  --edl-url http://127.0.0.1:8200 \
  --max-depth 1 \
  --persist-artifacts
```

| Service | Port | Endpoints |
| --- | ---: | --- |
| CAL | `8100` | `/augment`, `/augment/batch` |
| EDL | `8200` | `/dispatch`, `/dispatch/batch` |

## MCP Tool Surface

```bash
dullahan-mcp-cal
dullahan-mcp-edl
```

| Tool | Purpose |
| --- | --- |
| `send_to_CAL` | Given a subquery and parent context, return a bounded context bundle. |
| `send_to_EDL` | Given a contextualized subquery, dispatch it to the best expert and return the response. |

Set `CAL_BASE_URL` and `EDL_BASE_URL` to point MCP servers at remote CAL/EDL
instances.

## Development

```bash
pytest
```
