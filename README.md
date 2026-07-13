# Dullahan

Dullahan is an experimental app that orchestrates an agent swarm that performs
hierarchical task execution by dispatching specialized SLM-based agents to a
clustered, dynamically growing and morphing context graph to solve complex tasks
with reliable context control and modular expert delegation.

It is built for tasks where the hard part is not a single model call, but keeping
many smaller reasoning steps grounded in the right local context. Dullahan stores
knowledge as graph memory, partitions that memory into expert-owned clusters,
uses a Context Augmentation Layer (CAL) to build bounded context for each
subquery, and uses an Expert Dispatch Layer (EDL) to route work to the most
relevant specialist agent.

The project is intentionally modular: CAL, EDL, the agent runtime, the graph
builder, MCP tools, and filesystem memory all communicate through typed
contracts from `packages/shared`.

## Why This Exists

Large tasks often fail because context is too broad, too stale, or too entangled.
Dullahan explores a different pattern:

1. Break a root task into bounded subqueries.
2. Retrieve only the context each subquery needs from parent reasoning and
   long-term graph memory.
3. Route the contextualized subquery to a specialized expert.
4. Let experts recursively ask their own subquestions when needed.
5. Persist the execution as inspectable YAML and Markdown.

The aim is reliable context control: each expert receives a focused slice of the
world instead of a giant prompt, and each step leaves behind artifacts that can be
audited, replayed, or used for later distillation.

## What Is In The Repo

| Area | Purpose |
| --- | --- |
| `apps/agent-runtime` | Recursive hierarchical execution loop, CLI, local and HTTP CAL/EDL tools, tracing, artifacts. |
| `apps/cal` | Context Augmentation Layer. Merges parent context with WorldStateDB retrieval and enforces token budgets. |
| `apps/edl` | Expert Dispatch Layer. Routes subqueries to experts with embedding attention and runs expert instances. |
| `apps/graph-builder` | Builds K-sized graph clusters and can derive `experts.yaml` from those clusters. |
| `apps/inference` | Resolves CPU/CUDA policy and serves Qwen through vLLM or an Ollama compatibility proxy. |
| `apps/model-server` | Builds and runs the two persistent vLLM model-manager containers: Linux CPU and NVIDIA CUDA. |
| `apps/mcp-servers` | Dependency-light stdio JSON-RPC MCP surfaces for `send_to_CAL` and `send_to_EDL`. |
| `packages/kg` | Knowledge graph model, YAML graph storage, and K-partitioning. |
| `packages/world-state` | Local persistent vector index over graph-backed Markdown documents. |
| `packages/shared` | Pydantic schemas, IDs, inference-backed embedding/tokenization clients, and retrieval helpers. |
| `memory/` | Seed graph, cluster docs, expert registry, execution artifacts, and local indexes. |
| `configs/` | Runtime recursion, retrieval, routing, and local configuration. |

## Tech Stack

| Category | Tools |
| --- | --- |
| Runtime and services | Python, FastAPI, Uvicorn, REST APIs |
| Agent integration | MCP stdio tools, OpenAI-compatible planner, expert, vLLM, and Ollama endpoints |
| Context and retrieval | Graphify, graph-backed RAG, local WorldStateDB vector index, PostgreSQL + pgvector, Qwen3 semantic embeddings |
| Data and artifacts | JSON, YAML, Markdown, Mermaid |
| Validation and schemas | Pydantic, pytest |
| Local orchestration | Docker Compose, CLI entrypoints |

## Architecture

```mermaid
flowchart TD
    User["Root query"] --> Runtime["Agent Runtime"]
    Runtime --> Planner["Planner / subquery generator"]
    Planner --> Subqueries["Sibling subqueries"]
    Subqueries --> CAL["CAL: context augmentation"]
    CAL --> ParentContext["Parent context retrieval"]
    CAL --> WorldStateDB["WorldStateDB over graph docs / pgvector"]
    CAL --> ContextBundle["Bounded ContextBundle"]
    ContextBundle --> EDL["EDL: expert dispatch"]
    EDL --> Router["Attention router"]
    Router --> Experts["Cluster-specialized SLM experts"]
    Experts --> Inference["Local inference: Qwen/vLLM or Ollama"]
    Inference --> Runtime
    Runtime --> Artifacts["YAML + Markdown execution artifacts"]
```

The key runtime contracts are:

| Contract | Meaning |
| --- | --- |
| `QueryEnvelope` | The root query or generated subquery, including sender, query ID, depth, and metadata. |
| `ContextBundle` | Documents retrieved for a specific query, with optional token budget. |
| `ExpertProfile` | A specialist agent bound to a graph cluster and role context document. |
| `ExpertResponse` | The answer returned by an expert for one contextualized subquery. |
| `ExecutionSpan` | Trace metadata for runtime, CAL, EDL, timeout, and subquery events. |

## Ideal Use Cases

Dullahan is a good fit when you want many small, specialized agents to work over
a structured body of knowledge:

| Use case | Why Dullahan fits |
| --- | --- |
| Large codebase analysis | Files, classes, services, and docs can become graph nodes; experts specialize by cluster. |
| Infrastructure reasoning | Cloud resources, IAM boundaries, deployment workflows, and observability docs can be separated into expert domains. |
| Research synthesis | Papers, concepts, figures, datasets, and claims can be represented as graph memory with specialist reviewers. |
| Enterprise knowledge assistants | Teams, systems, SOPs, incidents, and domain docs can be routed to scoped expert agents. |
| Multi-step planning | Recursive subqueries make task decomposition explicit and inspectable. |
| Training data generation | YAML/Markdown traces provide structured examples for later distillation or evaluation. |

Dullahan is less ideal for one-shot chat, simple RAG over a small folder, or
tasks where a single general-purpose model call is already sufficient.

## Quickstart

### 1. Install

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

This installs the CLI entrypoints:

```bash
dullahan-agent
dullahan-cal
dullahan-edl
dullahan-inference
dullahan-benchmark-gguf
dullahan-mcp-cal
dullahan-mcp-edl
dullahan-graphify
```

The `graphify` command used by `dullahan-graphify` is provided by the
`graphifyy` package from `safishamsi/graphify`.

### 2. Graphify A Data Collection

Point the CLI at a file or directory collection to construct graph memory from
that world state:

```bash
dullahan-graphify ./research/market-notes --k 8
```

`dullahan-graphify` invokes the real
[`safishamsi/graphify`](https://github.com/safishamsi/graphify) CLI, imports its
`graphify-out/graph.json`, converts that graph into Dullahan's YAML graph
memory, partitions it into K-sized clusters, and regenerates the expert registry
from those clusters.

The generated memory lands in:

```text
memory/graph/graph.yaml
memory/graph/clusters.yaml
memory/graph/experts.yaml
memory/documents/nodes/
memory/documents/clusters/
memory/world_state/indexes/local.json
```

You can also pull source context from PostgreSQL before graphification. The SQL
query should return `id`, `title`, `content`, and optionally `metadata` columns:

```bash
dullahan-graphify \
  --postgres-dsn postgresql://dullahan:dullahan@127.0.0.1:5432/dullahan \
  --postgres-query "select id, title, content, metadata from research_notes order by id" \
  --k 8
```

That exports the pulled rows as Markdown under `memory/postgres_context/`, runs
Graphify over that collection, converts the result into Dullahan graph memory,
partitions the graph into K-bounded clusters, regenerates experts, and rebuilds
WorldStateDB retrieval.

Useful graphification options:

```bash
dullahan-graphify ./research/market-notes \
  --k 6 \
  --graphify-command graphify \
  --graphify-output-dir ./graphify-out
```

If you already have a `graphify` output file, import it directly:

```bash
dullahan-graphify ./research/market-notes \
  --from-graphify-json ./graphify-out/graph.json \
  --k 6
```

### 3. Start Local Inference

Inspect the automatically resolved device, quantization, model, offload, and
launch command without loading model weights:

```bash
dullahan-inference plan
```

The default [`configs/inference.yaml`](configs/inference.yaml) selects CUDA when
available and CPU otherwise. Automatic quantization selects GPTQ on CUDA and
GGUF on CPU. Set `device` to `cpu` or `cuda`, and `quantization` to `gptq`,
`gguf`, `awq`, or `none`, to override either decision. All default Qwen
checkpoints are between 7B and 9B parameters.

For direct Qwen hosting, install the appropriate vLLM build for the machine and
start its OpenAI-compatible server:

```bash
# GGUF additionally requires: uv pip install vllm-gguf-plugin
dullahan-inference serve
```

CPU and CUDA vLLM packages have different installation requirements; follow
the [official CPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/)
or the matching CUDA installation guide. vLLM currently describes GGUF support
as experimental and provides it through `vllm-gguf-plugin`.

To use Ollama instead, set `provider: ollama` and choose the desired Qwen model
tag under `ollama.model`. The proxy uses Ollama's non-streaming
[`/api/generate`](https://docs.ollama.com/api/generate) API while exposing the
OpenAI-compatible `/v1/completions` contract expected by Dullahan. Set
`ollama.launch_server: true` if Dullahan should start `ollama serve` itself.

EDL and the planner use the local OpenAI-compatible inference endpoint by default:

```bash
export EDL_MODEL_PROVIDER=http
export EDL_MODEL_BASE_URL=http://127.0.0.1:30000/v1
export AGENT_PLANNER_PROVIDER=http
export AGENT_PLANNER_BASE_URL=http://127.0.0.1:30000/v1
export AGENT_SYNTHESIS_PROVIDER=http
export AGENT_SYNTHESIS_BASE_URL=http://127.0.0.1:30000/v1
```

Start `dullahan-inference serve` before running the agent. Planning, expert
responses, and final answer synthesis fail visibly when the inference endpoint
is unavailable; there is no template-response fallback.

CUDA plans enable vLLM's `--cpu-offload-gb` and `--swap-space` controls. The
[vLLM offload documentation](https://docs.vllm.ai/en/v0.20.0/examples/basic/offline_inference/#cpu-offload)
notes that CPU offload effectively extends available GPU memory but benefits
from a fast CPU–GPU interconnect.

For persistent model storage and remote/container vLLM hosting, use the exact
two-variant workflow in [`apps/model-server/README.md`](apps/model-server/README.md).
It builds a Linux ARM64 CPU image for Apple Silicon hosts or an NVIDIA CUDA
image for Linux GPU hosts, then runs the common model-manager wrapper. Set
`model_server.enabled: true` in `configs/inference.yaml`; `device: cpu` selects
port 8001 and `device: cuda` selects port 8002. Run
`dullahan-inference activate` before sending requests to the selected `/v1`
endpoint.

The shared CPU/CUDA model manager stores complete model packages with optional
LoRA adapters, provides authenticated package CRUD, and supports compact
`lora_only` exports. Compact packages retain the model and adapter names but
resolve the base checkpoint from Hugging Face when activated. Configure the
client with `model_server.export_mode: full|lora_only`; see
`apps/model-server/README.md` for the package layout and endpoints.

For interactive practical-capability and concurrency testing, open
[`notebooks/dullahan_production_benchmark.ipynb`](notebooks/dullahan_production_benchmark.ipynb).
Install its kernel dependencies with `uv sync --extra dev --extra notebook --inexact`.
The notebook starts real CPU inference, accepts custom queries and prompts,
exports raw evidence, and plots latency, throughput, semantic coverage, and memory deltas.

### 4. Run A Local In-Process Execution

The fastest path runs the agent runtime, CAL, and EDL in one process:

```bash
dullahan-agent "Assess whether a long volatility strategy is attractive before this week's major earnings releases" --max-depth 1
```

Useful options:

```bash
dullahan-agent "Build a multi-factor trade thesis for rotating from mega-cap tech into regional banks" \
  --max-depth 2 \
  --max-breadth 3 \
  --max-total-instances 12 \
  --persist-artifacts
```

For the full structured result:

```bash
dullahan-agent "Explain the key risks in a pairs trade between two semiconductor stocks" --max-depth 1 --json
```

### 5. Inspect Artifacts

When `--persist-artifacts` is set, Dullahan writes a run folder under
`memory/executions/<trace_id>/`.

Each run contains aggregate files:

```text
queries.yaml
contexts.yaml
responses.yaml
trace.yaml
manifest.yaml
action_graph.json
action_graph.mmd
final_response.md
```

It also writes per-query instance folders:

```text
instances/<query_id>/query.yaml
instances/<query_id>/context.yaml
instances/<query_id>/responses.yaml
instances/<query_id>/summary.md
```

This is the filesystem memory surface: you can inspect what each agent asked,
what context CAL supplied, which expert EDL selected, and what the expert
returned. Each persisted `ContextBundle` also includes context optimization
metadata such as candidate token count, selected token count, tokenizer model,
and context reduction percentage for that subquery. Counts come from the
generation model's native tokenizer usage rather than word/character estimates.

### Exported Action / Inference Graph

Every persisted run also exports the completed hierarchical action graph:

| File | Purpose |
| --- | --- |
| `action_graph.json` | Machine-readable graph for downstream programs, graph databases, dashboards, notebooks, or web visualizers. |
| `action_graph.mmd` | Mermaid flowchart for quick visualization in Markdown viewers, Mermaid Live, or Mermaid CLI. |

The JSON graph uses this shape:

```json
{
  "schema": "dullahan.action_graph.v1",
  "trace_id": "trace:...",
  "root_query_id": "query:...",
  "nodes": [
    {
      "id": "query:...",
      "label": "Short query label",
      "depth": 1,
      "sender_id": "query:parent",
      "query": {},
      "context": {},
      "response": {},
      "responses": []
    }
  ],
  "edges": [
    {
      "id": "query__parent__to__query__child",
      "source": "query:parent",
      "target": "query:child",
      "query": "The subquery text that created this edge",
      "label": "Short edge label"
    }
  ]
}
```

In other words, graph nodes are query instances with their `(query, context,
response)` payloads, and graph edges are parent-to-child query delegations labeled
by the child query. `response` contains the primary expert response when one
exists, while `responses` preserves the full list. The JSON format is deliberately
plain so it can be loaded by tools such as NetworkX, Cytoscape.js, D3, Graphistry,
or a custom dashboard.

To render the Mermaid graph with Mermaid CLI:

```bash
mmdc -i memory/executions/<trace_id>/action_graph.mmd \
  -o memory/executions/<trace_id>/action_graph.svg
```

## Run CAL And EDL As Services

Start CAL and EDL with Docker Compose:

```bash
docker compose up cal edl
```

To run CAL against PostgreSQL + pgvector instead of the local JSON index, start
PostgreSQL and set `WORLD_STATE_BACKEND=postgres` for CAL:

```bash
docker compose up postgres

WORLD_STATE_BACKEND=postgres \
WORLD_STATE_POSTGRES_DSN=postgresql://dullahan:dullahan@127.0.0.1:5432/dullahan \
dullahan-cal
```

Then call them from the runtime over HTTP:

```bash
dullahan-agent "Evaluate whether a steepener trade makes sense given inflation, growth, and Fed path assumptions" \
  --transport http \
  --cal-url http://127.0.0.1:8100 \
  --edl-url http://127.0.0.1:8200 \
  --max-depth 1
```

You can also run the sample remote-agent profile:

```bash
docker compose --profile agent up --build
```

Or run services directly:

```bash
dullahan-cal
dullahan-edl
```

Default ports:

| Service | Port | Main endpoints |
| --- | ---: | --- |
| CAL | `8100` | `/augment`, `/augment/batch` |
| EDL | `8200` | `/dispatch`, `/dispatch/batch` |

Batch endpoints preserve request order. The agent runtime uses batch CAL/EDL
calls for sibling subqueries when the selected transport supports them.

## MCP Tool Surface

Dullahan exposes MCP-facing stdio tools for agent environments that want CAL and
EDL as tool calls:

```bash
dullahan-mcp-cal
dullahan-mcp-edl
```

The tools are:

| Tool | Purpose |
| --- | --- |
| `send_to_CAL` | Given a subquery and parent context, return a bounded context bundle. |
| `send_to_EDL` | Given a contextualized subquery, dispatch it to the best expert and return the response. |

Manifests live in:

```text
mcp/servers/
mcp/tools/
```

Set `CAL_BASE_URL` and `EDL_BASE_URL` to point the MCP servers at remote CAL/EDL
instances.

## Graphify, Cluster, And Generate Experts

The primary ingestion path is:

```bash
dullahan-graphify ./path/to/data --k 8
```

It performs the full pipeline:

1. Runs `graphify` on a file or directory collection.
2. Reads `graphify-out/graph.json`.
3. Converts graphify nodes and edges into Dullahan `graph.yaml`.
4. Writes Markdown node documents containing graphify metadata.
5. Partitions the imported graph into clusters of size at most `K`.
6. Rewrites `experts.yaml` so EDL can dispatch to one expert per cluster.
7. Rebuilds the local WorldStateDB vector index used by CAL retrieval.

For database-backed corpora, use `--postgres-dsn` and `--postgres-query` to pull
context rows from PostgreSQL before Graphify runs. For database-backed retrieval,
run CAL with `WORLD_STATE_BACKEND=postgres`; the PostgreSQL WorldStateDB creates
a pgvector table and uses cosine-distance ordering for RAG retrieval.

Lower-level cluster regeneration is still available when `graph.yaml` already
exists and you only want to re-cluster it:

```bash
PYTHONPATH=apps/graph-builder/src:packages/kg/src:packages/shared/src \
  python scripts/build_graph_clusters.py --k 2 --write-experts
```

### Automatically Publish Graphify Updates

The standard Graphify post-commit hook rebuilds `graphify-out/` but deliberately
leaves its output unstaged. Dullahan adds an opt-in extension that commits the
durable generated artifacts and pushes the current branch after a successful
rebuild:

```bash
graphify hook install
python scripts/install_graphify_auto_publish.py
```

The publisher creates a separate `chore: Refresh Graphify snapshot` commit, then
uses a normal non-force `git push` to the branch's configured upstream. It never
stages files outside `graphify-out/`, preserves unrelated staged work, and omits
machine-local files such as `.graphify_python`, saved query memory, reflections,
timestamped backups, and the mtime-based `cache/stat-index.json`. A detached
HEAD, missing upstream, pre-staged Graphify files, rebuild failure, or push
rejection stops publication and is recorded in `~/.cache/graphify-rebuild.log`.

The extension is installed into `.git/hooks/post-commit`, which is local Git
state. Re-run the installer after reinstalling or upgrading Graphify's hook.

## Configuration

Recursion and execution limits live in `configs/recursion.yaml`:

```yaml
max_depth: 4
max_breadth_per_agent: 6
max_total_agent_instances: 128
max_sibling_concurrency: 8
timeout_seconds_per_instance: 60
cycle_policy: reject_repeated_query_signature
```

Common environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DULLAHAN_REPO_ROOT` | Repo root used by services inside containers or external processes. | Current working directory |
| `WORLD_STATE_BACKEND` | `local` or `postgres` retrieval backend for CAL. | `local` |
| `WORLD_STATE_POSTGRES_DSN` | PostgreSQL DSN used when `WORLD_STATE_BACKEND=postgres`. | unset |
| `WORLD_STATE_POSTGRES_TABLE` | pgvector table used for WorldStateDB documents. | `world_state_documents` |
| `DULLAHAN_INFERENCE_BASE_URL` | Shared completion, embedding, and tokenizer endpoint. | `http://127.0.0.1:30000/v1` |
| `DULLAHAN_EMBEDDING_MODEL` | Semantic model used consistently by WorldStateDB and EDL. | `qwen3-embedding:0.6b` |
| `DULLAHAN_EMBEDDING_DIMENSIONS` | Vector size expected from the semantic model and pgvector. | `1024` |
| `DULLAHAN_TOKENIZER_MODEL` | Exact generation tokenizer used by CAL token accounting. | `Qwen/Qwen3-8B` |
| `DULLAHAN_INFERENCE_TIMEOUT_SECONDS` | Timeout for embedding and tokenizer inference calls. | `120` |
| `EDL_MODEL_PROVIDER` | Expert model provider. Only `http` is supported. | `http` |
| `EDL_MODEL_BASE_URL` | OpenAI-compatible model endpoint for expert execution. | `http://127.0.0.1:30000/v1` |
| `EDL_MODEL_TIMEOUT_SECONDS` | Timeout for expert model calls. | `30` |
| `EDL_MODEL_MAX_TOKENS` | Maximum completion tokens for an expert response. | `512` |
| `EDL_MAX_DISPATCH_CONCURRENCY` | Max concurrent EDL dispatch workers. | `16` |
| `AGENT_PLANNER_PROVIDER` | Planner provider. Only `http` is supported. | `http` |
| `AGENT_PLANNER_BASE_URL` | OpenAI-compatible planner endpoint. | `http://127.0.0.1:30000/v1` |
| `AGENT_PLANNER_MODEL` | Planner model name. | `local-planner` |
| `AGENT_SYNTHESIS_PROVIDER` | Final-answer provider. Only `http` is supported. | `http` |
| `AGENT_SYNTHESIS_BASE_URL` | OpenAI-compatible final synthesis endpoint. | `http://127.0.0.1:30000/v1` |
| `AGENT_SYNTHESIS_MODEL` | Model used to synthesize paired subquery answers. | `local-planner` |
| `AGENT_SYNTHESIS_TIMEOUT_SECONDS` | Timeout for final answer synthesis. | `60` |
| `AGENT_SYNTHESIS_MAX_TOKENS` | Maximum completion tokens in the final answer. | `1024` |
| `DULLAHAN_INFERENCE_CONFIG` | YAML file used by `dullahan-inference`. | `configs/inference.yaml` |

The planner, CAL, EDL, and final synthesizer require a real inference endpoint.
The default `dullahan-inference` Ollama proxy provides completion,
`/v1/embeddings`, and `/tokenize` endpoints from models placed by the same
CPU/GPU policy. Pull both default models before the first run:

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

Local vector indexes include the embedding model identifier and dimension in
their filename and payload, so legacy hash indexes are never reused. Existing
pgvector tables created with 128 dimensions must be rebuilt at 1,024 dimensions.

## How It Compares

| Framework / Pattern | Primary focus | Dullahan difference |
| --- | --- | --- |
| LangGraph | General graph-shaped agent workflows and state machines. | Dullahan focuses specifically on hierarchical task decomposition over a clustered context graph with CAL/EDL separation. |
| AutoGen-style multi-agent chat | Conversational collaboration between agents. | Dullahan treats agents as cluster specialists selected by retrieval/routing, with bounded context bundles and execution artifacts. |
| CrewAI-style role teams | Declarative role-based task delegation. | Dullahan derives experts from graph clusters and routes subqueries by attention over expert role context. |
| Basic RAG pipeline | Retrieve documents for a single model call. | Dullahan performs recursive subquery planning and expert dispatch, not just retrieve-then-answer. |
| Vector database memory alone | Similarity search over chunks. | Dullahan combines vector retrieval with graph structure, cluster ownership, expert role docs, and traceable execution. |
| Workflow orchestrators | Reliable execution of predefined steps. | Dullahan lets the agent recursively discover subqueries while still enforcing depth, breadth, timeout, and instance limits. |

## Scalability Model

Dullahan is designed to scale across several axes:

| Axis | Current mechanism | Scaling path |
| --- | --- | --- |
| Context volume | WorldStateDB indexes graph-backed Markdown documents locally or in PostgreSQL + pgvector. | Shard pgvector tables, add graph-aware retrieval, or move indexes beside CAL workers. |
| Expert count | One or more experts per cluster in `experts.yaml`. | Regenerate experts from larger graphs, split clusters by K, or specialize experts by domain and modality. |
| Subquery fanout | Breadth, depth, total-instance, timeout, and sibling-concurrency limits. | Tune per workload, add queue-backed execution, or distribute CAL/EDL workers. |
| Service deployment | Local process or HTTP CAL/EDL services. | Run CAL and EDL independently on Kubernetes, attach model-serving backends, and autoscale by request pressure. |
| Model execution | OpenAI-compatible HTTP provider backed by local Qwen inference. | Route experts to SGLang, KServe, TensorRT-LLM, or other serving stacks. |
| Observability | Execution spans and YAML/Markdown artifacts. | Export spans to OpenTelemetry, Prometheus, Grafana, or trace stores. |

The architecture maps naturally onto high-throughput inference infrastructure:
CAL can scale with retrieval load, EDL can scale with routing and expert
execution, and model servers can be independently optimized for SLM throughput.

## Domain Adaptation And Distillation

Dullahan is intended to become more domain-specific over time. A practical
adaptation loop looks like this:

1. Model the domain as graph memory: files, APIs, policies, images, incidents,
   research concepts, or business entities become nodes.
2. Attach Markdown role context to nodes and clusters.
3. Partition the graph with a chosen `K` and regenerate experts.
4. Run tasks with `--persist-artifacts`.
5. Review per-query artifacts to identify good and bad subquery decomposition,
   context retrieval, routing, and expert responses.
6. Distill specialized SLMs or adapters from successful traces.
7. Update `experts.yaml` so each cluster routes to the right specialized model.

Examples:

| Domain | Specialized expert clusters |
| --- | --- |
| Code intelligence | Build system, API surface, database layer, frontend components, infra modules. |
| Cloud operations | IAM, Kubernetes, GPU scheduling, observability, deployment automation. |
| Legal or policy review | Jurisdiction, clause type, precedent, compliance domain, evidence handling. |
| Biomedical research | Pathways, assays, datasets, papers, figures, experimental methods. |
| Customer support | Product area, account state, incident class, troubleshooting workflow. |

The long-term vision is a graph whose structure grows and morphs with use:
new documents become nodes, repeated traces reveal better clusters, and expert
models become more specialized through supervised fine-tuning, preference data,
or distillation from stronger teacher models.

## Development

Run the test suite:

```bash
pytest
```

## License

Dullahan, including the inference module and model-server wrapper, is licensed
under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for runtime and model
licensing boundaries.

Model weights and LoRA adapters are separate artifacts and are not relicensed
by Dullahan. Review the license attached to every configured or uploaded model
before use or redistribution.
