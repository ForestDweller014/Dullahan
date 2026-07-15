# Dullahan

Dullahan is an experimental architecture for turning an organization's scattered
knowledge into coordinated, inspectable work. Most teams do not suffer from a
shortage of information. They suffer because the right evidence is trapped in a
repository, document, file, or database row when a decision has to be made—and
because simply pouring all of it into one enormous prompt makes the result more
expensive, less focused, and harder to trust.

Dullahan approaches that problem as if it were building a high-performing firm.
It creates a durable map of what the organization knows, divides that landscape
into areas of expertise, gives each task only the context it needs, and assigns it
to the most relevant specialist. A coordinating agent keeps the work moving,
specialists can open focused lines of inquiry, and every important step leaves an
artifact behind. The result is not just an answer, but a visible account of how
the system arrived there.

Model hosting is a deployment choice, not another layer in that reasoning flow.
Teams can connect Dullahan to an external OpenAI API-compatible service, or use
the optional **Dullahan Inference** module as the local-hosting alternative. The
local module keeps models and data on operator-controlled hardware while
presenting the same OpenAI-compatible model boundary to the agent, CAL, and EDL.
Direct use of `api.openai.com` is supported: Dullahan adapts generation to the
Responses API, sends bearer-token authentication, and uses the Embeddings API.
CAL's context accounting deliberately remains on Dullahan's custom `/tokenize`
endpoint; Dullahan never sends that custom request to OpenAI.

## Quickstart: From Source Material To An Answer

The sequence below starts at the repository root and ends with a persisted answer.
Commands that keep running should be opened in separate terminals. The default
route demonstrates the optional local-hosting choice: Ollama runs behind
Dullahan's inference proxy, where one endpoint serves the three capabilities the
system needs—generation, semantic embeddings, and native tokenization. In hosted
mode, OpenAI supplies generation and embeddings while CAL still needs a
Dullahan-compatible custom tokenizer service. The bundled inference process can
provide only that route without becoming the generation host.

### 1. Install Dullahan

```bash
python -m pip install -e ".[dev]"
```

This installs `dullahan-graphify`, `dullahan-inference`, `dullahan-cal`,
`dullahan-edl`, and `dullahan-agent` along with the MCP and benchmark commands.

### 2. Ingest Files, Documents, Or A Repository

Point `dullahan-graphify` at one file or one directory. A directory can contain
documents, source code, configuration, or an entire repository. This is the
file-ingestion step; Dullahan reads the supplied path and writes derived project
memory under `memory/` rather than copying the source into a second upload store.

```bash
dullahan-graphify /absolute/path/to/project-material \
  --repo-root "$PWD" \
  --k 8
```

The command runs Graphify, imports its graph, partitions the knowledge into
clusters, generates the expert registry and role documents, and rebuilds the
local WorldState index. The main outputs are:

```text
memory/graph/graph.yaml
memory/graph/clusters.yaml
memory/graph/experts.yaml
memory/documents/nodes/
memory/documents/clusters/
memory/world_state/indexes/local.json
```

If a Graphify graph already exists, import it without rerunning extraction:

```bash
dullahan-graphify /absolute/path/to/project-material \
  --repo-root "$PWD" \
  --from-graphify-json ./graphify-out/graph.json \
  --k 8
```

### 3. Or Build Memory From Source PostgreSQL Rows

Use this instead of the file command when the source knowledge lives in
PostgreSQL. The query must return `id`, `title`, `content`, and optionally
`metadata`:

```bash
dullahan-graphify \
  --repo-root "$PWD" \
  --postgres-dsn postgresql://dullahan:dullahan@127.0.0.1:5432/dullahan \
  --postgres-query "select id, title, content, metadata from research_notes order by id" \
  --postgres-export-dir memory/postgres_context \
  --k 8
```

The selected rows are exported as readable Markdown, then pass through exactly
the same offline graph, clustering, expert-generation, and WorldState build as
file inputs. This source database is different from the optional pgvector
database that CAL can use later as a live retrieval backend.

### 4. Start The Local Inference Alternative

This step chooses Dullahan Inference instead of a hosted OpenAI API-compatible
service. It is the recommended first run because it provides the whole model
contract locally without changing how the rest of Dullahan communicates with
models.

Install Ollama, then pull the configured generation and embedding models:

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
ollama serve
```

Leave Ollama running. In a second terminal, inspect the resolved hardware plan
and start Dullahan's OpenAI-compatible proxy on port `30000`:

```bash
cd /absolute/path/to/Dullahan
dullahan-inference plan --config configs/inference.yaml
dullahan-inference serve --config configs/inference.yaml
```

The repository defaults already point the agent runtime, CAL, and EDL at this
endpoint. These exports make that connection explicit and are useful when
launching commands from different shells:

```bash
export DULLAHAN_INFERENCE_BASE_URL=http://127.0.0.1:30000/v1
export EDL_MODEL_PROVIDER=http
export EDL_MODEL_BASE_URL=http://127.0.0.1:30000/v1
export AGENT_PLANNER_PROVIDER=http
export AGENT_PLANNER_BASE_URL=http://127.0.0.1:30000/v1
export AGENT_SYNTHESIS_PROVIDER=http
export AGENT_SYNTHESIS_BASE_URL=http://127.0.0.1:30000/v1
```

To host Qwen directly with vLLM instead, change `provider` to `qwen` in
`configs/inference.yaml`, install the platform-appropriate vLLM build, and run
the same `plan` and `serve` commands. Automatic policy selects GPTQ for CUDA and
GGUF for CPU unless the configuration overrides it. Direct vLLM is the
generation route; a complete deployment must still provide the configured
embedding and tokenization capabilities, which is why the combined Ollama proxy
is the recommended first run.

### 4B. Use The Hosted OpenAI Alternative

To use OpenAI instead of starting Dullahan Inference, export one provider choice,
one credential, and the hosted generation model:

```bash
export DULLAHAN_INFERENCE_PROVIDER=openai
export OPENAI_API_KEY=your-api-key
export OPENAI_MODEL=gpt-5-mini
export DULLAHAN_EMBEDDING_MODEL=text-embedding-3-small
export DULLAHAN_EMBEDDING_DIMENSIONS=1024
export DULLAHAN_TOKENIZER_BASE_URL=http://127.0.0.1:30000/v1
export DULLAHAN_TOKENIZER_MODEL=Qwen/Qwen3-8B
```

These shared settings configure the planner, expert execution, final synthesis,
CAL embeddings, and EDL routing embeddings. Dullahan sends generation to
`/v1/responses` and embeddings to `/v1/embeddings`. CAL independently sends token
accounting to Dullahan's custom `/tokenize` route at
`DULLAHAN_TOKENIZER_BASE_URL`; it never sends that route to OpenAI. Start
`dullahan-inference serve --config configs/inference.yaml` to provide the bundled
custom tokenizer service, or point the variable at another Dullahan-compatible
tokenizer gateway. Only the tokenizer artifact is loaded when that route is
used; OpenAI still handles generation and embeddings. Because the expert
registry contains local LoRA model aliases, hosted mode replaces those aliases
at execution time with `OPENAI_MODEL` while preserving each expert's role,
prompt, routing decision, and evidence boundary.

Use `OPENAI_BASE_URL` to point at a compatible gateway that implements these
same endpoints. Individual `AGENT_*`, `EDL_*`, and `DULLAHAN_*` settings can
still override the shared values for split deployments.

### 5. Solve A Prompt

The simplest execution keeps CAL and EDL in the agent process. With the inference
proxy still running, open another terminal from the repository root:

```bash
dullahan-agent \
  "Assess whether a long-volatility strategy is attractive before this week's major earnings releases" \
  --repo-root "$PWD" \
  --max-depth 2 \
  --max-breadth 3 \
  --max-total-instances 12 \
  --persist-artifacts
```

Add `--json` when a machine-readable result is more useful than the concise
terminal response. Persisted runs are written under
`memory/executions/<trace_id>/`.

### 6. Run CAL And EDL As Separate Services

For a service-oriented deployment, start CAL and EDL in separate terminals with
the same inference environment shown above:

```bash
cd /absolute/path/to/Dullahan
dullahan-cal
```

```bash
cd /absolute/path/to/Dullahan
dullahan-edl
```

Then tell the agent to use HTTP transport:

```bash
dullahan-agent \
  "Evaluate whether a steepener trade makes sense given inflation, growth, and Fed-path assumptions" \
  --repo-root "$PWD" \
  --transport http \
  --cal-url http://127.0.0.1:8100 \
  --edl-url http://127.0.0.1:8200 \
  --tool-timeout-seconds 120 \
  --max-depth 1 \
  --persist-artifacts
```

`dullahan-cal` listens on `8100` and `dullahan-edl` on `8200`. The local and
HTTP modes use the same context and expert contracts; the difference is only
whether those responsibilities share a process.

### 7. Optional: Run A Persistent CPU Or CUDA Model Manager

The persistent managers are the LoRA package lifecycle path. They are not needed
for the recommended Ollama quickstart. Create the ignored environment file and
set `MODEL_ADMIN_TOKEN`; set `HF_TOKEN` too when the declared base model requires
it:

```bash
cp apps/model-server/.env.example apps/model-server/.env
```

For Apple Silicon or another Linux ARM64 CPU target:

```bash
cd apps/model-server
PLATFORM=linux/arm64 ./scripts/build-base.sh cpu
docker compose --env-file .env -f compose.cpu.yaml build
docker compose --env-file .env -f compose.cpu.yaml up -d
cd ../..
```

For an NVIDIA CUDA host, use the CUDA variant instead:

```bash
cd apps/model-server
./scripts/build-base.sh cuda
docker compose --env-file .env -f compose.cuda.yaml build
docker compose --env-file .env -f compose.cuda.yaml up -d
cd ../..
```

The CPU manager is available at `http://127.0.0.1:8001`; CUDA uses `8002`.
Upload an adapter-only package containing `dullahan-model.json` and one or more
`adapters/<name>/` directories:

```bash
export MODEL_SERVER_URL=http://127.0.0.1:8001
export MODEL_ADMIN_TOKEN='the-value-from-apps/model-server/.env'

curl --fail-with-body --request PUT \
  --header "X-Admin-Token: $MODEL_ADMIN_TOKEN" \
  --form "file=@./qwen-local.tar.gz;type=application/gzip" \
  "$MODEL_SERVER_URL/admin/models/qwen-local/archive"
```

Set `model_server.enabled: true`, `model_server.model: qwen-local`, and the
matching `device: cpu` or `device: cuda` in `configs/inference.yaml`. Then inspect
and activate the stored package:

```bash
dullahan-inference plan --config configs/inference.yaml
dullahan-inference metadata --config configs/inference.yaml
dullahan-inference activate --config configs/inference.yaml
```

Activation resolves the shared base model outside `/models`, registers every
stored LoRA adapter, and exposes the active generation API through the manager's
`/v1` route. The current EDL configuration uses one base URL for both expert
generation and routing embeddings, so a model manager becomes an end-to-end EDL
endpoint only when it, or a gateway in front of it, serves both capabilities.
See `apps/model-server/README.md` for adapter-level CRUD, exports, and the
complete package schema.

## Tech Stack

| Category | Tools |
| --- | --- |
| Runtime and services | Python, FastAPI, Uvicorn, REST APIs |
| Agent integration | MCP stdio tools, OpenAI-compatible planner, expert, vLLM, and Ollama endpoints |
| Context and retrieval | Graphify, graph-backed RAG, local WorldStateDB vector index, PostgreSQL with pgvector, Qwen3 semantic embeddings |
| Data and artifacts | JSON, YAML, Markdown, Mermaid |
| Validation and schemas | Pydantic, pytest |
| Local orchestration | Docker Compose, CLI entrypoints |

## Architecture

The first view is the executive summary of that journey. The following views
then open the routing desk, the inference engine room, and finally the complete
system without changing the underlying story.

Every architecture view uses the same palette: **purple** for agent orchestration,
**teal** for context and retrieval, **orange** for expert dispatch, **red** for
inference and model serving, **gold** for graph build and persistent memory,
**blue** for CLI/MCP integration, and **slate** for external actors, shared
contracts, and configuration. Connectors inherit the color of the module they
leave, so ownership of each interaction remains visible across subsystem boundaries.

Multi-line nodes also share one internal hierarchy: the **bold first line** is the
module, service, artifact, or decision name; the *italic second line* labels the
information below it; and bullets identify its inputs, outputs, capabilities,
policies, interfaces, or stored data.

```mermaid
flowchart TD
    User["<b>Root query</b>"] --> Runtime["<b>Agent Runtime</b><br/><i>Purpose</i><br/>• Coordinate the task"]
    Runtime --> Planner["<b>Planner</b><br/><i>Purpose</i><br/>• Generate bounded subqueries"]
    Planner --> Subqueries["<b>Sibling subqueries</b><br/><i>Contents</i><br/>• Focused assignments"]
    Subqueries --> CAL["<b>CAL</b><br/><i>Purpose</i><br/>• Build focused context"]
    CAL --> ParentContext["<b>Parent context</b><br/><i>Source</i><br/>• Earlier reasoning"]
    CAL --> WorldStateDB["<b>WorldStateDB</b><br/><i>Knowledge sources</i><br/>• Graph documents<br/>• Optional pgvector index"]
    CAL --> ContextBundle["<b>ContextBundle</b><br/><i>Output</i><br/>• Token-bounded evidence"]
    ContextBundle --> EDL["<b>EDL</b><br/><i>Purpose</i><br/>• Dispatch to one expert"]
    EDL --> Router["<b>Attention router</b><br/><i>Purpose</i><br/>• Match task to expertise"]
    Router --> Experts["<b>SLM experts</b><br/><i>Specialization</i><br/>• Knowledge-graph clusters"]
    Experts --> Inference["<b>Local inference</b><br/><i>Backends</i><br/>• Qwen through vLLM<br/>• Ollama"]
    Inference --> Runtime
    Runtime --> Artifacts["<b>Execution artifacts</b><br/><i>Formats</i><br/>• YAML<br/>• Markdown"]

    class User external
    class Runtime,Planner,Subqueries orchestration
    class CAL,ParentContext,WorldStateDB,ContextBundle contextLayer
    class EDL,Router,Experts dispatchLayer
    class Inference inferenceLayer
    class Artifacts memoryLayer

    classDef external fill:#475569,stroke:#1E293B,color:#FFFFFF
    classDef orchestration fill:#6D28D9,stroke:#4C1D95,color:#FFFFFF
    classDef contextLayer fill:#0F766E,stroke:#115E59,color:#FFFFFF
    classDef dispatchLayer fill:#C2410C,stroke:#9A3412,color:#FFFFFF
    classDef inferenceLayer fill:#B91C1C,stroke:#7F1D1D,color:#FFFFFF
    classDef memoryLayer fill:#A16207,stroke:#713F12,color:#FFFFFF

    linkStyle 0 stroke:#64748B,stroke-width:2px
    linkStyle 1,2,3,12 stroke:#7C3AED,stroke-width:2px
    linkStyle 4,5,6,7 stroke:#0D9488,stroke-width:2px
    linkStyle 8,9,10 stroke:#EA580C,stroke-width:2px
    linkStyle 11 stroke:#DC2626,stroke-width:2px
```

### Expert Routing And Dispatch

Think of EDL as Dullahan's staffing desk. It loads the generated expert registry,
compares each single or batched assignment with the specialists' cluster-derived
roles, and sends the work to exactly one selected expert. Batch requests reuse
the loaded registry and can move concurrently while preserving request order.
Once routing selects an expert, a service-wide gate admits at most that profile's
`max_concurrency` active runner invocations; excess work waits without preventing
other experts from using their own capacity. Slots are released even when
inference raises an error. `max_dispatch_concurrency` remains the outer
batch-wide worker cap.

```mermaid
flowchart TB
    Request["<b>DispatchRequest</b><br/><i>Payload</i><br/>• Subquery<br/>• Bounded ContextBundle"] --> Service["<b>ExpertDispatchService</b><br/><i>Interfaces</i><br/>• /dispatch<br/>• /dispatch/batch"]

    subgraph RegistryData["Generated expert memory"]
        ExpertsYaml["<b>memory/graph/experts.yaml</b><br/><i>Stored profile fields</i><br/>• Expert ID and cluster<br/>• Model and role path<br/>• max_concurrency"]
        RoleDocs["<b>Cluster role documents</b><br/><i>Stored content</i><br/>• Expert domain descriptions"]
    end

    Service --> Registry["<b>ExpertRegistry</b><br/><i>Purpose</i><br/>• Load registered experts"]
    ExpertsYaml --> Registry
    RoleDocs --> Registry
    Registry --> Profiles["<b>Expert profiles</b><br/><i>Output</i><br/>• Validated ExpertProfile records"]

    Service --> Router["<b>AttentionRouter</b><br/><i>Purpose</i><br/>• Select the best-matched expert"]
    Request --> Router
    Profiles --> Router
    Router --> EmbedClient["<b>Embedding client</b><br/><i>Purpose</i><br/>• Request semantic vectors"]
    EmbedClient -->|"POST /v1/embeddings"| EmbeddingModel["<b>Inference embedding model</b><br/><i>Inputs</i><br/>• Subquery<br/>• Each expert role description"]
    EmbeddingModel --> Similarity["<b>Cosine similarity</b><br/><i>Processing</i><br/>• Score semantic alignment<br/>• Clamp negative scores"]
    Similarity --> Softmax["<b>Softmax distribution</b><br/><i>Output</i><br/>• Probability for each expert"]
    Softmax --> Threshold{"<b>Routing threshold</b><br/><i>Decision</i><br/>• Does the best score qualify?"}
    Threshold -->|"yes"| TopExpert["<b>Matched expert</b><br/><i>Outcome</i><br/>• Highest routing probability"]
    Threshold -->|"no"| Fallback["<b>Fallback expert</b><br/><i>Outcome</i><br/>• First expert by ID"]

    TopExpert --> Gate["<b>Per-expert concurrency gate</b><br/><i>Policy</i><br/>• Shared by single and batch dispatch<br/>• Active work stays below max_concurrency"]
    Fallback --> Gate
    Gate --> Prompt
    Request --> Prompt["<b>ExpertPromptBuilder</b><br/><i>Inputs</i><br/>• Role instructions<br/>• Subquery<br/>• Five highest-ranked context documents"]
    Prompt --> Runner["<b>ExpertRunner</b><br/><i>Purpose</i><br/>• Execute the selected expert"]
    Runner --> Provider["<b>OpenAI-compatible provider</b><br/><i>Purpose</i><br/>• Send the completion request"]
    Provider -->|"POST /v1/completions<br/>selected expert model alias"| Generation["<b>Expert generation</b><br/><i>Backends</i><br/>• vLLM<br/>• Model manager<br/>• Ollama proxy"]
    Generation --> Response["<b>ExpertResponse</b><br/><i>Outputs</i><br/>• Answer and citations<br/>• Route confidence<br/>• Model and token metadata"]

    Service -. "Batch policy: max_dispatch_concurrency workers" .-> Router

    class Request orchestration
    class Service,Registry,Profiles,Router,Similarity,Softmax,Threshold,TopExpert,Fallback,Gate,Prompt,Runner,Response dispatchLayer
    class ExpertsYaml,RoleDocs memoryLayer
    class EmbedClient,EmbeddingModel,Provider,Generation inferenceLayer

    classDef orchestration fill:#6D28D9,stroke:#4C1D95,color:#FFFFFF
    classDef dispatchLayer fill:#C2410C,stroke:#9A3412,color:#FFFFFF
    classDef inferenceLayer fill:#B91C1C,stroke:#7F1D1D,color:#FFFFFF
    classDef memoryLayer fill:#A16207,stroke:#713F12,color:#FFFFFF

    linkStyle 0,6,18 stroke:#7C3AED,stroke-width:2px
    linkStyle 1,4,5,7,8,11,12,13,14,15,16,17,19,20,23 stroke:#EA580C,stroke-width:2px
    linkStyle 2,3 stroke:#CA8A04,stroke-width:2px
    linkStyle 9,10,21,22 stroke:#DC2626,stroke-width:2px
```

### Inference Module

The inference module is Dullahan's optional local-hosting alternative to an
external OpenAI API-compatible service. It is not an extra reasoning hop: when
it is enabled, the agent, CAL, and EDL simply send their configured model calls
to this local boundary instead of a hosted one. Internally, it turns an
operator's declarative policy into a concrete, inspectable
`ResolvedInferencePlan` so agents do not have to reason about hardware. Serving
then follows one of three paths: a direct vLLM process, an OpenAI-compatible
Ollama proxy, or one of the persistent Docker model servers. The default Ollama
path exposes generation, embeddings, and native tokenization through one
Dullahan endpoint. In a hosted OpenAI deployment, its custom `/tokenize` route
can instead remain as a narrow CAL accounting sidecar while generation and
embeddings bypass it.

```mermaid
flowchart TB
    Config["<b>configs/inference.yaml</b><br/><i>Policies</i><br/>• Provider and device<br/>• Models and quantization<br/>• Offload and serving"] --> Load["<b>InferenceConfig</b><br/><i>Purpose</i><br/>• Validate operator policy"]
    Command["<b>dullahan-inference</b><br/><i>Commands</i><br/>• plan and serve<br/>• activate and metadata<br/>• export"] --> Load
    Load --> Detect["<b>Device detection</b><br/><i>Purpose</i><br/>• Inspect available hardware"]
    Detect --> Inventory["<b>DeviceInventory</b><br/><i>Detection order</i><br/>• NVIDIA CUDA<br/>• Apple Metal<br/>• CPU"]
    Inventory --> Quantize["<b>Quantization resolver</b><br/><i>Policies</i><br/>• Ollama uses GGUF<br/>• Qwen on CUDA uses GPTQ<br/>• Qwen on CPU or Metal uses GGUF"]
    Quantize --> Memory["<b>Memory planner</b><br/><i>Responsibilities</i><br/>• Estimate model and runtime memory<br/>• Reserve system RAM<br/>• Size CPU offload per GPU"]
    Memory --> Plan["<b>ResolvedInferencePlan</b><br/><i>Outputs</i><br/>• Engine and model<br/>• Endpoint and command<br/>• Offload, fit status, and notes"]
    Plan --> Fit{"<b>Serving mode</b><br/><i>Decision</i><br/>• Which runtime path applies?"}

    subgraph Direct["Direct Qwen / vLLM"]
        VllmCommand["<b>vLLM launch command</b><br/><i>Arguments</i><br/>• Quantization, tokenizer, and swap<br/>• GPU utilization and CPU offload<br/>• Tensor parallelism"]
        VllmProcess["<b>vLLM process</b><br/><i>Lifecycle</i><br/>• Replaces the inference CLI process"]
        VllmApi["<b>Generation API</b><br/><i>Interface</i><br/>• OpenAI-compatible /v1<br/>• Default port 30000"]
        VllmCommand --> VllmProcess
        VllmProcess --> VllmApi
    end

    subgraph OllamaPath["Ollama compatibility path"]
        OllamaApp["<b>FastAPI compatibility proxy</b><br/><i>Responsibilities</i><br/>• Translate Dullahan requests<br/>• Optionally manage Ollama"]
        OllamaGenerate["<b>Ollama generation</b><br/><i>Upstream interface</i><br/>• /api/generate"]
        OllamaEmbed["<b>Ollama embeddings</b><br/><i>Upstream interface</i><br/>• /api/embed"]
        NativeTokenizer["<b>ModelTokenizer</b><br/><i>Capability</i><br/>• Count native model tokens"]
        OllamaApi["<b>Dullahan inference API</b><br/><i>Interfaces</i><br/>• /v1/completions<br/>• /v1/embeddings<br/>• /tokenize and /v1/tokenize"]
        OllamaApp --> OllamaGenerate
        OllamaApp --> OllamaEmbed
        OllamaApp --> NativeTokenizer
        OllamaGenerate --> OllamaApi
        OllamaEmbed --> OllamaApi
        NativeTokenizer --> OllamaApi
    end

    subgraph Managed["Persistent model-server path"]
        AdminClient["<b>Inference admin client</b><br/><i>Authentication</i><br/>• X-Admin-Token"]
        CpuManager["<b>CPU model manager</b><br/><i>Endpoint</i><br/>• Port 8001"]
        CudaManager["<b>CUDA model manager</b><br/><i>Endpoint</i><br/>• Port 8002"]
        ModelStore["<b>/models volume</b><br/><i>Stored data</i><br/>• Adapter-only expert packages"]
        HfHub["<b>Hugging Face Hub</b><br/><i>External sources</i><br/>• Base-model references<br/>• Adapter repositories"]
        ManagedVllm["<b>Managed vLLM process</b><br/><i>Loaded assets</i><br/>• Shared base model<br/>• Registered adapters"]
        StableApi["<b>Managed generation API</b><br/><i>Interface</i><br/>• Stable proxied /v1"]
        AdminClient -->|"activate / metadata / export"| CpuManager
        AdminClient -->|"activate / metadata / export"| CudaManager
        HfHub --> ModelStore
        ModelStore --> CpuManager
        ModelStore --> CudaManager
        CpuManager --> ManagedVllm
        CudaManager --> ManagedVllm
        ManagedVllm --> StableApi
    end

    Fit -->|"provider=qwen<br/>model_server=false"| VllmCommand
    Fit -->|"provider=ollama"| OllamaApp
    Fit -->|"model_server=true"| AdminClient

    GenerationConsumers["<b>Generation consumers</b><br/><i>Capabilities used</i><br/>• Agent planning and synthesis<br/>• EDL expert execution<br/>• Per-expert gating in EDL"]
    SemanticConsumers["<b>Semantic consumers</b><br/><i>Capabilities used</i><br/>• CAL retrieval and token budgeting<br/>• EDL expert routing"]
    VllmApi --> GenerationConsumers
    StableApi --> GenerationConsumers
    OllamaApi --> GenerationConsumers
    OllamaApi --> SemanticConsumers

    class Config foundationLayer
    class Command integrationLayer
    class Load,Detect,Inventory,Quantize,Memory,Plan,Fit,VllmCommand,VllmProcess,VllmApi,OllamaApp,OllamaGenerate,OllamaEmbed,NativeTokenizer,OllamaApi,AdminClient,CpuManager,CudaManager,ManagedVllm,StableApi inferenceLayer
    class HfHub,ModelStore memoryLayer
    class GenerationConsumers orchestration
    class SemanticConsumers contextLayer

    classDef orchestration fill:#6D28D9,stroke:#4C1D95,color:#FFFFFF
    classDef contextLayer fill:#0F766E,stroke:#115E59,color:#FFFFFF
    classDef inferenceLayer fill:#B91C1C,stroke:#7F1D1D,color:#FFFFFF
    classDef memoryLayer fill:#A16207,stroke:#713F12,color:#FFFFFF
    classDef integrationLayer fill:#1D4ED8,stroke:#1E3A8A,color:#FFFFFF
    classDef foundationLayer fill:#475569,stroke:#1E293B,color:#FFFFFF

    linkStyle 0 stroke:#64748B,stroke-width:2px
    linkStyle 1 stroke:#2563EB,stroke-width:2px
    linkStyle 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,21,22,23,24,25,26,27,28,29,30 stroke:#DC2626,stroke-width:2px
    linkStyle 18,19,20 stroke:#CA8A04,stroke-width:2px
```

### Complete Project Architecture

This view combines the offline graph-memory build path with the online execution
path. Solid arrows show runtime or data flow; dashed arrows show shared contracts,
configuration, or an alternative transport boundary.

```mermaid
flowchart TB
    subgraph Inputs["Inputs and integration surfaces"]
        User["<b>User or automation</b>"]
        AgentCli["<b>dullahan-agent CLI</b><br/><i>Purpose</i><br/>• Submit a root query"]
        McpClient["<b>MCP client</b><br/><i>Purpose</i><br/>• Call context or expert tools"]
        Corpus["<b>File-based knowledge</b><br/><i>Source types</i><br/>• Repositories<br/>• Documents<br/>• Configuration"]
        SourcePostgres["<b>Source PostgreSQL</b><br/><i>Input</i><br/>• Rows selected by an ingestion query"]
    end

    subgraph Build["Offline knowledge build"]
        GraphBuilder["<b>apps/graph-builder</b><br/><i>Purpose</i><br/>• Orchestrate the knowledge build"]
        Graphify["<b>Graphify</b><br/><i>Capabilities</i><br/>• Structural extraction<br/>• Semantic extraction"]
        KgPackage["<b>packages/kg</b><br/><i>Capabilities</i><br/>• KnowledgeGraph model<br/>• YAML persistence<br/>• Topology-aware K partitioning"]
        ExpertGenerator["<b>Expert generators</b><br/><i>Outputs</i><br/>• Cluster definitions<br/>• Expert profiles and role documents"]
        WorldIndexBuild["<b>packages/world-state</b><br/><i>Build task</i><br/>• Rebuild the graph-document index"]
    end

    subgraph Memory["Persistent project memory"]
        GraphMemory["<b>memory/graph</b><br/><i>Stored data</i><br/>• graph.yaml<br/>• clusters.yaml"]
        ExpertMemory["<b>Expert memory</b><br/><i>Stored data</i><br/>• experts.yaml<br/>• Cluster role documents"]
        LocalIndex["<b>Local WorldState index</b><br/><i>Stored data</i><br/>• Semantic vectors for graph documents"]
        ExecutionMemory["<b>memory/executions</b><br/><i>Stored data</i><br/>• Queries and contexts<br/>• Responses and traces<br/>• Action graphs"]
    end

    subgraph Online["Online hierarchical execution"]
        Runtime["<b>apps/agent-runtime</b><br/><i>Purpose</i><br/>• Coordinate recursive execution"]
        Planner["<b>Planner</b><br/><i>Output</i><br/>• Bounded subqueries"]
        Guard["<b>RecursionGuard</b><br/><i>Policies</i><br/>• Depth and breadth limits<br/>• Total-instance limit<br/>• Deduplication and timeouts"]
        CalTools["<b>CAL tool adapters</b><br/><i>Interfaces</i><br/>• Local single request<br/>• HTTP single or batch requests"]
        EdlTools["<b>EDL tool adapters</b><br/><i>Interfaces</i><br/>• Local single request<br/>• HTTP single or batch requests"]
        Aggregator["<b>ResponseAggregator</b><br/><i>Purpose</i><br/>• Synthesize final evidence"]
        TraceStore["<b>Execution recorders</b><br/><i>Responsibilities</i><br/>• Collect traces<br/>• Persist execution artifacts"]
    end

    subgraph Services["Context and expert services"]
        Cal["<b>apps/cal</b><br/><i>Capabilities</i><br/>• Build bounded context<br/>• /augment<br/>• /augment/batch"]
        WorldState["<b>packages/world-state</b><br/><i>Backends</i><br/>• LocalWorldStateDB<br/>• PostgresWorldStateDB"]
        PgVector["<b>PostgreSQL with pgvector</b><br/><i>Role</i><br/>• Optional live WorldState backend"]
        Edl["<b>apps/edl</b><br/><i>Capabilities</i><br/>• Route and execute experts<br/>• /dispatch<br/>• /dispatch/batch"]
        Router["<b>Expert routing</b><br/><i>Components</i><br/>• ExpertRegistry<br/>• AttentionRouter"]
        ExpertGate["<b>Expert concurrency gate</b><br/><i>Policy</i><br/>• Enforce profile max_concurrency<br/>• Scope capacity to one EDL process"]
        ExpertRunner["<b>Expert execution</b><br/><i>Components</i><br/>• Prompt builder<br/>• ExpertRunner"]
    end

    subgraph ModelBoundary["Model and hardware boundary"]
        Inference["<b>apps/inference</b><br/><i>Responsibilities</i><br/>• Resolve serving policy<br/>• Detect hardware<br/>• Serve models or administer managers"]
        Ollama["<b>Ollama</b><br/><i>Capabilities</i><br/>• Generation<br/>• Embeddings"]
        DirectVllm["<b>Direct vLLM</b><br/><i>Execution targets</i><br/>• CPU<br/>• CUDA"]
        ModelServer["<b>apps/model-server</b><br/><i>Capabilities</i><br/>• Persistent CPU or CUDA managers<br/>• Adapter package lifecycle"]
        ManagedVllm["<b>Managed vLLM process</b><br/><i>Lifecycle</i><br/>• Started by the active model manager"]
    end

    subgraph Integration["External agent integration"]
        McpServers["<b>apps/mcp-servers</b><br/><i>Interface</i><br/>• stdio JSON-RPC"]
        CalMcp["<b>send_to_CAL</b><br/><i>Purpose</i><br/>• Request bounded context"]
        EdlMcp["<b>send_to_EDL</b><br/><i>Purpose</i><br/>• Request expert execution"]
    end

    subgraph Foundation["Shared foundation and configuration"]
        Shared["<b>packages/shared</b><br/><i>Capabilities</i><br/>• Typed contracts and IDs<br/>• Embedding and token clients<br/>• Lexical retrieval"]
        RuntimeConfig["<b>Runtime configuration</b><br/><i>Settings</i><br/>• Recursion policy<br/>• Planner and synthesis environment"]
        InferenceConfig["<b>Inference configuration</b><br/><i>Settings</i><br/>• Model, hardware, and serving policy"]
        ServiceConfig["<b>CAL and EDL configuration</b><br/><i>Settings</i><br/>• Environment-backed service policy"]
        ReferencePolicies["<b>Reference policies</b><br/><i>Examples</i><br/>• Local execution<br/>• Retrieval<br/>• Routing<br/><i>Note: not loaded directly by services</i>"]
    end

    User --> AgentCli
    AgentCli --> Runtime
    RuntimeConfig -.-> Runtime
    Runtime --> Planner
    Runtime --> Guard
    Planner -->|"generated sibling QueryEnvelopes"| CalTools
    Guard --> Planner
    CalTools -->|"in-process call or HTTP :8100"| Cal
    Cal --> WorldState
    GraphMemory --> WorldState
    LocalIndex --> WorldState
    PgVector <--> WorldState
    Cal -->|"bounded ContextBundle"| EdlTools
    EdlTools -->|"in-process call or HTTP :8200"| Edl
    ExpertMemory --> Router
    Edl --> Router
    Router --> ExpertGate
    ExpertGate --> ExpertRunner
    ExpertRunner -->|"ExpertResponse"| Runtime
    Runtime --> Aggregator
    Aggregator -->|"final response"| User
    Planner -->|"planning completion"| Inference
    Aggregator -->|"synthesis completion"| Inference
    Cal -->|"Embeddings and exact token counts"| Inference
    Router -->|"expert-role embeddings"| Inference
    ExpertRunner -->|"expert completion"| Inference
    Runtime --> TraceStore
    TraceStore --> ExecutionMemory

    InferenceConfig -.-> Inference
    ServiceConfig -.-> Cal
    ServiceConfig -.-> Edl
    Inference --> Ollama
    Inference --> DirectVllm
    Inference -->|"admin activate / inspect / export"| ModelServer
    ModelServer --> ManagedVllm

    McpClient --> McpServers
    McpServers -. "reuses agent-runtime HTTP adapters" .-> CalTools
    McpServers -. "reuses agent-runtime HTTP adapters" .-> EdlTools
    McpServers --> CalMcp
    CalMcp -->|"HTTP"| Cal
    McpServers --> EdlMcp
    EdlMcp -->|"HTTP"| Edl

    Corpus --> GraphBuilder
    SourcePostgres --> GraphBuilder
    GraphBuilder -->|"run external CLI"| Graphify
    Graphify -->|"graphify-out/graph.json"| GraphBuilder
    GraphBuilder --> KgPackage
    KgPackage --> GraphMemory
    GraphMemory --> ExpertGenerator
    ExpertGenerator --> ExpertMemory
    GraphBuilder --> WorldIndexBuild
    KgPackage -.-> WorldIndexBuild
    WorldIndexBuild --> LocalIndex

    Shared -. "typed contracts and clients" .-> Runtime
    Shared -.-> Cal
    Shared -.-> Edl
    Shared -.-> McpServers
    Shared -.-> WorldState
    Shared -.-> GraphBuilder

    class User external
    class AgentCli,McpClient,McpServers,CalMcp,EdlMcp integrationLayer
    class Corpus,SourcePostgres,GraphBuilder,Graphify,KgPackage,ExpertGenerator,WorldIndexBuild,GraphMemory,ExpertMemory,LocalIndex,ExecutionMemory memoryLayer
    class Runtime,Planner,Guard,CalTools,EdlTools,Aggregator,TraceStore orchestration
    class Cal,WorldState,PgVector contextLayer
    class Edl,Router,ExpertGate,ExpertRunner dispatchLayer
    class Inference,Ollama,DirectVllm,ModelServer,ManagedVllm inferenceLayer
    class Shared,RuntimeConfig,InferenceConfig,ServiceConfig,ReferencePolicies foundationLayer

    classDef external fill:#475569,stroke:#1E293B,color:#FFFFFF
    classDef orchestration fill:#6D28D9,stroke:#4C1D95,color:#FFFFFF
    classDef contextLayer fill:#0F766E,stroke:#115E59,color:#FFFFFF
    classDef dispatchLayer fill:#C2410C,stroke:#9A3412,color:#FFFFFF
    classDef inferenceLayer fill:#B91C1C,stroke:#7F1D1D,color:#FFFFFF
    classDef memoryLayer fill:#A16207,stroke:#713F12,color:#FFFFFF
    classDef integrationLayer fill:#1D4ED8,stroke:#1E3A8A,color:#FFFFFF
    classDef foundationLayer fill:#475569,stroke:#1E293B,color:#FFFFFF

    linkStyle 0 stroke:#64748B,stroke-width:2px
    linkStyle 1,35,36,37,38,39,40,41 stroke:#2563EB,stroke-width:2px
    linkStyle 2,28,29,30,53,54,55,56,57,58 stroke:#64748B,stroke-width:2px
    linkStyle 3,4,5,6,7,13,19,20,21,22,26,27 stroke:#7C3AED,stroke-width:2px
    linkStyle 8,11,12,23 stroke:#0D9488,stroke-width:2px
    linkStyle 15,16,17,18,24,25 stroke:#EA580C,stroke-width:2px
    linkStyle 31,32,33,34 stroke:#DC2626,stroke-width:2px
    linkStyle 9,10,14,42,43,44,45,46,47,48,49,50,51,52 stroke:#CA8A04,stroke-width:2px
```

The key runtime contracts are:

| Contract | Meaning |
| --- | --- |
| `QueryEnvelope` | The root query or generated subquery, including sender, query ID, depth, and metadata. |
| `ContextBundle` | Documents retrieved for a specific query, with optional token budget. |
| `ExpertProfile` | A specialist agent bound to a graph cluster and role context document. |
| `ExpertResponse` | The answer returned by an expert for one contextualized subquery. |
| `ExecutionSpan` | Trace metadata for runtime, CAL, EDL, timeout, and subquery events. |

## From Source Material To An Answer

The architectural bet is simple: useful AI depends as much on managing context
and responsibility as it does on choosing a model. Dullahan is therefore designed
around controlled context, modular specialists, shared base-model capacity, and
persistent project memory rather than one opaque model call.

Imagine Dullahan joining a project for the first time. The project already has a
history: source code and configuration in repositories, product and research
documents, loose Markdown files, and operational facts stored as rows in
PostgreSQL. Those materials remain the source of truth. Dullahan does not try to
replace them with a new proprietary vault. Instead, it builds a navigational layer
over them so that an agent can find relationships that no folder tree, keyword
search, or database table can explain on its own.

### The offline knowledge build creates the map

That journey begins with Graphify and `apps/graph-builder`. Graphify reads files,
documents, and repositories and identifies the concepts and relationships that
connect them. When the source is PostgreSQL, the graph-builder can export selected
rows into temporary, human-readable Markdown so the same offline process can
understand database knowledge without coupling every future answer to a live
production system. This separation matters: teams can build and review knowledge
deliberately, protect operational databases from unpredictable agent traffic, and
still preserve the meaning and provenance of the original rows.

`packages/kg` turns that material into a knowledge graph and partitions the graph
into coherent neighborhoods. Those neighborhoods are more than search buckets.
They become the foundations of expertise: the system writes cluster documents
that describe each domain and produces an expert registry that says which
specialist owns it. `packages/world-state` then makes the graph-backed documents
retrievable by meaning, using either a lightweight local index or PostgreSQL with
pgvector when a larger shared deployment needs it.

This is the first important distinction in Dullahan's context story. A source
PostgreSQL database contributes business knowledge to the offline build; the
optional WorldState PostgreSQL database is an operational retrieval service for
the memory that build produced. One is an input to understand. The other is a way
to serve the resulting memory efficiently.

The outputs live under `memory/`: the graph, cluster documents, expert registry,
retrieval indexes, and later the records of completed executions. This persistent
project memory is not a transcript of everything an agent has ever seen. It is a
curated working map that can survive across runs, be inspected by a person, and be
rebuilt when the underlying sources change. That solves a common failure mode in
agent systems: every new session no longer has to rediscover the organization from
scratch.

### The agent runtime turns a question into managed work

When a person submits a question, `apps/agent-runtime` acts like the engagement
lead. Its planner decides whether the problem can be answered directly or should
be broken into smaller subqueries. Its recursive execution loop allows those
subqueries to open their own bounded investigations, while configured depth,
branching, and budget limits prevent the work from expanding without control.

The runtime's tool layer is how it asks for context and expertise. Its aggregation
layer brings specialist responses back into one coherent answer. Its tracing and
artifact components preserve the execution as readable YAML and Markdown. These
submodules exist for the same reason a well-run project separates planning,
research, delivery, and record keeping: each responsibility can evolve or be
audited without turning the entire system into a single inseparable agent loop.

### CAL gives each task a useful field of view

Before a specialist sees a subquery, `apps/cal`—the Context Augmentation
Layer—builds its working brief. CAL combines the useful conclusions from the
parent task with relevant material from WorldState, removes duplicate or weak
context, and respects a token budget. Its retrieval, merging, and budgeting
submodules are deliberately separate because relevance is not the same thing as
volume. The goal is to give a specialist enough evidence to reason well without
burying the assignment inside the entire project archive.

That bounded `ContextBundle` is one of Dullahan's central design choices. It makes
context a managed asset with a visible origin, not an invisible side effect of a
large prompt. It also means a team can improve retrieval or budgeting without
rewriting the planner or the expert models.

### EDL staffs the task with the right specialist

`apps/edl`—the Expert Dispatch Layer—receives that brief and decides who should
handle it. Its registry loads the experts created from the knowledge graph. Its
attention router compares the subquery with each expert's domain description. Its
gate enforces the capacity promised by each expert profile, and its prompt builder,
runner, and provider turn the selected role and context into a model request.

This is closer to staffing a case team than broadcasting a prompt to a swarm.
Only the most relevant expert is selected for a given subquery, the route is
recorded, and `max_concurrency` limits how many instances of that expert may run
at once. Excess work waits for that specialist while unrelated experts retain
their own capacity. That protects expensive inference resources without forcing
the whole organization of agents into a single global queue.

Out of the box, an expert is differentiated by its cluster-derived role context:
what it knows, what it is responsible for, and which model identity it requests.
The same registry can also point those identities at LoRA adapters, allowing
specialists to gain weight-level behavior without storing a full foundation model
for every role. Role documents make specialization immediate and inspectable;
LoRA adapters provide a path to deeper learned specialization when training data
is available.

### The inference layer turns policy into capacity

`apps/inference` is the operations layer beneath those specialists. It translates
configuration into an explicit serving plan: CPU or CUDA, direct vLLM or an
Ollama-compatible route, local execution or an offloaded endpoint. Its device,
configuration, planning, tokenization, proxy, and benchmarking components keep
hardware policy out of the agent's reasoning. The agent asks for a model identity;
the inference layer decides how that request can be served.

`apps/model-server` owns the longer-lived model lifecycle. Its CPU and CUDA
services can receive, inspect, activate, and remove expert packages through a
small administrative API. The package store is intentionally adapter-only. Each
server keeps LoRA packages under `/models/<package>/adapters/<adapter>/`, while
the shared base model is named in the package manifest and resolved through the
separate runtime cache. Dullahan therefore does not duplicate billions of base
weights every time a new expert is created. Many specialist adapters can share a
compatible base model; a different base-model family requires its own serving
capacity.

This boundary also explains the two levels of concurrency in the project. EDL
controls how much work may be assigned to an expert, while the inference server
controls how that admitted work uses the underlying hardware. Separating those
concerns lets Dullahan scale specialist demand without pretending that compute is
unlimited.

### The answer becomes part of the project's history

As specialist responses return, the agent runtime's aggregation layer assembles
them into the final result. Citations, routing decisions, model metadata, and the
shape of the recursive execution can be persisted alongside the answer. The
execution artifact store is therefore more than logging: it is the project's
institutional record of what was asked, which evidence was used, which expert was
trusted, and what the system concluded.

That record supports review today and improvement tomorrow. A team can inspect a
weak route, rebuild memory after its sources change, replay a task against a new
model, or use successful traces as material for future evaluation and
distillation. Dullahan's memory is cyclical: source knowledge informs the graph,
the graph informs the experts, experts produce artifacts, and those artifacts help
the organization improve the next run.

### The connective tissue keeps the system replaceable

The remaining modules make that journey usable outside a demo. `apps/mcp-servers`
exposes CAL and EDL as small MCP tools so other agents and applications can request
context or expert work without adopting the whole runtime. `packages/shared`
provides the common contracts, identifiers, embedding and tokenization clients,
and retrieval helpers that let independently deployed services agree on what a
subquery, context bundle, route, and response mean. `configs/` captures the policy
choices—recursion, retrieval, routing, inference, and capacity—that operators
should be able to change without rewriting code.

Together, these boundaries make Dullahan a platform for experimentation rather
than a bet on one frozen stack. A team can replace the vector backend, improve the
router, introduce a new model server, expose the capabilities through MCP, or
change how memory is partitioned while keeping the rest of the story intact:
understand the organization's knowledge, give each task the right field of view,
assign accountable expertise, and preserve what happened.

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

## Execution Artifacts

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
and context reduction percentage for that subquery. Every deployment counts
through Dullahan's configured custom `/tokenize` endpoint; hosted OpenAI mode
does not redirect accounting to an OpenAI endpoint. Dullahan does not substitute
word or character estimates.

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

To run CAL against PostgreSQL with pgvector instead of the local JSON index, start
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
| `DULLAHAN_INFERENCE_PROVIDER` | Shared inference contract: `http` for local compatible endpoints or `openai` for the hosted Responses API. | `http` |
| `DULLAHAN_INFERENCE_BASE_URL` | Optional shared inference URL override. In OpenAI mode, `OPENAI_BASE_URL` is used when this is unset. | `http://127.0.0.1:30000/v1` locally |
| `DULLAHAN_INFERENCE_API_KEY` | Optional shared bearer token; falls back to `OPENAI_API_KEY` in OpenAI mode. | unset |
| `OPENAI_API_KEY` | Bearer credential required when the shared provider is `openai`. | unset |
| `OPENAI_BASE_URL` | Hosted Responses and Embeddings API base URL. | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Hosted generation model shared by planning, experts, and synthesis. | `gpt-5-mini` |
| `DULLAHAN_EMBEDDING_MODEL` | Semantic model used consistently by WorldStateDB and EDL. | `qwen3-embedding:0.6b` locally; `text-embedding-3-small` on OpenAI |
| `DULLAHAN_EMBEDDING_DIMENSIONS` | Vector size expected from the semantic model and pgvector. | `1024` |
| `DULLAHAN_TOKENIZER_BASE_URL` | Independent Dullahan-compatible custom `/tokenize` service used by CAL in both provider modes. | selected local inference URL; `http://127.0.0.1:30000/v1` in OpenAI mode |
| `DULLAHAN_TOKENIZER_API_KEY` | Optional bearer token for the custom tokenizer service. | shared key in local `http` mode; unset in OpenAI mode |
| `DULLAHAN_TOKENIZER_MODEL` | Native tokenizer model used by CAL context accounting. | `Qwen/Qwen3-8B` |
| `DULLAHAN_INFERENCE_TIMEOUT_SECONDS` | Timeout for embedding and tokenizer inference calls. | `120` |
| `EDL_MODEL_PROVIDER` | Optional expert provider override: `http` or `openai`. | shared provider |
| `EDL_MODEL_BASE_URL` | OpenAI-compatible model endpoint for expert execution. | `http://127.0.0.1:30000/v1` |
| `EDL_MODEL_API_KEY` | Optional expert endpoint bearer-token override. | shared API key |
| `EDL_MODEL` | Optional generation-model override for all hosted expert calls. | `OPENAI_MODEL` in OpenAI mode; expert alias locally |
| `EDL_MODEL_TIMEOUT_SECONDS` | Timeout for expert model calls. | `30` |
| `EDL_MODEL_MAX_TOKENS` | Maximum completion tokens for an expert response. | `512` |
| `EDL_MAX_DISPATCH_CONCURRENCY` | Max concurrent EDL dispatch workers. | `16` |
| `AGENT_PLANNER_PROVIDER` | Optional planner provider override: `http` or `openai`. | shared provider |
| `AGENT_PLANNER_BASE_URL` | OpenAI-compatible planner endpoint. | `http://127.0.0.1:30000/v1` |
| `AGENT_PLANNER_API_KEY` | Optional planner bearer-token override. | shared API key |
| `AGENT_PLANNER_MODEL` | Planner model name. | `local-planner` |
| `AGENT_SYNTHESIS_PROVIDER` | Optional final-answer provider override: `http` or `openai`. | shared provider |
| `AGENT_SYNTHESIS_BASE_URL` | OpenAI-compatible final synthesis endpoint. | `http://127.0.0.1:30000/v1` |
| `AGENT_SYNTHESIS_API_KEY` | Optional final-synthesis bearer-token override. | shared API key |
| `AGENT_SYNTHESIS_MODEL` | Model used to synthesize paired subquery answers. | `local-planner` |
| `AGENT_SYNTHESIS_TIMEOUT_SECONDS` | Timeout for final answer synthesis. | `60` |
| `AGENT_SYNTHESIS_MAX_TOKENS` | Maximum completion tokens in the final answer. | `1024` |
| `DULLAHAN_INFERENCE_CONFIG` | YAML file used by `dullahan-inference`. | `configs/inference.yaml` |

The planner, CAL, EDL, and final synthesizer require real inference endpoints.
In local mode, the default `dullahan-inference` Ollama proxy provides completion,
`/v1/embeddings`, and Dullahan's custom `/tokenize` endpoint from models placed
by the same CPU/GPU policy. Pull both default models before the first run:

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
| Context volume | WorldStateDB indexes graph-backed Markdown documents locally or in PostgreSQL with pgvector. | Shard pgvector tables, add graph-aware retrieval, or move indexes beside CAL workers. |
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
