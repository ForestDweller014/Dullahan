# Dullahan model server containers

This directory has exactly two vLLM 0.24.0 variants:

- `cpu`: Linux CPU vLLM. On Apple Silicon, build it for `linux/arm64`.
- `cuda`: Linux NVIDIA CUDA vLLM.

Both wrappers expose the manager on port `8080` inside the container, persist
LoRA expert packages under `/models`, and proxy the active vLLM process through
a stable `/v1` API. Host ports are `8001` for CPU and `8002` for CUDA.

`/models` is backed by the `cpu-models` or `cuda-models` Docker named volume.
The CRUD store never contains base checkpoint weights. Base models are named in
the package manifest and resolved by vLLM at activation time; Hugging Face may
cache those shared base weights separately under `/root/.cache/huggingface` in
the `hf-cache` volume.

The local `.env` is ignored by Git and excluded from the Docker build context.
Create it from `.env.example` and set `MODEL_ADMIN_TOKEN` and `HF_TOKEN` before
building.

## Apple Silicon / Linux ARM64 CPU

Docker Desktop must be running. From this directory:

```bash
PLATFORM=linux/arm64 ./scripts/build-base.sh cpu
docker compose --env-file .env -f compose.cpu.yaml build
docker compose --env-file .env -f compose.cpu.yaml up -d
docker compose --env-file .env -f compose.cpu.yaml logs -f
```

This is a Linux ARM64 CPU container. Docker Desktop does not pass the macOS
Metal API into Linux containers. Native Apple GPU inference remains available
through Dullahan's Ollama provider.

## NVIDIA CUDA

Run this on a Linux NVIDIA host with a compatible driver and NVIDIA Container
Toolkit:

```bash
./scripts/build-base.sh cuda
docker compose --env-file .env -f compose.cuda.yaml build
docker compose --env-file .env -f compose.cuda.yaml up -d
docker compose --env-file .env -f compose.cuda.yaml logs -f
```

## Operate the containers

```bash
# Health
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8002/health

# Stop one variant without deleting its named model volume
docker compose --env-file .env -f compose.cpu.yaml down
docker compose --env-file .env -f compose.cuda.yaml down
```

Set `model_server.enabled: true` in `configs/inference.yaml`, select `device:
cpu` or `device: cuda`, and configure the stored model name. Then inspect and
activate it:

```bash
export MODEL_ADMIN_TOKEN='the-value-from-apps/model-server/.env'
dullahan-inference plan
dullahan-inference activate
```

Expert archives are adapter-only Hugging Face-compatible packages. Their
`dullahan-model.json` manifest records the remote `base_model`, quantization,
supported backends, and adapter inventory. The requested automatic policy is
GGUF for CPU and GPTQ for CUDA; GPTQ, GGUF, and AWQ remain explicit
configuration choices for the remotely resolved base model.

## Model packages and LoRA adapters

The manager stores one package per model name under `/models`. Each package
contains only a base-model reference and any number of named LoRA adapters:

```text
qwen-local/
  dullahan-model.json
  adapters/
    local-slm-legal/
      adapter_config.json
      adapter_model.safetensors
```

The absolute container paths are therefore
`/models/qwen-local/adapters/local-slm-legal/`. Metadata responses expose the
package `storage_directory`, each adapter's relative `directory`, and its
resolved `storage_path`.

The manifest exposes `package_mode: lora_only`, `base_model`, and an `adapters`
inventory. Upload validation rejects top-level base checkpoint/configuration
files. Activation starts the shared base model from its manifest reference and
registers every stored adapter with vLLM; clients select the base model or an
adapter through the OpenAI-compatible `model` request field.

LoRA concurrency is explicit rather than relying on vLLM's single-adapter
batch default. `model_server.max_loras` controls the number of distinct
adapters allowed in one execution batch, while `max_cpu_loras` controls the
host-memory adapter cache and must be at least as large. Defaults are `4` and
`8`; the activation client sends them to the manager, and raw API callers use
`VLLM_MAX_LORAS`/`VLLM_MAX_CPU_LORAS`. Larger values reserve more accelerator
and host memory, so benchmark representative adapter ranks and swarm traffic
before increasing them.

Every package contains only `dullahan-model.json` and `adapters/`. Its manifest
must name the Hugging Face base model. At activation time vLLM resolves that
base name (using `HF_TOKEN` when required) and loads the packaged adapters.
Exports are always adapter-only:

```bash
dullahan-inference metadata
dullahan-inference export --output ./qwen-local.tar.gz
```

The authenticated CRUD endpoints are:

- `PUT /admin/models/{name}/archive?replace=false` — create or fully replace a package.
- `GET /admin/models/{name}` — read package metadata and adapter inventory.
- `GET /admin/models/{name}/archive?mode=lora_only` — export an adapter-only package.
- `DELETE /admin/models/{name}` — delete an inactive package.
- `POST /admin/models/hf` — import a LoRA adapter repository as a package.
- `PUT /admin/models/{name}/adapters/{adapter}/archive` — add or replace one adapter.
- `GET /admin/models/{name}/adapters/{adapter}` — inspect one stored adapter.
- `DELETE /admin/models/{name}/adapters/{adapter}` — delete an adapter when others remain.

All `/admin` requests require `X-Admin-Token`. Package and adapter mutation is
rejected while that package is active. Every adapter upload must include
`adapter_config.json` and `.safetensors` or `.bin` weights, and its declared base
model must match the package manifest.

The schema preserves the requested base-model policy, but the target vLLM
runtime remains the final compatibility authority. In particular, vLLM 0.24.0's documented ARM
CPU quantization matrix does not guarantee GGUF execution; the Linux ARM64 CPU
image can still run supported unquantized or compressed-tensor checkpoints.

## License

The Dullahan model-server wrapper is licensed under Apache-2.0. Both wrapper
images copy `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` into
`/licenses/dullahan/`. Upstream base images, remotely resolved base-model
weights, and LoRA adapters retain their own licenses.
