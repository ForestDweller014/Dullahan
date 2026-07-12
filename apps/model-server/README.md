# Dullahan model server containers

This directory has exactly two vLLM 0.24.0 variants:

- `cpu`: Linux CPU vLLM. On Apple Silicon, build it for `linux/arm64`.
- `cuda`: Linux NVIDIA CUDA vLLM.

Both wrappers expose the manager on port `8080` inside the container, persist
complete Hugging Face model directories under `/models`, and proxy the active
vLLM process through a stable `/v1` API. Host ports are `8001` for CPU and
`8002` for CUDA.

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

Model archives are complete Hugging Face-compatible directories. Their
`dullahan-model.json` manifest records `quantization` and
`supported_backends`. The requested automatic policy is GGUF for CPU and GPTQ
for CUDA; GPTQ, GGUF, and AWQ remain explicit configuration choices.

## Model packages and LoRA adapters

The manager stores one package per model name under `/models`. A full package
contains the base checkpoint and may contain any number of named LoRA adapters:

```text
qwen-local/
  config.json
  model.safetensors
  dullahan-model.json
  adapters/
    legal/
      adapter_config.json
      adapter_model.safetensors
```

The manifest exposes `package_mode`, `base_model`, and an `adapters` inventory.
Activation starts one base model and registers every stored adapter with vLLM;
clients select the base model or an adapter through the OpenAI-compatible
`model` request field.

A `lora_only` package contains only `dullahan-model.json` and `adapters/`. Its
manifest must name the Hugging Face base model. At activation time vLLM resolves
that base name (using `HF_TOKEN` when required) and loads the packaged adapters.

Choose the default archive format with `MODEL_EXPORT_MODE=full|lora_only` in
`.env`, override it per request with `?mode=`, or set
`model_server.export_mode` in `configs/inference.yaml` when using the inference
client:

```bash
dullahan-inference metadata
dullahan-inference export --output ./qwen-local.tar.gz
```

The authenticated CRUD endpoints are:

- `PUT /admin/models/{name}/archive?replace=false` — create or fully replace a package.
- `GET /admin/models/{name}` — read package metadata and adapter inventory.
- `GET /admin/models/{name}/archive?mode=full|lora_only` — export a package.
- `DELETE /admin/models/{name}` — delete an inactive package.
- `POST /admin/models/hf` — import a full model or LoRA adapter repository by name.

All `/admin` requests require `X-Admin-Token`. A LoRA-only export is rejected
when the package has no adapters or no portable `base_model` name.

The schema preserves that requested policy, but the target vLLM runtime remains
the final compatibility authority. In particular, vLLM 0.24.0's documented ARM
CPU quantization matrix does not guarantee GGUF execution; the Linux ARM64 CPU
image can still run supported unquantized or compressed-tensor checkpoints.

## License

The Dullahan model-server wrapper is licensed under Apache-2.0. Both wrapper
images copy `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` into
`/licenses/dullahan/`. Upstream base images, installed packages, model weights,
and LoRA adapters retain their own licenses.
