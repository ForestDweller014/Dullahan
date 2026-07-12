# Local Inference

The inference application resolves `configs/inference.yaml` into either a Qwen
vLLM server or an OpenAI-compatible proxy backed by Ollama. Device and
quantization selection are deterministic and can be inspected without loading a
model:

```bash
dullahan-inference plan
dullahan-inference serve
```

The Qwen provider requires a platform-appropriate vLLM installation. GGUF
serving also requires `vllm-gguf-plugin`. The Ollama provider requires the
`ollama` executable when `ollama.launch_server` is enabled.

## Memory and offload policy

Automatic selection uses a 4-bit GGUF model on CPU/Metal and a 4-bit GPTQ model
on CUDA. It does not automatically quantize below
`offload.minimum_quantization_bits` (4 by default). On CUDA, any remaining VRAM
shortfall is assigned to vLLM CPU offload, up to
`offload.max_cpu_offload_gb` per device. The plan reserves system memory for the
OS and reports `memory_fit: false` when host memory cannot absorb the shortfall.

`--swap-space` is retained for vLLM request-state headroom; it is not treated as
disk-backed model-weight offload. CPU and Apple unified-memory execution must fit
the quantized model in usable system memory. Apple Metal acceleration is exposed
through Ollama because vLLM does not provide a native Metal device.

Ollama reasoning is disabled by default (`ollama.think: false`) so short token
budgets return visible text instead of being consumed entirely by Qwen3's hidden
thinking. Enable it in config when a workload benefits from reasoning traces.

Run the deterministic suite normally. The real-model smoke test is opt-in:

```bash
pytest apps/inference/tests
DULLAHAN_RUN_LOCAL_INFERENCE=1 pytest \
  apps/inference/tests/test_local_inference.py -m local_inference -v
```

## GGUF performance benchmark

The benchmark runner verifies that the selected Ollama model is GGUF, unloads
it for a cold-start sample, performs warmups, then repeats factual, structured,
long-context, and code-generation prompts with deterministic sampling:

```bash
dullahan-benchmark-gguf \
  --model qwen3:8b \
  --warmups 1 \
  --repetitions 3 \
  --output apps/inference/benchmark-results/qwen3-8b.json
```

Add `--num-gpu 0` for a controlled CPU-placement comparison. Residency is still
read from Ollama after every run, so the report shows whether the installed
Ollama version honored the override. Add `--quick --warmups 0 --repetitions 1`
to keep that comparison to one short cold and one short warm generation.

When the package entrypoint is not installed, run the repository script:

```bash
PYTHONPATH=apps/inference/src python \
  apps/inference/scripts/benchmark_gguf.py \
  --model qwen3:8b --warmups 1 --repetitions 3
```

Each JSON report contains:

- quantization format, level, parameter count, stored bytes, and effective
  storage bits per parameter;
- cold model-load time, wall latency, streamed time-to-first-token, prompt
  throughput, and generation tokens per second;
- Ollama process RSS and macOS active/wired/compressed system-memory samples;
- model-resident bytes, accelerator-resident bytes, inferred host-offload bytes,
  and accelerator residency percentage;
- disk-space delta and deterministic response hashes for repeated prompts.

Generated reports are local machine artifacts and are ignored by Git. The
opt-in regression test enforces conservative headroom around model format,
quantization density, cold load, TTFT, throughput, RAM growth, and residency:

```bash
DULLAHAN_RUN_LOCAL_INFERENCE=1 pytest \
  apps/inference/tests/test_local_gguf_benchmark.py -m local_inference -v
```
