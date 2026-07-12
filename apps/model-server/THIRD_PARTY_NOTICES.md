# Third-party licenses

Dullahan is licensed under Apache-2.0. It interoperates with, installs, or
downloads third-party components that remain under their own licenses.

## Inference runtimes

- [vLLM](https://github.com/vllm-project/vllm), including the pinned v0.24.0
  CPU/CUDA base runtime: Apache-2.0.
- [vLLM GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin):
  Apache-2.0.
- [Ollama](https://github.com/ollama/ollama): MIT. Ollama is invoked as an
  external runtime and is not redistributed in this repository.

The upstream base images and installed Python distributions retain their own
license and notice files. Those upstream files are authoritative for the exact
versions assembled into a container.

## Default model references

Dullahan does not include model weights in its source or container images.
Models are fetched or uploaded separately and are not relicensed by Dullahan.
The official default repositories currently declare Apache-2.0:

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4)
- [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)
- [Qwen/Qwen2.5-7B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ)

Operators may configure other models and LoRA adapters. They are responsible
for reviewing and complying with each artifact's license, acceptable-use
terms, attribution requirements, and redistribution restrictions. Full model
package exports preserve files supplied with the package; LoRA-only exports
refer to the base model by name and do not redistribute its weights.

## Python dependencies

Python dependencies are installed from their upstream distributions and are
not relicensed by Dullahan. Their package metadata and included license files
are authoritative. Notable direct dependencies include FastAPI, Uvicorn,
HTTPX, Hugging Face Hub, python-multipart, Pydantic, PyYAML, psycopg, and
Graphify.
