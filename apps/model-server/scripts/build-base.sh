#!/usr/bin/env bash
set -euo pipefail

variant="${1:-}"
VLLM_VERSION="${VLLM_VERSION:-0.24.0}"

case "$variant" in
  cpu)
    PLATFORM="${PLATFORM:-linux/arm64}"
    WORKDIR="${WORKDIR:-.build/vllm-${VLLM_VERSION}}"
    IMAGE="${IMAGE:-dullahan/vllm-cpu-base:${VLLM_VERSION}}"
    rm -rf "$WORKDIR"
    mkdir -p "$(dirname "$WORKDIR")"
    git clone --depth 1 --branch "v${VLLM_VERSION}" \
      https://github.com/vllm-project/vllm.git "$WORKDIR"
    docker buildx build \
      --load \
      --platform "$PLATFORM" \
      --file "$WORKDIR/docker/Dockerfile.cpu" \
      --target vllm-openai \
      --tag "$IMAGE" \
      "$WORKDIR"
    ;;
  cuda)
    upstream="vllm/vllm-openai:v${VLLM_VERSION}"
    image="${IMAGE:-dullahan/vllm-cuda-base:${VLLM_VERSION}}"
    docker pull "$upstream"
    docker tag "$upstream" "$image"
    ;;
  *)
    echo "usage: $0 cpu|cuda" >&2
    exit 2
    ;;
esac
