#!/usr/bin/env bash
set -euo pipefail
exec uvicorn server.app:app --host 0.0.0.0 --port "${MANAGER_PORT:-8080}"
