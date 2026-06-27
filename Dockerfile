FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app/apps/cal/src:/app/apps/edl/src:/app/apps/agent-runtime/src:/app/apps/mcp-servers/src:/app/apps/graph-builder/src:/app/packages/shared/src:/app/packages/kg/src:/app/packages/world-state/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY apps ./apps
COPY packages ./packages
COPY configs ./configs
COPY memory ./memory
COPY mcp ./mcp
COPY scripts ./scripts

RUN pip install --no-cache-dir -e .

CMD ["dullahan-agent", "How should CAL and EDL cooperate?", "--max-depth", "1"]
