# Agent Runtime

The Agent Runtime owns local hierarchical execution. It receives a root query,
creates structured subqueries, asks CAL to construct context, sends each
contextualized subquery to EDL, and aggregates expert responses into a final
answer.

The runtime can execute in-process or through remote CAL and EDL HTTP adapters.
When `persist_artifacts=True`, it writes aggregate run files and per-query
YAML/Markdown instance directories under `memory/executions`.
