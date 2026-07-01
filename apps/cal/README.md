# Context Augmentation Layer

CAL receives a subquery and parent context, retrieves relevant short-term and
long-term context, and returns a merged `ContextBundle`.

The default implementation uses deterministic lexical retrieval over parent
context documents and local WorldStateDB graph documents. CAL can also retrieve
long-term memory from PostgreSQL + pgvector by setting
`WORLD_STATE_BACKEND=postgres` and `WORLD_STATE_POSTGRES_DSN`.
