# Context Augmentation Layer

CAL receives a subquery and parent context, retrieves relevant short-term and
long-term context, and returns a merged `ContextBundle`.

Parent-context reranking remains lexical, while WorldStateDB retrieval uses the
semantic embedding model selected by the shared inference provider. CAL packs
the merged documents using an exact input-token count from Dullahan's custom
`/tokenize` endpoint. The tokenizer URL is independent from the generation and
embedding provider, so hosted OpenAI mode still calls the Dullahan endpoint at
`DULLAHAN_TOKENIZER_BASE_URL`. It never sends the custom route to OpenAI and
never estimates tokens from words or characters.

CAL can also retrieve long-term memory from PostgreSQL + pgvector by setting
`WORLD_STATE_BACKEND=postgres` and `WORLD_STATE_POSTGRES_DSN`. The pgvector
column dimension must match `DULLAHAN_EMBEDDING_DIMENSIONS` (1,024 by default).
