# Context Augmentation Layer

CAL receives a subquery and parent context, retrieves relevant short-term and
long-term context, and returns a merged `ContextBundle`.

Parent-context reranking remains lexical, while WorldStateDB retrieval uses the
semantic embedding model served by `dullahan-inference`. CAL packs the merged
documents using the generation model's native tokenizer count from `/tokenize`.
It does not estimate tokens from words or characters.

CAL can also retrieve long-term memory from PostgreSQL + pgvector by setting
`WORLD_STATE_BACKEND=postgres` and `WORLD_STATE_POSTGRES_DSN`. The pgvector
column dimension must match `DULLAHAN_EMBEDDING_DIMENSIONS` (1,024 by default).
