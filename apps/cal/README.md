# Context Augmentation Layer

CAL receives a subquery and parent context, retrieves relevant short-term and
long-term context, and returns a merged `ContextBundle`.

The first implementation uses deterministic lexical retrieval over parent
context documents and local knowledge graph Markdown documents. Vector retrieval
can replace the retriever implementation without changing the public contract.
