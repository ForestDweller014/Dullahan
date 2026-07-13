# Expert Dispatch Layer

EDL receives contextualized subqueries, selects an expert from the registered
expert pool, runs an expert instance, and returns an `ExpertResponse`.

Routing uses semantic Qwen3 embeddings from the shared inference endpoint and
softmax scoring over expert role contexts. Expert execution always uses the
configured OpenAI-compatible HTTP inference endpoint. Unit tests inject
explicit embedding and completion fakes; the opt-in inference suite covers the
real CPU models.
