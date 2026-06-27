# Expert Dispatch Layer

The Expert Dispatch Layer, or EDL, receives a contextualized subquery and routes
it to an expert instance. Routing is based on similarity between the subquery
embedding and expert role context embeddings.

EDL may run many expert instances concurrently. The same expert profile can be
used by multiple concurrent instances when several subqueries map to that
expertise.
