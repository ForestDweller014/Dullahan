# Context Augmentation Layer

The Context Augmentation Layer, or CAL, receives a subquery and the parent
agent context. It retrieves relevant short-term context from the parent context
and long-term context from WorldStateDB, then merges those results into a
bounded context bundle for downstream expert execution.

CAL is not responsible for answering the subquery. Its contract is to return
the context that should make the subquery answerable.
