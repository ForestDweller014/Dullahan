# WorldStateDB

WorldStateDB stores long-term world-state memory for the system. It is queried
by CAL during context construction and should contain durable facts, graph
documents, snapshots, and external state that can be retrieved by semantic
similarity.

In the local prototype this can be implemented as a local vector index backed by
filesystem manifests.
