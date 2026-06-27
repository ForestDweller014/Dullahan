# Expert Pool

The Expert Pool is the set of specialized SLM-backed expert profiles available
to EDL. Each expert is associated with a knowledge graph cluster and has a role
context derived from that cluster's documents.

An expert profile is not a single running process. It is a reusable definition
that can be instantiated many times for concurrent subquery execution.
