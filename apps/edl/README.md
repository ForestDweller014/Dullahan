# Expert Dispatch Layer

EDL receives contextualized subqueries, selects an expert from the registered
expert pool, runs an expert instance, and returns an `ExpertResponse`.

The first implementation uses deterministic lexical scoring over expert role
contexts. This is the same seam where embedding attention and softmax routing
will be introduced.
