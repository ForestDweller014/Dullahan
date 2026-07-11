# Agent Runtime

The Agent Runtime owns the recursive execution loop. An agent receives a query
and context, identifies structured subqueries, asks CAL to augment each
subquery, asks EDL to dispatch each contextualized subquery, and incorporates
expert responses into its final response.

Runtime execution must be governed by depth, breadth, timeout, duplicate-query,
and total-instance limits.
