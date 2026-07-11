---
name: architecture-first-changes
description: Plan and implement architecture-sensitive code changes before editing. Use for cross-module features, API or schema changes, data-flow changes, concurrency changes, new services, dependency-boundary changes, or refactors that affect system structure. Do not trigger for trivial, isolated edits with no architectural impact.
---

# Architecture-First Changes

Make architecture-sensitive changes by understanding the existing system, preserving its invariants, and choosing the smallest coherent design before editing code.

## Core rule

Do not begin broad implementation until you can state:

- The current architecture relevant to the request
- The behavior or constraint that must change
- The boundaries and interfaces affected
- The invariants that must remain true
- The smallest design that satisfies the request
- The validation needed to prove the change works

For a small, reversible change, keep this analysis brief and proceed. For a consequential or ambiguous design choice, surface the choice and its tradeoffs before committing to it.

## 1. Orient to the existing system

1. Read the active `AGENTS.md` instructions and respect the user's scope.
2. If `graphify-out/graph.json` exists, use Graphify before broad source inspection:
   - Use `graphify query "<architectural question>"` to identify relevant subsystems.
   - Use `graphify path "<A>" "<B>"` to trace important connections.
   - Use `graphify explain "<symbol>"` for focused node context.
3. Verify important graph conclusions against current source files.
4. Identify the entry points, data flow, state ownership, external boundaries, and tests relevant to the requested change.
5. Start with a narrow working set of files. Expand it only when imports, calls, types, configuration, or runtime evidence require expansion.

Do not read the repository indiscriminately.

## 2. Establish constraints and invariants

Write a compact internal architecture brief containing:

- User-visible outcome
- In-scope and out-of-scope behavior
- Existing interfaces that callers rely on
- Compatibility requirements
- Security, privacy, latency, consistency, and failure-handling constraints
- Data ownership and lifecycle
- Concurrency or ordering guarantees
- Deployment or migration constraints

Distinguish confirmed facts from assumptions. Verify assumptions that materially affect the design.

## 3. Choose the smallest coherent design

Prefer designs that:

- Reuse existing boundaries and conventions
- Minimize new abstractions and dependencies
- Keep ownership explicit
- Preserve backward compatibility when required
- Make failure modes observable
- Can be tested at the affected boundary
- Can be introduced incrementally
- Are easy to remove or revise if the premise changes

Avoid speculative frameworks, generic layers with only one caller, global state, hidden coupling, and unrelated cleanup.

When multiple designs are plausible, compare them using:

- Correctness
- Complexity
- Coupling and cohesion
- Operational risk
- Migration cost
- Testability
- Reversibility

Choose the conservative option unless the request clearly rewards a larger redesign.

## 4. Define the change contract

Before editing, specify the implementation contract:

- Inputs and outputs
- Public API or schema changes
- Error and retry behavior
- State transitions
- Side effects
- Compatibility behavior
- Observability requirements
- Acceptance tests

For persistent data changes, define migration, rollback, and mixed-version behavior.

For distributed or asynchronous changes, define idempotency, ordering, timeout, cancellation, and partial-failure behavior.

## 5. Plan an implementation sequence

Break the work into dependency-ordered steps that remain reviewable and testable. A typical sequence is:

1. Introduce or adjust contracts and types
2. Add compatibility or migration scaffolding
3. Implement the core behavior
4. Integrate callers and adapters
5. Add tests and observability
6. Remove temporary compatibility code only when safe

Do not create commits unless the user asks. When commits are requested, use the repository's atomic-commit workflow and keep architecture scaffolding, behavior, and cleanup logically separated when they can stand independently.

## 6. Implement within the design

During implementation:

- Stay within the agreed scope
- Follow existing project patterns unless they are the problem being fixed
- Keep changes local to the responsible subsystem
- Update all affected callers and tests
- Avoid drive-by refactors
- Re-evaluate the design if implementation requires unexpected cross-system changes
- Stop and report when a hidden constraint invalidates the chosen design

Do not silently widen scope.

## 7. Validate at multiple levels

Run the narrowest useful checks first, then broader checks as warranted:

- Unit tests for local invariants
- Contract or integration tests for boundaries
- Regression tests for changed behavior
- Type checks, linting, and formatting
- Migration or rollback tests when applicable
- Concurrency, load, or failure-injection tests when relevant

Confirm that unchanged callers retain their expected behavior.

## 8. Completion report

Report concisely:

- Architecture understood
- Design chosen and why
- Boundaries and contracts changed
- Files or subsystems modified
- Validation performed
- Risks, assumptions, and follow-up work
- Any divergence from the original plan

Do not claim architectural guarantees that were not validated.
