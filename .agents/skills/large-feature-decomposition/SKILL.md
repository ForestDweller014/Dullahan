---
name: large-feature-decomposition
description: Decompose a large or cross-cutting feature into dependency-ordered, testable implementation chunks with bounded context. Use for multi-module work, multi-stage migrations, features likely to require several commits or pull requests, or requests too broad for one safe implementation pass. Do not trigger for a small localized change.
---

# Large-Feature Decomposition

Turn a broad feature request into a sequence of independently understandable, testable, and reviewable increments. Reduce risk and context pressure by working one bounded slice at a time.

## Core rule

Do not implement a large feature as one undifferentiated change. First define the target outcome, map the affected architecture, identify dependencies and risks, and split the work into vertical slices that produce verifiable progress.

## 1. Frame the feature

Establish:

- User or system outcome
- Primary use cases
- Non-goals
- Acceptance criteria
- Compatibility requirements
- Security, privacy, reliability, latency, and cost constraints
- Rollout, migration, and rollback requirements
- Explicitly requested delivery scope

Distinguish requirements from assumptions and optional enhancements.

## 2. Map the affected system

1. Read active repository instructions.
2. If Graphify is available, use scoped graph queries to identify relevant communities, entry points, dependencies, and cross-module paths.
3. Verify key graph results against source.
4. Identify affected boundaries:
   - APIs and contracts
   - Data models and storage
   - Services and modules
   - Background jobs or event flows
   - User interfaces
   - Configuration and deployment
   - Tests and observability

Avoid repository-wide reading. Build only the architecture map needed for decomposition.

## 3. Identify workstreams and dependencies

List the necessary capabilities, then model dependencies between them.

Typical categories include:

- Contracts and types
- Persistence or migration
- Core domain behavior
- Integration adapters
- User-facing behavior
- Authorization and validation
- Observability and operations
- Tests and documentation

Mark each item as:

- Prerequisite
- Independently deliverable
- Parallelizable
- Migration-sensitive
- High-risk or uncertain

Resolve high-risk unknowns with a small investigation or spike before committing to a large design.

## 4. Prefer vertical slices

A good chunk crosses the minimum necessary layers to produce a testable behavior or prove a design assumption.

Prefer:

- One narrow end-to-end path
- One API operation with validation, behavior, and tests
- One migration step with compatibility behavior
- One integration behind an existing interface
- One feature flag or rollout stage

Avoid decomposing solely into horizontal layers such as “all database work,” then “all backend work,” then “all UI work” when that delays validation until the end.

Use horizontal prerequisite chunks only when they are independently testable and genuinely unblock several slices.

## 5. Define each chunk

For every chunk, specify:

- Goal
- User-visible or system-visible result
- In-scope files or subsystems
- Dependencies and prerequisites
- Interfaces created or changed
- Acceptance criteria
- Tests and validation
- Migration or rollback behavior
- Risks and open questions
- Explicit non-goals

Size chunks so one agent can complete each with a bounded working set and a clear stopping point. A chunk should normally fit one reviewable pull request and one or a few coherent commits.

Do not force arbitrary commit counts. Commit structure should follow logical change boundaries.

## 6. Sequence for risk reduction

Order chunks to:

1. Resolve architectural uncertainty early
2. Establish stable contracts before dependent implementations
3. Preserve compatibility during transitions
4. Produce a thin working path as early as possible
5. Add breadth after the core path is validated
6. Defer cleanup until dependents have migrated

Use feature flags, adapters, dual reads or writes, or compatibility layers when they materially reduce rollout risk. Include explicit removal criteria for temporary mechanisms.

## 7. Create a context packet per chunk

Before implementing a chunk, prepare a compact packet containing only:

- Chunk goal and acceptance criteria
- Relevant architecture summary
- Required interfaces and invariants
- Known files and symbols
- Dependencies completed
- Decisions already made
- Tests to run
- Out-of-scope work

Do not carry the full exploration history into every chunk. After completing a chunk, replace details with a concise outcome summary and the new system state.

## 8. Execute one chunk at a time

For each chunk:

1. Reconfirm its prerequisites
2. Orient to its narrow code path
3. Implement only its defined scope
4. Run targeted validation
5. Review the diff for leakage into future chunks
6. Record decisions and newly discovered dependencies
7. Update the remaining sequence when evidence requires it

Do not silently pull future chunks into the current one.

Do not create commits or push unless the user requests it. When commits are requested, use the repository's atomic-commit workflow.

## 9. Completion and handoff

After each chunk, report:

- What is now working
- Interfaces or data state introduced
- Validation completed
- Remaining chunks
- New risks or changed assumptions
- The exact next chunk and why it comes next

At feature completion, verify the end-to-end acceptance criteria, rollout path, rollback path, observability, documentation, and cleanup obligations.

## 10. Decomposition output format

Present the plan as:

1. Feature summary
2. Architecture impact
3. Key decisions and assumptions
4. Dependency graph or ordered milestones
5. Chunk specifications
6. Validation strategy
7. Rollout and rollback plan
8. Risks and unresolved questions
9. Recommended first chunk

Keep the plan actionable. Do not substitute a vague checklist for concrete boundaries and acceptance criteria.
