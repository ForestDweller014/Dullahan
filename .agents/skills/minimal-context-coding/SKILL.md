---
name: minimal-context-coding
description: Solve codebase tasks with a deliberately small, evidence-driven context window. Use for unfamiliar or large repositories, long-running tasks, Graphify-enabled projects, or whenever broad file reading and context bloat are risks. Do not use as a reason to skip necessary verification.
---

# Minimal-Context Coding

Complete the task using the smallest sufficient working set of code, evidence, and instructions. Optimize for relevance, not for reading volume.

## Core rule

Every file, symbol, log, command, or dependency added to the working context must answer a concrete question required to complete the task.

Do not confuse minimal context with shallow understanding. Read less, but verify what matters.

## 1. Establish the task boundary

Extract a compact task frame:

- Requested outcome
- Explicit constraints
- Allowed paths or components
- Completion criteria
- Operations requiring user approval
- Known non-goals

Do not explore outside the boundary unless evidence shows a necessary dependency.

## 2. Orient before reading files

1. Read active repository instructions.
2. If `graphify-out/graph.json` exists, use Graphify first:
   - `graphify query "<focused question>"`
   - `graphify explain "<symbol>"`
   - `graphify path "<A>" "<B>"`
3. Use symbol search, filename search, call/reference search, tests, and targeted line ranges before opening whole files.
4. Verify graph or search results against source before changing behavior.

Do not start by reading `GRAPH_REPORT.md` end to end unless the task requires broad architecture. Prefer a scoped graph query.

## 3. Build a bounded working set

Begin with no more than the smallest plausible set of symbols and files. As a default, orient with at most five source files before deciding whether expansion is necessary.

For each candidate file, identify why it is needed:

- Defines the changed behavior
- Calls the changed behavior
- Defines an affected contract or type
- Contains the relevant test
- Configures or deploys the affected path

Skip files that are merely adjacent, similarly named, or potentially interesting.

## 4. Expand only through evidence

Expand the working set only when one of these occurs:

- An unresolved import, call, type, or configuration reference
- A test failure pointing elsewhere
- Runtime evidence contradicting the current model
- A public interface with additional callers
- A state, security, migration, or concurrency boundary that must be verified

State the reason for expansion before broadening the search.

## 5. Read surgically

Prefer:

- Relevant functions, classes, and line ranges
- Diffs instead of complete changed files
- Interface definitions before implementations
- Focused tests before entire test suites
- Structured logs and stack traces over speculative browsing

Avoid:

- Re-reading unchanged files
- Dumping large generated files
- Loading dependency lockfiles unless dependency resolution is relevant
- Reading vendored code, build output, caches, media, or binaries
- Inspecting every caller when a query can identify the affected set

## 6. Maintain a compact working summary

Keep a short, updateable summary containing only:

- Current understanding
- Confirmed relevant symbols and files
- Important invariants
- Decisions made
- Remaining uncertainties
- Next validation step

Replace obsolete assumptions rather than accumulating them. After each completed milestone, compress the result into a few durable facts and discard incidental exploration details.

Do not expose hidden chain-of-thought. Report conclusions, evidence, and concise rationale only.

## 7. Use bounded investigations

When parallel agents or subagents are available, delegate only narrow questions with explicit outputs, such as:

- Identify all callers of one interface
- Reproduce one failing test
- Compare two implementation options
- Inspect one subsystem for a stated invariant

Require compact findings with file and symbol references. Do not delegate overlapping repository-wide exploration.

## 8. Implement narrowly

- Change only what is required for the requested outcome
- Reuse existing abstractions where appropriate
- Avoid unrelated formatting and cleanup
- Keep tests close to the changed behavior
- Re-run targeted checks after each meaningful edit
- Inspect the final diff for accidental scope expansion

When the task reveals a larger design problem, report it separately rather than absorbing it into the current change without permission.

## 9. Stop conditions

Stop exploring when:

- The behavior and responsible code path are identified
- Relevant contracts and callers are known
- The planned edit is supported by evidence
- Required tests can be named
- Remaining uncertainty is immaterial or explicitly reported

Do not continue reading merely to increase confidence after the decision-relevant facts are established.

## 10. Completion report

Report:

- Outcome achieved
- Minimal working set used
- Key evidence consulted
- Files changed
- Validation run
- Any assumptions or unverified edges
- Any broader issue intentionally left out of scope
