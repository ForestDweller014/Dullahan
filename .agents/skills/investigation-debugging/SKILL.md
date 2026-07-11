---
name: investigation-debugging
description: Investigate bugs, test failures, crashes, regressions, performance anomalies, and unexpected behavior using evidence-driven debugging. Use when the root cause is unknown or disputed. Do not trigger for a straightforward edit whose cause and fix are already established.
---

# Investigation and Debugging

Diagnose the actual failure before changing production behavior. Move from observation to reproduction to tested root cause to minimal fix.

## Core rule

Do not patch a symptom based on the first plausible explanation. A fix is justified only when evidence connects the observed failure to a root cause and validation shows the fix resolves it without creating a regression.

## 1. Define the failure precisely

Capture:

- Expected behavior
- Actual behavior
- Exact error, output, or performance symptom
- Environment, version, branch, and configuration
- Reproduction steps
- Frequency and timing
- First known bad state, if available
- User or system impact
- What has already been tried

Separate observed facts from interpretations.

## 2. Reproduce before modifying

Attempt the smallest reliable reproduction.

Prefer, in order:

1. Existing failing test
2. New minimal regression test
3. Focused command or request
4. Controlled local scenario
5. Production evidence when local reproduction is impossible

Record whether the failure is deterministic, intermittent, load-dependent, order-dependent, or environment-specific.

Do not change code merely to make the failure disappear before preserving a reproduction.

## 3. Orient to the failing path

1. Read active repository instructions.
2. If Graphify is available, query the failing symbol, entry point, or subsystem before broad searches.
3. Trace the path from input to failure, including state changes and external boundaries.
4. Inspect the narrowest relevant logs, stack frames, tests, and source ranges.
5. Identify recent changes when regression timing is known.

Use version-control history, blame, or bisect only when they answer a concrete causal question.

## 4. Form ranked hypotheses

Create a short list of plausible causes. For each hypothesis, state:

- Why it could explain the evidence
- What observation would support it
- What observation would falsify it
- The cheapest decisive experiment

Rank hypotheses by explanatory power and test cost, not by intuition alone.

Do not maintain a long speculative list. Remove falsified hypotheses.

## 5. Test one variable at a time

Run focused experiments that distinguish hypotheses:

- Add temporary assertions or structured instrumentation
- Isolate inputs or dependencies
- Compare good and bad versions or configurations
- Reduce concurrency or timing variables
- Mock one external boundary
- Inspect state immediately before and after the failure

Change one meaningful variable per experiment. Record the result before proceeding.

Do not stack several speculative fixes and infer causality from a passing run.

## 6. Identify the root cause

A root-cause statement must explain:

- The triggering condition
- The faulty assumption, state transition, contract, or implementation
- Why the observed symptom follows
- Why the problem appears only under the reported conditions
- Why existing tests or safeguards did not catch it

When evidence supports only a likely cause, label it as likely rather than proven.

## 7. Implement the minimal durable fix

Prefer a fix at the violated invariant or responsible boundary, not at a downstream symptom.

The fix should:

- Address the proven cause
- Preserve intended behavior
- Avoid unrelated refactoring
- Handle the failing condition explicitly
- Improve diagnostics when the failure could recur
- Remain understandable without the investigation history

Remove temporary instrumentation unless it provides durable operational value.

## 8. Add regression coverage

Create or update a test that:

- Fails on the previous behavior
- Passes with the fix
- Reproduces the essential triggering condition
- Asserts the intended contract rather than incidental implementation details

Add broader tests only when the root cause crosses a boundary or exposes a missing class of coverage.

## 9. Validate competing explanations

After the fix passes, confirm that:

- The original reproduction is resolved
- The regression test would fail without the fix
- Relevant neighboring behavior still works
- No error is merely suppressed or retried indefinitely
- Performance or concurrency behavior remains acceptable when relevant

Run targeted checks first, then the appropriate broader suite.

## 10. Failure handling

If reproduction is impossible:

- Preserve available evidence
- State what is missing
- Add safe diagnostics or observability when justified
- Avoid a speculative behavior change presented as a confirmed fix

If the failure is outside the user's scope, stop and report the boundary rather than modifying unrelated systems.

## 11. Completion report

Report:

- Reproduction used
- Root cause and confidence level
- Evidence that confirmed it
- Fix applied
- Regression coverage
- Validation commands and results
- Remaining uncertainty or operational follow-up
