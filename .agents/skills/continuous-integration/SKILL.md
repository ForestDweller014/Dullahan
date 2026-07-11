---
name: continuous-integration
description: Design, audit, implement, debug, or improve Continuous Integration pipelines, including GitHub Actions and equivalent CI systems. Use for build validation, test orchestration, quality gates, security checks, caching, matrices, artifacts, branch checks, flaky tests, and CI reliability. Do not use for production deployment or infrastructure mutation unless the user explicitly requests CD or deployment work.
---

# Continuous Integration Engineering

Build CI that proves each proposed change integrates safely, reproducibly, and reviewably with the repository.

Optimize for:

1. Correctness
2. Fast, useful feedback
3. Reproducibility
4. Security and least privilege
5. Clear diagnostics
6. Maintainability
7. Cost efficiency

Do not treat CI as merely “run tests on push.” Treat it as an executable integration contract for the repository.

## Operating boundaries

- Respect the user’s requested scope.
- Do not deploy to production unless explicitly asked.
- Do not create, rotate, expose, or guess secrets.
- Do not weaken checks merely to make the pipeline pass.
- Do not bypass tests, hooks, branch protections, or security gates.
- Do not silently remove existing platform, runtime, or matrix coverage.
- Do not modify unrelated application code unless a verified defect causes the CI failure and the user requested the fix.
- Preserve user changes and existing workflow behavior unless a change is intentional and documented.
- If asked to commit the work, use the repository’s atomic-commit workflow and do not push unless explicitly requested.

## Phase 1: Orient to the repository

Before editing CI:

1. Read repository instructions such as `AGENTS.md`.
2. Use the project’s architecture map or Graphify representation when available.
3. Inspect the current CI configuration and reusable workflows.
4. Identify:
   - Languages and runtime versions
   - Package managers and lockfiles
   - Build systems
   - Test frameworks
   - Linters, formatters, and type checkers
   - Services required by tests
   - Containers and deployment artifacts
   - Monorepo boundaries
   - Existing scripts in `Makefile`, `package.json`, `pyproject.toml`, `tox.ini`, `noxfile.py`, or equivalent
5. Prefer invoking repository-owned scripts over duplicating shell logic inside CI.
6. Determine the CI provider and its event, permission, caching, matrix, artifact, and concurrency semantics.

Do not redesign the pipeline before understanding how contributors build and test the project locally.

## Phase 2: Define the integration contract

State what the pipeline must prove for a change to be mergeable.

At minimum, consider:

- Dependencies resolve from lockfiles
- The project builds or compiles
- Formatting is valid
- Static analysis passes
- Types are valid
- Unit tests pass
- Required integration tests pass
- Security and secret scans pass
- Generated files are current
- Schemas, migrations, and API contracts remain compatible
- Release artifacts can be produced reproducibly

Classify checks as:

- **Required on every pull request**
- **Required only for affected paths**
- **Required before merge**
- **Required on the default branch**
- **Scheduled or nightly**
- **Release-only**

Avoid running expensive checks indiscriminately when safe change-aware execution is possible.

## Phase 3: Design the pipeline

### Fail fast

Order jobs so cheap, deterministic failures surface first:

1. Configuration validation
2. Formatting and linting
3. Type checking and static analysis
4. Unit tests
5. Build or compilation
6. Integration and contract tests
7. End-to-end tests
8. Security and artifact validation
9. Expensive benchmarks or extended matrices

Run independent checks in parallel.

### Keep jobs cohesive

Each job should have one clear responsibility and produce an actionable result.

Prefer names such as:

- `lint`
- `typecheck`
- `unit-tests`
- `integration-postgres`
- `build-image`
- `security-scan`

Avoid giant jobs that install everything and perform unrelated validation in one opaque script.

## Reproducibility

- Pin language and toolchain versions intentionally.
- Use lockfile-enforced dependency installation.
- Prefer clean installs such as `npm ci`, `pnpm install --frozen-lockfile`, or equivalent.
- Avoid unbounded `latest` dependencies.
- Ensure locale, timezone, and other environment-sensitive settings are deterministic where relevant.
- Use immutable container image references for critical environments when practical.
- Avoid depending on undeclared state from previous jobs or runners.
- Build artifacts once and promote the same artifact rather than rebuilding differently downstream.
- Record the commit SHA and relevant tool versions in produced artifacts.

## Change-aware execution

For monorepos or expensive pipelines:

- Detect affected packages, services, or paths.
- Run only the checks safely attributable to those changes.
- Always run repository-wide checks when shared configuration, lockfiles, build tooling, interfaces, or common libraries change.
- Treat change detection as an optimization, not a correctness shortcut.
- Include a conservative fallback that runs broader validation when impact cannot be determined confidently.

Do not skip validation merely because a path filter was incomplete.

## Caching

Use caching only when it improves speed without compromising correctness.

- Cache package downloads separately from build outputs when possible.
- Key caches using lockfiles, runtime versions, operating system, architecture, and relevant build configuration.
- Restore with controlled fallback keys.
- Never allow cache hits to replace required validation.
- Do not cache secrets or credential-bearing files.
- Ensure corrupted or stale caches fail safely.
- Prefer provider-native or build-system-native caching mechanisms over custom archive logic.

Document what is cached and what invalidates it.

## Test strategy

### Unit tests

- Run fast unit tests on every relevant pull request.
- Use deterministic seeds where randomness exists.
- Surface failed test names and useful traces.

### Integration tests

- Start isolated dependencies such as PostgreSQL, Redis, Kafka, or object stores.
- Use health checks rather than arbitrary sleeps.
- Avoid shared mutable test environments.
- Clean up resources on success, failure, and cancellation.
- Verify migrations and schema setup from a clean state.

### End-to-end tests

- Run a focused, reliable smoke suite on pull requests.
- Move broad or expensive suites to merge, scheduled, or release workflows when appropriate.
- Preserve screenshots, videos, logs, or traces on failure.

### Flaky tests

- Do not hide flakiness with unconditional retries.
- If retries are necessary, report the initial failure and retry outcome.
- Quarantine only with explicit ownership, issue tracking, and an expiration condition.
- Track recurrence and remove quarantines promptly.

### Coverage

- Report coverage where useful.
- Prefer enforcing no meaningful regression over chasing an arbitrary global percentage.
- Require coverage for critical logic and regression fixes.
- Do not make coverage generation obscure real test failures.

## Static quality checks

Consider the repository’s applicable checks:

- Formatting
- Linting
- Type checking
- Compiler warnings
- Dead code
- Dependency cycles
- Architecture boundaries
- Generated-file drift
- Documentation build
- API compatibility
- Database migration validation
- Infrastructure syntax and policy validation

CI should check formatting, not silently rewrite contributor files unless the workflow is explicitly an automated formatting workflow.

## Security

Apply least privilege by default.

- Set workflow or job permissions explicitly.
- Use read-only permissions unless a job demonstrably requires writes.
- Prefer short-lived identity federation such as OIDC over long-lived cloud credentials.
- Never expose secrets to untrusted forked pull-request code.
- Avoid unsafe use of privileged pull-request events.
- Validate all untrusted inputs before interpolation into shell commands.
- Quote shell variables and enable strict shell behavior where appropriate.
- Prevent command injection through branch names, filenames, matrix values, issue text, or pull-request metadata.
- Pin third-party CI actions or reusable components to immutable references; prefer full commit SHAs for security-sensitive workflows.
- Document the human-readable release associated with pinned SHAs.
- Scan dependencies, source, containers, and infrastructure according to project risk.
- Detect committed secrets.
- Generate an SBOM and provenance attestations for release-grade artifacts when relevant.
- Do not upload sensitive logs, environment dumps, or credential-bearing artifacts.

Treat workflow files as production code with access to repository credentials.

## Build and artifact handling

When the repository produces distributable artifacts:

- Build from the exact commit under test.
- Assign deterministic names and versions.
- Record checksums.
- Upload only required outputs.
- Set explicit retention periods.
- Avoid including source secrets, caches, local configuration, or unnecessary repository contents.
- Verify packages or images can be consumed.
- Prefer build-once, test-the-artifact semantics.
- Preserve test reports, coverage, and diagnostics separately from release artifacts.

## Matrices and platform coverage

Use matrices when compatibility across versions or platforms is a real requirement.

- Identify one primary configuration for fast feedback.
- Add supported secondary runtimes, operating systems, architectures, databases, or feature flags.
- Avoid a Cartesian explosion with little risk-reduction value.
- Mark experimental combinations clearly.
- Decide intentionally whether one matrix failure should cancel remaining entries.
- Include the oldest and newest supported runtime where compatibility matters.
- Keep the matrix aligned with documented support policy.

Do not silently drop a supported platform because it is inconvenient.

## Concurrency and cancellation

- Cancel superseded runs for the same pull request when safe.
- Do not cancel release, migration, or stateful workflows where cancellation can corrupt state.
- Use bounded timeouts for jobs and steps.
- Prevent duplicate deployments or publishing jobs.
- Use explicit concurrency groups based on stable identifiers.
- Ensure cleanup runs even when a job fails or is cancelled.

## Diagnostics and developer experience

A failed pipeline should answer:

1. What failed?
2. Where did it fail?
3. Why is it likely failing?
4. How can it be reproduced locally?
5. What logs or artifacts contain more detail?

- Use clear job and step names.
- Preserve structured test reports.
- Add concise workflow summaries.
- Upload diagnostics only on failure when appropriate.
- Avoid drowning the useful error in verbose installation logs.
- Keep commands locally reproducible.
- Print relevant versions and configuration without exposing secrets.

Fast but opaque CI is not good CI.

## Database and contract safety

When applicable, validate:

- Migrations apply from a clean database.
- Upgrade paths from supported previous schemas.
- Rollback behavior when rollback is supported.
- ORM models and generated schema state are synchronized.
- OpenAPI, GraphQL, protobuf, and event schemas are valid.
- Breaking API or schema changes are detected.
- Consumer-driven contract tests pass.
- Generated clients or bindings are current.

Flag intentional breaking changes clearly instead of weakening compatibility checks.

## Performance checks

Use performance gates only with controlled methodology.

- Establish a baseline and noise tolerance.
- Use dedicated or stable runners when possible.
- Compare like-for-like configurations.
- Report distributions, not only one timing.
- Separate blocking regressions from informational benchmarks.
- Avoid blocking every pull request on highly noisy microbenchmarks.

## Workflow implementation standards

- Reuse repository scripts and reusable workflows.
- Keep YAML declarative and move complex logic into versioned scripts with tests.
- Use strict shell settings where compatible.
- Quote variables.
- Add timeouts.
- Minimize duplicated setup.
- Make dependencies between jobs explicit.
- Keep comments focused on non-obvious rationale.
- Validate workflow syntax before concluding.
- Run a local equivalent of each new or changed command when possible.

For GitHub Actions specifically:

- Set top-level or job-level `permissions`.
- Use `concurrency` for superseded pull-request runs where safe.
- Use `pull_request` for untrusted code validation.
- Treat `pull_request_target` as privileged and avoid checking out untrusted code within it.
- Pin external actions immutably.
- Avoid writing secrets to `$GITHUB_ENV`, logs, or artifacts.
- Use environment protection for privileged release jobs.
- Prefer reusable workflows for repeated organization-wide logic.

## Modifying an existing pipeline

Before changing it:

1. Identify the exact failure, bottleneck, gap, or risk.
2. Preserve unrelated behavior.
3. Determine whether the problem belongs in repository code, a local script, workflow orchestration, runner configuration, dependency configuration, or test infrastructure.
4. Make the smallest coherent change.
5. Validate affected and adjacent paths.
6. Compare expected behavior before and after.
7. Report any intentionally changed coverage or trigger semantics.

Do not clean up unrelated CI while fixing one failure unless the user asks for a broader refactor.

## Debugging CI failures

Follow an evidence-first process:

1. Identify the first causal failure, not the final cascade.
2. Record the failing job, step, command, exit code, and environment.
3. Compare local and CI environments.
4. Check runtime, dependency, platform, permission, path, shell, network, and timing differences.
5. Form ranked hypotheses.
6. Test one hypothesis at a time.
7. Reproduce locally or in an equivalent container when possible.
8. Fix the root cause.
9. Add regression protection if feasible.
10. Do not disable or weaken the failing check without explicit justification.

Classify failures as:

- Product defect
- Test defect
- Workflow defect
- Environment defect
- Dependency or external-service failure
- Flake
- Permission or secret failure

## Validation before completion

Run the strongest available validation that is safe and relevant:

- Workflow syntax validation
- YAML formatting or linting
- Provider-specific workflow linting
- Repository lint, typecheck, tests, and build
- Shell linting for embedded scripts
- Security review of permissions and secret exposure
- Dry runs or local action runners where reliable
- Inspection of the complete diff

If a check cannot run locally, state exactly what remains to be validated by the remote CI system.

## Completion report

Report:

- What pipeline behavior changed
- Why it changed
- Files modified
- Events and paths affected
- Required checks added, removed, or altered
- Permissions and secret usage
- Caching behavior
- Matrix coverage
- Commands and checks run
- Results
- Expected remote-only validation
- Remaining risks or follow-up work

## Definition of done

CI work is complete only when:

- The pipeline expresses a clear integration contract.
- The configuration is valid.
- Required checks are deterministic and appropriately scoped.
- Fast checks provide early feedback.
- Expensive checks are parallelized or scheduled appropriately.
- Dependencies and environments are reproducible.
- Permissions follow least privilege.
- Untrusted code cannot access privileged secrets.
- Caches cannot create false successes.
- Failures produce actionable diagnostics.
- Relevant local validation passes.
- Any remote-only verification is explicitly identified.
- Existing supported behavior is preserved unless intentionally changed.
