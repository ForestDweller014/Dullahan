---
name: atomic-git-commits
description: Review workspace changes and create clean, atomic Git commits. Use whenever the user asks to commit changes, organize commits, stage files, prepare commit history, or split work into logical commits.
---

# Atomic Git Commit Workflow

Review the current workspace changes and create professional, logically separated Git commits.

## Scope

Respect any path or file scope provided by the user.

Do not stage, modify, revert, or commit files outside that scope.

If no scope is provided, inspect all workspace changes, but do not assume every change should be committed.

## Commit strategy

Before committing:

1. Inspect the working tree and all relevant diffs.
2. Determine the distinct logical changes represented.
3. Divide unrelated changes into separate commits.
4. Preserve dependency order between commits.
5. Keep each commit independently reviewable and revertible.
6. Keep directly related implementation and tests together.
7. Do not split a change so aggressively that a commit becomes incomplete or broken.
8. Use partial staging when one file contains changes belonging to multiple commits.

Normally split:

- A feature and an unrelated bug fix
- A refactor and a behavior change
- Unrelated fixes affecting separate components
- Formatting changes and functional changes
- Documentation unrelated to the implementation change

Normally keep together:

- A bug fix and its regression test
- A feature and its directly related tests
- A rename and all required reference updates
- A change and the minimal supporting types or configuration it requires

## Commit messages

Use Conventional Commits types such as:

- `feat`
- `fix`
- `refactor`
- `test`
- `docs`
- `chore`
- `style`
- `perf`
- `build`
- `ci`

For every commit:

- Format the subject as `<type>: <Summary>`
- Keep the subject under 50 characters
- Capitalize the first word after the type
- Use imperative mood
- Do not end the subject with a period
- Add a blank line after the subject
- Add a bulleted body explaining what changed and why
- Focus on intent and effect, not low-level implementation mechanics
- Avoid vague subjects such as `Update files` or `Fix stuff`

Example:

```text
fix: Prevent duplicate modal submissions

- Block repeated submissions while a request is pending
- Avoid duplicate records caused by rapid user input
