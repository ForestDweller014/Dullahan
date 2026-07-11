#!/usr/bin/env python3
"""Commit and push generated Graphify artifacts without touching user work."""

from __future__ import annotations

import argparse
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


COMMIT_SUBJECT = "chore: Refresh Graphify snapshot"
COMMIT_BODY = (
    "- Update the codebase graph after the preceding source commit\n"
    "- Keep generated navigation artifacts synchronized on the remote branch"
)

# New files are limited to durable graph outputs. Local interpreter paths,
# query memory, reflections, backups, and temporary state remain untracked.
NEW_GENERATED_PATHS = (
    "graphify-out/.graphify_analysis.json",
    "graphify-out/.graphify_labels.json",
    "graphify-out/.graphify_labels.json.sig",
    "graphify-out/GRAPH_REPORT.md",
    "graphify-out/cache/ast",
    "graphify-out/cache/stat-index.json",
    "graphify-out/cost.json",
    "graphify-out/graph.html",
    "graphify-out/graph.json",
    "graphify-out/manifest.json",
)


class PublishError(RuntimeError):
    """Raised when Graphify artifacts cannot be published safely."""


def _git(
    repo_root: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _nul_paths(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [path for path in result.stdout.split("\0") if path]


def _staged_graphify_paths(repo_root: Path) -> list[str]:
    return _nul_paths(
        _git(
            repo_root,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--",
            "graphify-out",
        )
    )


def _resolve_repo_root(path: Path) -> Path:
    result = _git(path, "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve()


def _require_push_target(repo_root: Path) -> tuple[str, str]:
    branch = _git(repo_root, "symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    if branch.returncode != 0:
        raise PublishError("refusing to publish Graphify output from detached HEAD")

    upstream = _git(
        repo_root,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    )
    if upstream.returncode != 0:
        raise PublishError(
            f"branch {branch.stdout.strip()!r} has no upstream; configure one before auto-publish"
        )
    return branch.stdout.strip(), upstream.stdout.strip()


@contextmanager
def _publish_lock(repo_root: Path) -> Iterator[None]:
    git_dir_raw = _git(repo_root, "rev-parse", "--git-dir").stdout.strip()
    git_dir = Path(git_dir_raw)
    if not git_dir.is_absolute():
        git_dir = repo_root / git_dir
    lock_path = git_dir / "graphify-auto-publish.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise PublishError("another Graphify publish is already running") from exc

    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _stage_generated_outputs(repo_root: Path) -> list[str]:
    pre_staged = _staged_graphify_paths(repo_root)
    if pre_staged:
        joined = ", ".join(pre_staged)
        raise PublishError(f"Graphify paths were already staged; refusing to overwrite: {joined}")

    # Capture changes to outputs already tracked by the repository, including
    # deletions, without staging anything elsewhere in the workspace.
    _git(repo_root, "add", "-u", "--", "graphify-out")

    durable_new_paths = [path for path in NEW_GENERATED_PATHS if (repo_root / path).exists()]
    if durable_new_paths:
        _git(repo_root, "add", "-A", "--", *durable_new_paths)

    return _staged_graphify_paths(repo_root)


def _commit_outputs(repo_root: Path, paths: Sequence[str]) -> None:
    env = os.environ.copy()
    env["GRAPHIFY_SKIP_HOOK"] = "1"
    _git(
        repo_root,
        "commit",
        "--only",
        "-m",
        COMMIT_SUBJECT,
        "-m",
        COMMIT_BODY,
        "--",
        *paths,
        env=env,
    )


def publish(repo_root: Path) -> bool:
    """Commit generated artifacts when needed, then push the current branch.

    Returns ``True`` when a Graphify commit was created. A successful call
    always attempts a normal, non-force push, even when no artifact changed.
    """

    repo_root = _resolve_repo_root(repo_root)
    branch, upstream = _require_push_target(repo_root)

    with _publish_lock(repo_root):
        paths = _stage_generated_outputs(repo_root)
        committed = bool(paths)
        if committed:
            _commit_outputs(repo_root, paths)

        push = _git(repo_root, "push")
        action = "committed and pushed" if committed else "pushed"
        print(f"[graphify publish] {action} {branch} -> {upstream}")
        if push.stdout.strip():
            print(push.stdout.strip())
        return committed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Commit durable Graphify outputs and push the current branch."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        publish(args.repo_root)
    except (PublishError, subprocess.CalledProcessError) as exc:
        print(f"[graphify publish] failed: {exc}")
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.strip())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
