from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.install_graphify_auto_publish import PATCH_MARKER, patch_hook_text
from scripts.publish_graphify_update import COMMIT_SUBJECT, publish


def run(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def initialize_repo(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "--bare", remote.as_posix()], check=True)
    subprocess.run(["git", "init", "-b", "main", repo.as_posix()], check=True)
    run(repo, "config", "user.name", "Graphify Test")
    run(repo, "config", "user.email", "graphify@example.test")

    graph_dir = repo / "graphify-out"
    graph_dir.mkdir()
    (graph_dir / "graph.json").write_text('{"version": 1}\n', encoding="utf-8")
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    run(repo, "add", ".")
    run(repo, "commit", "-m", "feat: Seed test repository")
    run(repo, "remote", "add", "origin", remote.as_posix())
    run(repo, "push", "-u", "origin", "main")
    return repo, remote


# Verifies that auto-publish commits only Graphify outputs while preserving user work.
def test_publish_commits_graph_outputs_without_user_work(tmp_path: Path) -> None:
    repo, remote = initialize_repo(tmp_path)
    graph_dir = repo / "graphify-out"
    (graph_dir / "graph.json").write_text('{"version": 2}\n', encoding="utf-8")
    (graph_dir / ".graphify_python").write_text("/local/python\n", encoding="utf-8")
    (graph_dir / "cache").mkdir()
    (graph_dir / "cache" / "stat-index.json").write_text("{}\n", encoding="utf-8")
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    run(repo, "add", "source.py")

    assert publish(repo) is True

    assert run(repo, "log", "-1", "--format=%s").stdout.strip() == COMMIT_SUBJECT
    assert run(repo, "diff", "--cached", "--name-only").stdout.strip() == "source.py"
    assert run(repo, "ls-files", "graphify-out/.graphify_python").stdout == ""
    assert run(repo, "ls-files", "graphify-out/cache/stat-index.json").stdout == ""

    remote_graph = subprocess.run(
        ["git", f"--git-dir={remote}", "show", "main:graphify-out/graph.json"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert remote_graph.stdout == '{"version": 2}\n'


# Verifies that hook patch is idempotent.
def test_hook_patch_is_idempotent() -> None:
    hook = """#!/bin/sh
# graphify-hook-start
_src = '''
import os, signal, sys
try:
    try:
        pass
    except Exception:
        pass
except TimeoutError as exc:
    pass
'''
# graphify-hook-end
"""

    patched, changed = patch_hook_text(hook)
    patched_again, changed_again = patch_hook_text(patched)

    assert changed is True
    assert PATCH_MARKER in patched
    assert "import os, signal, subprocess, sys" in patched
    assert changed_again is False
    assert patched_again == patched
