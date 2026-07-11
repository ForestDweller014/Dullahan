#!/usr/bin/env python3
"""Extend Graphify's managed post-commit hook with Dullahan auto-publish."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


PATCH_MARKER = "# dullahan-graphify-auto-publish-start"
GRAPHIFY_MARKER = "# graphify-hook-start"
IMPORT_ANCHOR = "import os, signal, sys"
CALL_ANCHOR = "    except Exception:\n        pass\nexcept TimeoutError as exc:"
PUBLISH_BLOCK = """    except Exception:
        pass

    # dullahan-graphify-auto-publish-start
    _publisher = _root / 'scripts' / 'publish_graphify_update.py'
    if _publisher.is_file():
        subprocess.run([sys.executable, str(_publisher)], cwd=_root, check=True)
    # dullahan-graphify-auto-publish-end
except TimeoutError as exc:"""


class InstallError(RuntimeError):
    """Raised when the installed Graphify hook has an unexpected shape."""


def patch_hook_text(text: str) -> tuple[str, bool]:
    if PATCH_MARKER in text:
        return text, False
    if GRAPHIFY_MARKER not in text:
        raise InstallError("Graphify post-commit hook is not installed")
    if text.count(IMPORT_ANCHOR) != 1:
        raise InstallError("Graphify hook import anchor was not found exactly once")
    if text.count(CALL_ANCHOR) != 1:
        raise InstallError("Graphify hook publish anchor was not found exactly once")

    patched = text.replace(IMPORT_ANCHOR, "import os, signal, subprocess, sys", 1)
    patched = patched.replace(CALL_ANCHOR, PUBLISH_BLOCK, 1)
    return patched, True


def _hook_path(repo_root: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-path", "hooks/post-commit"],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    path = Path(result.stdout.strip())
    return path if path.is_absolute() else repo_root / path


def install(repo_root: Path) -> bool:
    repo_root = repo_root.resolve()
    hook_path = _hook_path(repo_root)
    if not hook_path.exists():
        raise InstallError("post-commit hook is missing; run `graphify hook install` first")

    patched, changed = patch_hook_text(hook_path.read_text(encoding="utf-8"))
    if changed:
        hook_path.write_text(patched, encoding="utf-8")
        hook_path.chmod(hook_path.stat().st_mode | 0o111)
        print(f"Installed Graphify auto-publish extension in {hook_path}")
    else:
        print(f"Graphify auto-publish extension already installed in {hook_path}")
    return changed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch the installed Graphify hook to publish generated outputs."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        install(args.repo_root)
    except (InstallError, subprocess.CalledProcessError, OSError) as exc:
        print(f"Failed to install Graphify auto-publish: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
