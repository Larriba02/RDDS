#!/usr/bin/env python3
"""
RDDS — block-push-without-review hook
=====================================

PreToolUse hook on Bash. Bails out of `git push` if there are unreviewed
commits between the local HEAD and the remote tracking branch.

Behavior:
- Reads the tool-call JSON from stdin.
- If the tool is not Bash, exits 0 (allow).
- If the command is not a `git push`, exits 0 (allow).
- Otherwise, runs `git log <remote>/<branch>..HEAD --oneline`.
  - If empty, exits 0 (nothing to push, or fully fast-forwarded already).
  - If non-empty, exits 2 with an explanatory message on stderr. Claude
    Code will surface that message to the agent and block the push.

The hook never auto-runs a review or modifies files. Its job is only to
block the push and ask Claude to invoke the `code-reviewer` subagent
first. After review, the user can re-invoke `git push` and the hook will
not fire again on the same commits within the same session if a review
marker is present (see ".claude/.review-marker" below).

Review marker
-------------
The hook honors a soft "I have already reviewed this commit range" marker
at `.claude/.review-marker`. The file should contain a single line: the
SHA of the highest commit that has been reviewed in this session. When
present and matching `git rev-parse HEAD`, the hook lets `git push` go
through. The marker is gitignored.

Drop the marker by deleting the file or running `git commit` again
(commits invalidate the marker because HEAD changes).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


REVIEW_MARKER = Path(".claude") / ".review-marker"
GIT_PUSH_RE = re.compile(r"(?:^|[\s;&|])git\s+push\b")


def _read_tool_call() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _is_git_push(tool_name: str, tool_input: dict) -> bool:
    if tool_name != "Bash":
        return False
    command = tool_input.get("command", "")
    return bool(GIT_PUSH_RE.search(command))


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10
        )
        return result.returncode, result.stdout, result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, "", str(exc)


def _current_branch() -> str | None:
    rc, out, _ = _run(["git", "branch", "--show-current"])
    if rc != 0 or not out.strip():
        return None
    return out.strip()


def _head_sha() -> str | None:
    rc, out, _ = _run(["git", "rev-parse", "HEAD"])
    if rc != 0 or not out.strip():
        return None
    return out.strip()


def _unreviewed_commits(branch: str) -> list[str]:
    """Return one-line summaries of commits in origin/<branch>..HEAD."""
    # Try the upstream tracking branch first.
    rc, out, _ = _run(
        ["git", "log", f"origin/{branch}..HEAD", "--oneline"]
    )
    if rc != 0:
        # No upstream yet (first push of the branch). Treat all reachable
        # commits not on origin's default branch as unreviewed.
        rc2, out2, _ = _run(
            ["git", "log", "origin/HEAD..HEAD", "--oneline"]
        )
        if rc2 != 0:
            return []
        out = out2
    return [line for line in out.splitlines() if line.strip()]


def _review_marker_matches_head() -> bool:
    if not REVIEW_MARKER.exists():
        return False
    try:
        marker_sha = REVIEW_MARKER.read_text().strip()
    except OSError:
        return False
    head = _head_sha()
    return bool(head) and marker_sha == head


def main() -> int:
    payload = _read_tool_call()
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})

    if not _is_git_push(tool_name, tool_input):
        return 0

    if _review_marker_matches_head():
        # User explicitly marked the current HEAD as reviewed.
        return 0

    branch = _current_branch()
    if not branch:
        # Detached HEAD or non-git directory; let git itself handle it.
        return 0

    pending = _unreviewed_commits(branch)
    if not pending:
        return 0

    msg = [
        "",
        "[block-push-without-review] Push blocked.",
        "",
        f"There are {len(pending)} commit(s) on '{branch}' that have not "
        "been reviewed in this session:",
        "",
        *[f"  {line}" for line in pending],
        "",
        "Before pushing, invoke the `code-reviewer` subagent on the diff:",
        "",
        f"    git diff origin/{branch}...HEAD",
        "",
        "If the review passes (no blockers), record approval by writing the",
        "current HEAD SHA to .claude/.review-marker, then retry the push.",
        "",
        "    git rev-parse HEAD > .claude/.review-marker",
        "    git push",
        "",
    ]
    print("\n".join(msg), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
