---
description: Run a local code-review of the current branch against a base branch (no GitHub app needed).
allowed-tools: Bash, Read, Grep, Glob, Task
argument-hint: "[base-branch]  default: dev"
---

# /rdds-review

Run a strict code review of the changes on the current branch against the base
branch (default `dev`). The review is delegated to the `code-reviewer`
subagent.

## Procedure

1. Determine the base branch:
   - If `$1` is provided, use it. Otherwise default to `dev`.
   - If currently *on* the base branch, abort and tell the user to switch to
     a feature branch first.

2. Capture the diff and the file list:

   ```bash
   git fetch origin
   git log origin/<base>..HEAD --oneline
   git diff origin/<base>...HEAD --stat
   git diff origin/<base>...HEAD
   ```

3. Invoke the `code-reviewer` subagent with the Task tool. Pass it:
   - The base branch and current branch.
   - The full diff (or, if very large, the stat plus diffs of changed files
     in chunks).
   - The non-negotiable rules from `CLAUDE.md` §2 and the official metric in §3.
   - Any open issues the user has called out.

4. Relay the subagent's findings back to the user, grouped as:
   - **Blockers** (must fix before merge).
   - **Should fix** (strongly recommended).
   - **Nits** (style / minor).

## Do not

- Do not auto-fix anything. The `code-reviewer` is read-only by design. If the
  user wants fixes applied, invoke `step-implementer` or the relevant
  domain debugger subagent in a separate step.
- Do not mark a PR as ready to merge unless **zero blockers** remain and the
  user explicitly asks for that confirmation.
