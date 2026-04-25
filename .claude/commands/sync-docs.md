---
description: Check that the documentation matches the current state of the code.
allowed-tools: Bash, Read, Grep, Glob, Edit, Task
argument-hint: ""
---

# /sync-docs

Detect drift between the code and the project documentation, and patch it.
The work is delegated to the `doc-syncer` subagent.

## Procedure

1. Identify recently changed code paths:

   ```bash
   git log --since="2 weeks ago" --name-only --pretty=format: -- src/ | sort -u
   ```

2. Invoke the `doc-syncer` subagent with the Task tool. Pass it:
   - The changed code files.
   - The four authoritative documentation surfaces:
     - `README.md`
     - `DOCUMENTATION/RDDS_Dev_Steps.md`
     - `DOCUMENTATION/RDDS_Pipeline.md`
     - `DOCUMENTATION/IN DETAIL/*.md`

3. The subagent must produce, for each drift it finds:
   - **Where** (file + section).
   - **What** (one-line summary of the mismatch).
   - **Patch** (the exact edit it would apply, *not yet applied*).

4. Show the user the proposed patches as a list. Apply only those the user
   approves. Default: ask before applying.

## Drift signals to look for

- A function or module described in docs no longer exists or was renamed.
- A non-negotiable rule in `CLAUDE.md` contradicts wording in `RDDS_Pipeline.md`
  or `RDDS_Dev_Steps.md`.
- A done-criterion in `RDDS_Dev_Steps.md` references a command that no longer
  works (`python script.py` instead of `python -m src.…`).
- A file path in `File Structure Reference` does not exist on disk.
- A metric mentioned as "primary" in one doc and "secondary" in another.

## Do not

- Do not edit `RDDS_Technical_Document_v3.docx` (binary, deliverable for the
  course; keep manually).
- Do not rewrite whole docs. Patches must be minimal — one section at a time.
