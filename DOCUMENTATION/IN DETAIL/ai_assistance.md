# AI Assistance — In Detail
**Road Damage Detection System · Group 3 · UFV**

This document records exactly how, when, and why AI tools are used in the
RDDS project, and the safeguards that surround that use. It is part of the
project's transparency commitment to the course and to any future maintainer
of this codebase. If you only read one section of this document, read §10
("Boundaries of Trust") — it states what AI is *not* allowed to decide.

---

## 1. Why this document exists

The RDDS project is a final-year academic deliverable. Course rules require
that all use of AI tools is disclosed. This document provides:

1. A factual statement of which AI tools are used.
2. A breakdown of *what* the AI does and *what it explicitly does not do*.
3. The configuration that makes the AI's behavior reproducible by another
   team member or by the evaluator.
4. The trust boundaries that keep the *human team* responsible for every
   substantive decision.
5. A procedure to remove or replace the AI tooling without breaking the
   project.

---

## 2. AI tools in use

### 2.1 Claude Code (Anthropic)

The primary AI tool is **Claude Code**, Anthropic's command-line agent for
software engineering. It is invoked from the developer's terminal in the
repository root and operates with full read access to the repo, scoped write
access to specific paths, and command execution under user approval.

Claude Code is a paid subscription product. **Using the configuration shipped
in this repository (`CLAUDE.md` and `.claude/`) requires an active Claude
Code subscription** on the developer's Anthropic account. Without that
subscription the configuration files are inert (they are plain markdown and
JSON; nothing breaks if they are simply ignored). See §6 for the alternative
path.

The version of Claude Code used during this project is documented in the
`FOLLOW-UP/` session records, which capture the model and CLI version active
at each working session.

### 2.2 Codex CLI (OpenAI) — alternative

The configuration is **partially portable** to OpenAI's Codex CLI. See §6
for the migration path and its limitations.

### 2.3 What is *not* used in this project

For the avoidance of doubt:

- No GitHub Copilot autocompletion.
- No ChatGPT web interface for code generation (it is used occasionally for
  research and explanation, but never to author code that lands in the repo).
- No Devin, Cursor agent mode, Aider, or other agentic coding tools.
- No fine-tuned or custom-trained model.

If any of this changes, this section is the first thing to update.

---

## 3. What the AI is allowed to do

The AI is treated as a junior team member who must justify every change.
Concretely:

| Activity | AI assists | Human authority |
|---|---|---|
| Scaffolding modules from a Step's spec | Yes | M reviews diff before commit |
| Writing data-pipeline boilerplate (download, validate, convert) | Yes | M and L review |
| Writing or editing documentation (`.md` files in `DOCUMENTATION/`) | Yes | M reviews |
| Debugging Atlas / pymongo errors | Yes | L approves DB-layer fixes |
| Debugging Ultralytics / SLURM / OOM | Yes | M approves training-layer fixes |
| Generating Mongo queries and index definitions | Yes | L reviews |
| Suggesting refactors | Yes | M decides |
| Writing tests | Yes | M reviews |
| Sanity-checking PRs before merge to `dev` or `main` | Yes (mandatory) | M decides merge |
| Updating documentation when code changes | Yes (mandatory) | M reviews |

Mandatory items run through the slash commands `/rdds-review` and `/sync-docs`
and the corresponding subagents (see §5).

---

## 4. What the AI is *not* allowed to do

These are hard limits encoded in `CLAUDE.md`, in each subagent's `tools` list,
and in the hook in `.claude/settings.json`:

1. **No autonomous merges.** Every merge to `dev` or `main` is initiated by a
   human running `git merge` (or accepting a PR) in their own terminal.
2. **No autonomous push.** The `block-push-without-review` hook (§8) blocks
   `git push` until a fresh `code-reviewer` pass is recorded.
3. **No editing of `.env` or any secret.** Hard-coded in `CLAUDE.md` §2 and
   in every subagent's hard-limits section.
4. **No starting Phase 1 (A100) jobs.** `training-debugger` may run a
   1-epoch smoke test on the synthetic mini-dataset. Real Phase 1 runs are
   `sbatch`-ed by M on the cluster login node.
5. **No starting unattended Phase 0 runs longer than ~10 minutes** without
   explicit user approval per run.
6. **No deletion of checkpoints from Backblaze.** Bad runs get
   `status: "superseded"` in MongoDB.
7. **No modification of binary deliverables** (`.docx`, `.pptx`, `.pdf`).
8. **No metric redefinition.** The CRDDC2022 protocol (F1 at IoU ≥ 0.5,
   promotion margin 0.01, noise band 0.005) is fixed by the course
   competition, not by the team. Any AI-suggested metric change requires
   human discussion before it lands.
9. **No model architecture change.** YOLO11s (baseline) + YOLO11m (main)
   are decisions made by the human team, in writing (`RDDS_Pipeline.md` and
   `RDDS_Dev_Steps.md`).
10. **No interpretation of results in the final report.** The AI may compute
    metrics, format tables, and draft bullet points, but the written
    interpretation and discussion sections of the final deliverable are
    written by the human team.

---

## 5. Configuration files in this repository

All AI configuration lives in three places:

### 5.1 `CLAUDE.md` (repo root)

The standing context loaded by Claude Code at the start of every session in
this directory. Contains:

- Project at a glance (team, hardware, timeline).
- The non-negotiable rules (§2 of `CLAUDE.md`).
- The official metric (CRDDC2022 F1 at IoU ≥ 0.5).
- Phase 0 / Phase 1 philosophy.
- Resource locations (Atlas, Backblaze, MLflow).
- Repository layout.
- Common commands.
- The list of subagents and slash commands available.
- Working preferences (language, tone, decision style).
- Known operational quirks (OneDrive + git, Python version, cluster name).

`CLAUDE.md` is checked into git and shared with the team.

### 5.2 `.claude/commands/` — slash commands

Markdown files defining short, repeatable workflows the user invokes by typing
`/<name>` in Claude Code:

- **`/smoke-test`** — runs `python -m src.db.setup_atlas` and
  `python -m src.db.test_connection` and interprets the result. Routes
  failures to `mongo-debugger`.
- **`/rdds-review [base-branch]`** — captures the diff vs the base branch
  (default `dev`), invokes `code-reviewer`, and reports findings grouped as
  Blockers / Should fix / Nits.
- **`/sync-docs`** — invokes `doc-syncer` to detect and patch drift between
  the code and the documentation.
- **`/debug-mongo <error>`** — wraps `mongo-debugger` for an Atlas / pymongo
  error pasted as the argument.

### 5.3 `.claude/agents/` — subagents

Markdown files defining specialized AI agents with restricted tool access.
Each subagent declares its allowed tools in its frontmatter. The full list
(with rationale for each tool restriction):

| Subagent | Tools | Rationale |
|---|---|---|
| `code-reviewer` | Read, Grep, Glob, Bash | Read-only by design. A reviewer that can edit ceases to be a reviewer. Bash is restricted to read-only git commands. |
| `mongo-debugger` | Read, Grep, Glob, Bash, Edit, Write, WebFetch | Edits `src/db/`, writes throwaway test scripts, fetches pymongo / Atlas docs. |
| `training-debugger` | Read, Grep, Glob, Bash, Edit, Write, WebFetch | Edits `src/training/` and `scripts/`, writes smoke scripts, fetches Ultralytics / SLURM docs. |
| `doc-syncer` | Read, Grep, Glob, Edit, Write, Bash | Edits markdown docs only. Bash is for read-only git inspection. |
| `step-implementer` | Read, Grep, Glob, Bash, Edit, Write, **Task**, WebFetch | The only subagent with `Task`. Coordinates the others to implement and verify a Step end-to-end. |

Notes on the design:

- Only **one** subagent has `Task`. The implementer coordinates; the others
  are focused specialists. This keeps the orchestration legible and stops
  cycles (a verifier cannot recursively invoke other verifiers without going
  through the implementer).
- `code-reviewer` does **not** have `Edit` or `Write`. This is enforced both
  by the tools list and by the system prompt in the subagent file.
- `step-implementer` has `Task` but its system prompt explicitly forbids
  pushing, A100 jobs, and binary edits.

### 5.4 `.claude/settings.json` — hooks

Defines the `block-push-without-review` hook described in §8.

### 5.5 `.claude/hooks/block-push-without-review.py`

The hook script. See §8 for behavior.

---

## 6. Codex CLI as alternative

OpenAI's Codex CLI implements equivalents to the four building blocks above:

- **`AGENTS.md`** ≡ `CLAUDE.md`. Codex CLI loads `AGENTS.md` automatically;
  files closer to the working directory take precedence. **This is portable
  and committable**: a teammate using Codex CLI can rename `CLAUDE.md` to
  `AGENTS.md` (or symlink it) and most of the standing context applies.

- **Subagents** in `~/.codex/agents/` as TOML files. **Not portable**: they
  live in the user's home directory, not in the repo. A teammate who switches
  to Codex would need to manually translate the markdown subagent definitions
  in `.claude/agents/` into TOML and place them in their own
  `~/.codex/agents/`.

- **Custom slash commands (custom prompts)** in `~/.codex/prompts/` as
  Markdown files. **Not portable** for the same reason. The four `/smoke-test`,
  `/rdds-review`, `/sync-docs`, `/debug-mongo` commands would need to be
  recreated per developer.

- **Hooks** are configurable in `~/.codex/config.toml` and observe MCP tools
  and Bash. The `block-push-without-review` Python script in
  `.claude/hooks/` can be reused; only the configuration plumbing changes.

In summary: Codex CLI **can** replace Claude Code for this project. The
standing context (`CLAUDE.md` / `AGENTS.md`) is preserved without effort. The
subagents and slash commands have to be recreated by hand on each developer's
machine. The hook script is portable.

This document does not endorse either tool. Either is acceptable; both
require a paid subscription to their respective vendors.

---

## 7. Workflows: how AI participates in each Step

### Step 0 — Repo and environment setup
Done before this document existed. Used Claude as a research/explanation
assistant for git, `.gitignore`, and venv conventions. No code authored by AI.

### Step 1 — MongoDB setup
- Scaffolding of `src/db/connection.py`, `src/db/setup_atlas.py`, and
  `src/db/test_connection.py`: drafted by Claude, reviewed by L and M.
- Atlas cluster provisioning: done manually by L on cloud.mongodb.com.
- Index design: discussed with Claude as a sounding board; final decisions in
  `DOCUMENTATION/IN DETAIL/mongo.md` §4 owned by L.

### Step 2 — Data Ingestion (upcoming)
Will use `step-implementer` to scaffold `src/data/*.py` from the spec, then
`code-reviewer` and `doc-syncer` for verification. Real RDD2022 download
launched by M; the synthetic mini-dataset (`tests/data/tiny_rdd2022/`) is
used for smoke testing.

### Step 3 — Phase 0 Training (upcoming)
Phase 0 is the laptop sandbox (see `RDDS_Dev_Steps.md` Step 3). Code authored
with `step-implementer`, training launched manually by M. `training-debugger`
used reactively when runs fail or OOM.

### Step 4 — Phase 1 Training (upcoming)
Cluster smoke test (1 epoch on the mini-dataset) authored and run with
`step-implementer` and `training-debugger`. Real Phase 1 runs are
`sbatch`-ed by M on the cluster login node — the AI never queues an A100
job autonomously.

### Step 5 — Evaluation (upcoming)
Metrics scripts authored with `step-implementer`. The 200 qualitative
samples are inspected and described by humans; the AI does not write the
qualitative analysis section of the final report.

### Step 6 — Inference (upcoming)
Same pattern as Step 5.

### Step 7 — Retraining (upcoming)
`retrain()` authored with `step-implementer`. Smoke-tested with the
synthetic dataset. The promotion logic is reviewed by `code-reviewer` with
extra attention to the atomicity guarantee.

### Step 8 — Web demo (optional, conditional)
If undertaken, FastAPI scaffolding and frontend boilerplate are obvious AI
candidates. Final UX decisions by humans.

---

## 8. The `block-push-without-review` hook

### Purpose

Prevent unreviewed code from leaving the developer's machine. The hook fires
on every `git push` and blocks the push if there are commits between the
local HEAD and the remote tracking branch that have not been recorded as
reviewed in the current session.

### Mechanism

Configured in `.claude/settings.json` as a `PreToolUse` hook on the `Bash`
tool. The script is `.claude/hooks/block-push-without-review.py`.

When Claude Code is about to run a Bash command, the hook receives the tool
input over stdin as JSON. The script:

1. Returns 0 (allow) if the tool is not Bash or the command is not `git push`.
2. Otherwise computes `git log origin/<branch>..HEAD --oneline`.
3. Returns 0 if that list is empty (nothing unreviewed).
4. Otherwise returns 2 with an explanatory message on stderr. Claude Code
   surfaces the message to the agent and blocks the push.

### Review marker

The hook honors a soft acknowledgment marker at `.claude/.review-marker`. To
authorize a push after a review:

```powershell
git rev-parse HEAD > .claude\.review-marker
git push
```

The marker invalidates automatically when HEAD changes (any new commit). The
file is gitignored so it is per-developer state.

### Bypass

To bypass entirely (e.g. emergency hotfix, automated CI), invoke the push
from a shell **outside** Claude Code. The hook only fires when Claude Code
is executing the Bash tool. Direct user-driven `git push` from PowerShell or
Terminal is never blocked. This is intentional: the hook protects against
agent-initiated pushes, not against the human.

---

## 9. Trust and verification

### Why we trust the AI for some things

- **Boilerplate**: `pymongo` connection setup, `argparse`, file I/O, YAML
  parsing — all well-understood code patterns where an LLM is at least as
  reliable as a tired student at 1am.
- **Documentation drift detection**: a tireless reader is better than a
  human at noticing that a function description in `Pipeline.md` no longer
  matches the code.
- **Code review checklists**: catching a missing `try/except`, a hardcoded
  password, or a missing `seed=42` argument is high-recall AI work.

### Why we do not trust the AI for some things

- **Metric decisions**: the project competes against CRDDC2022's official
  protocol. F1 vs mAP is a *contest rule*, not a tunable. AI assistance here
  introduces drift risk.
- **Result interpretation**: the discussion section of the final report
  must reflect what the team actually understood and concluded — fabricated
  insight is a course-integrity issue.
- **Architecture choices**: YOLO11s vs YOLO11m vs alternatives is a design
  decision that depends on hardware, dataset size, and timeline trade-offs
  that the team owns.
- **Resource provisioning**: AI does not create Atlas clusters, Backblaze
  buckets, GitHub repos, SLURM accounts. Humans own the credentials.

### Verification: what the human always does

For every commit that contains AI-authored code:

1. Read the diff before committing. No `git commit -a` without inspection.
2. Run smoke tests if the change touches tested code paths.
3. Confirm the change matches the relevant section of `RDDS_Dev_Steps.md`.

For every PR before merging:

4. `/rdds-review` to invoke `code-reviewer`.
5. Read the verdict. Apply blockers and should-fix items.
6. Push (`block-push-without-review` hook will require the marker).
7. Merge manually.

---

## 10. Boundaries of Trust (read this section)

**The human team is the author of this project.** AI is a tool the team uses
to move faster and catch its own mistakes. The team is responsible for:

- Every commit message and every line of code that lands in the repo.
- Every metric reported in the final document.
- Every architectural decision in `RDDS_Pipeline.md` and
  `RDDS_Dev_Steps.md`.
- Every interpretation in the final discussion and conclusion.

If a reader of the final report or codebase finds a bug, a wrong metric, or
a misleading claim, the answer is *not* "Claude wrote it." The answer is "the
team is responsible." This is not a disclaimer to escape blame; it is the
working assumption that decides how AI assistance is used.

---

## 11. Removing AI from the project

If the team or a future maintainer wants to remove AI tooling entirely:

1. Delete `CLAUDE.md` and `.claude/`.
2. Delete `DOCUMENTATION/IN DETAIL/ai_assistance.md`.
3. Delete the AI-assistance section in `README.md`.
4. Remove the `.claude/.review-marker` line from `.gitignore` (if present).

The codebase will continue to work. None of the runtime behavior depends on
the AI configuration files. They are pure documentation and orchestration
metadata for the development workflow.

---

## 12. Logbook

| Date | Change |
|---|---|
| 2026-04-25 | Document created. AI configuration scaffolded: `CLAUDE.md`, `.claude/commands/` (4), `.claude/agents/` (5), `.claude/settings.json`, `.claude/hooks/block-push-without-review.py`. Disclaimer added to `README.md`. |

This logbook is appended to whenever the AI configuration changes. The goal
is that the evaluator can read this single section and know exactly what
shape the AI assistance had at any point in the project's life.
