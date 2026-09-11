# CLAUDE.md — Project Instructions for Claude Code

This file is loaded automatically by Claude Code in any session opened on this
repository. It is the standing context: rules, conventions, and pointers that
should be respected on every interaction. Items here are **always-on**, not
suggestions.

For the human-facing project overview see `README.md`.
For the development plan see `DOCUMENTATION/RDDS_Dev_Steps.md`.
For the full pipeline reference see `DOCUMENTATION/RDDS_Pipeline.md`.
For the AI-assistance policy and configuration see
`DOCUMENTATION/IN DETAIL/ai_assistance.md`.

---

## 1. Project at a glance

- **Name:** Road Damage Detection System (RDDS).
- **Course:** Universidad Francisco de Vitoria, Integrating Project, Group 3.
- **Team:** M (lead — pipeline architecture, data, YOLO11s baseline, evaluation, web demo), L (MongoDB only — schema, indexes, Atlas; does not train models), J (YOLO11m main model on RTX 4060).
- **Goal:** End-to-end road-damage detection pipeline using deep learning on the
  RDD2022 dataset. Reports F1 at IoU ≥ 0.5 per CRDDC2022 protocol.
- **Timeline:** ~3 months. Final deliverables: trained model, evaluation report,
  optional web demo.
- **Hardware:** RTX 4050 laptop (6 GB, M — Phase 0 baseline and everything else),
  RTX 4060 (8 GB, J — YOLO11m main model). The A100 cluster was planned for
  Phase 1 but is **not used in the final deliverable** (kept as a documented
  design artifact).

## 2. Non-negotiable rules

These are invariants. Do not propose changes to them without explicit
discussion with M.

1. **`RANDOM_SEED = 42`** in every training run.
2. **Test split is always 100%.** Never sampled, never partial.
3. **Fixed validation set:** the same 1,000 images per country across all
   experiments, stratified by (country, dominant damage class).
4. **`is_production` is atomic.** Exactly one experiment document has
   `is_production=true` at any instant. Promotion uses a Mongo transaction.
5. **`.env` is never committed to git.** It is in `.gitignore` and must stay
   there. Suggesting `git add .env` is a hard error.
6. **No Jupyter notebooks in cluster jobs.** Python scripts only.
7. **Every training run writes to MongoDB** before, during, and after training.
8. **Checkpoints upload to Backblaze B2** immediately after a run.
9. **MLflow tracking URI** is per-machine (`./mlruns/`). MongoDB is the
   cross-machine source of truth for experiment results.

## 3. Official metric — CRDDC2022

Source: https://crddc2022.sekilab.global/overview/

- **Reported metric:** F1-score at IoU ≥ 0.5, per country and overall.
- **Training-time tracker:** mAP@0.5 on the fixed val set. Used to select
  `best.pt` and to drive Ultralytics early-stopping (`patience`).
- **Promotion rule:** `F1_new > F1_current + 0.01`. Runs within ±0.005 of the
  current production F1 are logged but **not** promoted (noise band).

Any code or doc that mentions promotion in terms of mAP must be flagged as
out of date and updated to F1.

## 4. Phase 0 / Phase 1 philosophy

The two-phase design (laptop sandbox + A100 cluster) is unchanged **in spirit**.
Phase 1 is **documented but not executed** in the final deliverable: the
YOLO11m main model is trained locally by J on the RTX 4060 instead.

- **Phase 0 = laptop sandbox (executed).** Trains YOLO11s with growing dataset
  fractions (`SAMPLE_RATIO=0.10 → 0.25 → 0.50 → 1.00`). Used to map the
  F1-vs-data curve, tune hyperparameters, and exercise the retraining workflow
  end-to-end. Output: YOLO11s baseline at `SAMPLE_RATIO=1.0`.
- **YOLO11m main model (executed locally).** Trained by J on the RTX 4060.
  This replaces the originally planned A100 cluster YOLO11m run and is the
  project's final reported main model. Promotion follows the same F1 margin
  rule in §3.
- **Phase 1 = A100 cluster (not executed).** `scripts/train_cluster.sh` and
  `scripts/submit_sweep.sh` remain in the repo as a documented design
  artifact for the SLURM-based training path. No SLURM job is reported in
  the final deliverable.
- If Phase 1 were ever executed, the design calls for a **1-epoch smoke test**
  on `tests/data/tiny_rdd2022/` first, to confirm SLURM, CUDA, MongoDB writes,
  and Backblaze upload all work on the cluster node before queueing the real
  run.

## 5. Resource locations

- **GitHub repo:** https://github.com/Larriba02/rdds (public).
- **MongoDB Atlas:** cluster `rdds`, database `rdds`, three collections
  (`images_metadata`, `experiments`, `predictions`). Connection via `MONGO_URI`
  in `.env`.
- **Backblaze B2:** bucket name in `BACKBLAZE_BUCKET`, credentials in `.env`.
  Stores `best.pt`, `last.pt`, `best.onnx`, and `results.csv` per run; URLs
  written into the `experiments.checkpoints` document (the dashboard reads
  `results_csv` as a fallback when the local file is missing).
- **Dataset (RDD2022):** Sekilab S3, public CC BY-SA 4.0. Local path on each
  machine in `RDD_DATA_ROOT`.
- **MLflow:** local `./mlruns/` per machine.

## 6. Repository layout (key paths)

```
RDDS/
├── CLAUDE.md                              # this file
├── README.md
├── setup.py                               # bootstrap script (venv assumed)
├── requirements.txt
├── .env.example                           # placeholders only — committed
├── .env                                   # real secrets — NEVER committed
├── DOCUMENTATION/
│   ├── RDDS_Dev_Steps.md                  # the development plan
│   ├── RDDS_Pipeline.md                   # full pipeline reference
│   └── IN DETAIL/                         # one doc per pipeline stage
│       ├── setup.md
│       ├── mongo.md
│       ├── data.md
│       ├── training.md
│       ├── evaluation.md
│       ├── dashboard.md
│       ├── inference.md
│       ├── retraining.md
│       ├── api.md
│       └── ai_assistance.md               # AI / Claude usage policy
├── src/
│   ├── db/                                # connection.py, setup_atlas.py, test_connection.py
│   ├── data/                              # download, validate, convert, split, ingest
│   ├── training/                          # train, upload_checkpoint, promote, retrain
│   ├── evaluation/                        # evaluate, qualitative
│   ├── inference/                         # predict, extract_frames
│   ├── api/                               # FastAPI web demo (Step 8 — optional)
│   └── dashboard.py                       # Streamlit experiment dashboard
├── tests/
│   └── data/tiny_rdd2022/                 # synthetic 5×2×4 mini-dataset
├── scripts/
│   └── train_cluster.sh                   # SLURM sbatch
└── .claude/
    ├── settings.json                      # hooks
    ├── commands/                          # /smoke-test, /rdds-review, /sync-docs, /debug-mongo
    └── agents/                            # subagents (see §8)
```

## 7. Common commands

Always run from the repo root with the venv activated.

```bash
# Step 1 — MongoDB setup and smoke test
python -m src.db.setup_atlas
python -m src.db.test_connection

# Step 2 — data ingestion (after dataset downloaded)
python -m src.data.download
python -m src.data.validate
python -m src.data.convert
python -m src.data.analyse_distribution
python -m src.data.split
python -m src.data.ingest

# Step 3 — Phase 0 training (laptop)
python -m src.training.train --model yolo11s --sample-ratio 0.10 --epochs 50 --batch 8 --patience 15

# Step 4 — Phase 1 training (cluster)
sbatch scripts/train_cluster.sh

# Evaluation
python -m src.evaluation.evaluate
```

## 8. Subagents available in this repo

Defined in `.claude/agents/`. Invoke with the Task tool when needed.

- **`code-reviewer`** — strict review of correctness, security, error handling,
  and alignment with this CLAUDE.md and the docs. Read-only (no Edit/Write).
- **`mongo-debugger`** — pymongo, Atlas, indexes, transactions, atomic
  promotion. Can edit code and write test scripts.
- **`training-debugger`** — Ultralytics, SLURM, OOM, checkpoint management,
  early stopping. Can edit code and write test scripts.
- **`doc-syncer`** — keeps `RDDS_Pipeline.md`, `RDDS_Dev_Steps.md`, the
  `IN DETAIL/` docs, and `README.md` in sync with the code. No code edits.
- **`step-implementer`** — coordinator. Implements a Step end-to-end, then
  invokes the other subagents (Task tool) to verify and correct. The only
  subagent with `Task` permission.

## 9. Slash commands

Defined in `.claude/commands/`.

- `/smoke-test` — runs `setup_atlas` + `test_connection` and interprets the result.
- `/rdds-review` — wraps an invocation of `code-reviewer` over the current branch
  diff against `dev`.
- `/sync-docs` — wraps `doc-syncer` over the recently changed code.
- `/debug-mongo` — wraps `mongo-debugger` for an Atlas / pymongo error you paste in.

## 10. Hooks

Defined in `.claude/settings.json`.

- **`block-push-without-review`** — `PreToolUse` on `Bash`. Detects `git push`,
  bails out if any commits in `origin/<branch>..HEAD` have not been reviewed in
  the current session, and asks Claude to invoke `code-reviewer` first.

## 11. Working preferences

- **Language:** M prefers Spanish for conversation, but all code, comments,
  commit messages, and documentation are in English (course requirement and
  long-term maintainability).
- **Tone:** concise, fact-based, no hedging. When ambiguous, ask a focused
  question rather than guessing.
- **Decisions:** M wants to *understand* what is happening. Prefer
  explanations that build mental model over instructions to copy-paste.
- **Edits to docs:** when adjusting `RDDS_Pipeline.md` or `RDDS_Dev_Steps.md`,
  preserve their version markers and update them when the change is material.

## 12. Known operational quirks

- **Working directory:** the repo lives at
  `<repo-root>`.
  It was originally inside OneDrive (which caused `.git/index.lock` issues
  when git was driven from a sandbox / WSL bash session), and was moved to
  `Local\` to avoid the cloud-sync interference. There is no OneDrive copy
  on disk any more. Run git from the user's PowerShell; file edits via
  Read/Write/Edit tools are fine from any shell.
- **Python version:** **3.12.x is the project standard.** The venv must be
  created with `py -3.12 -m venv .venv` (Windows) or `python3.12 -m venv
  .venv` (Mac/Linux). Python 3.13 and 3.14 are too new for Pillow / torch
  wheels (April 2026) and will fail at `pip install`. If a teammate reports
  a build error around Pillow `KeyError: '_version_'`, they are on 3.13+ —
  send them to the README §Prerequisites.
- **MongoDB cluster name:** the actual Atlas cluster is `rdds` (hostname
  `rdds.<cluster-hash>.mongodb.net`), DB user `<db-user>`. The `cluster0`
  default name is *not* what we have.
- **SAMPLE_RATIO** is set per run, not globally. The `.env` value is a fallback.
- **CUDA is required for training and evaluation.** `train.py`, `retrain.py`
  and `evaluate.py` all default to `device=0`. On a machine without a
  visible NVIDIA GPU (or where torch was installed from the CPU wheel by
  mistake — happens if `nvidia-smi` was unreachable when `setup.py` ran),
  these scripts fail immediately with `Invalid CUDA 'device=0' requested`
  or a torch DLL load error. Workaround: pass `--device cpu` to the
  script. Inference (`predict.py`) and the FastAPI demo fall back to CPU
  silently. The README §Troubleshooting documents the diagnose-and-fix
  recipe (verify with `python -c "import torch; print(torch.cuda.is_available())"`).
