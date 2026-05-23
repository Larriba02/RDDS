# Dashboard — In Detail

**Road Damage Detection System · Experiment Tracking & Validation Viewer**  
*Version 1.0 — May 2026*

---

## 1. Purpose

The dashboard is a local Streamlit web app that centralises all experiment
data stored in MongoDB.  It serves two main purposes:

1. **Track training runs** — monitor hyperparameters, training curves, and
   training-time metrics (mAP@0.5) across all experiments without querying
   MongoDB manually.
2. **View and compare validation results** — display the official CRDDC2022
   F1 scores produced by `evaluate.py`, broken down by country and damage
   class, and compare them across multiple runs side by side.

---

## 2. Running the dashboard

```bash
# From the repo root with the venv activated:
streamlit run src/dashboard.py
```

Opens at **http://localhost:8501**.  Data is cached for 30–60 seconds;
click **Refresh data** in the sidebar to force a reload.

---

## 3. Pages

### Overview

- Production model card with F1, Precision, Recall, mAP@0.5.
- Phase 0 F1-vs-data-fraction curve (best F1 per `sample_ratio`).
- Run count summary (total / completed / running / failed).

### Experiments

- Filterable table of all runs with training-time metrics.
- Horizontal bar chart comparing F1 across runs (production model marked).
- mAP@0.5 vs F1 scatter coloured by model architecture.

### Validation ★

Displays the results produced by `evaluate.py` (CRDDC2022 protocol,
IoU ≥ 0.5, conf = 0.5 on the fixed 7,000-image val set).

**Multi-run comparison** — use the multiselect at the top to choose which
runs to compare side by side.  The production model is marked with ★.

| Section | What it shows |
|---------|---------------|
| Summary table | F1 overall + Precision + Recall + mAP@0.5 + F1 per country for each selected run |
| F1 per country chart | Grouped bar chart — one group per country, one bar per run |
| F1 per class chart | Grouped bar chart — D00 / D10 / D20 / D40 per run |
| F1 overall comparison | Horizontal bar chart when more than one run is selected |

A run only appears here after `evaluate.py` has been run for it and written
results to `experiments.metrics.evaluation_val` in MongoDB.

### Run Detail

Per-run deep dive: hyperparameters, checkpoint URLs (Backblaze B2), and
per-epoch training curves read from `runs/train/<run_id>/results.csv`.

### MLflow

Summary table and param/metric detail for runs logged via the MLflow
integration in `train.py`.  For the full MLflow UI with per-epoch graphs run:

```bash
mlflow ui --backend-store-uri mlruns
# opens http://localhost:5000
```

---

## 4. Auto-evaluation on promotion

When a model is promoted to production (either automatically at the end of
`train.py` or manually via `promote.py`), the system calls `evaluate.py`
automatically on the newly promoted run.  This means that as soon as a model
becomes production, its CRDDC2022 F1 scores appear in the Validation page
without any manual intervention.

The evaluation runs synchronously after the MongoDB transaction that flips
`is_production`.  If it fails for any reason (missing checkpoint, CUDA error,
etc.) a warning is printed but the promotion itself is not rolled back.

To trigger evaluation manually for any run:

```bash
# Production model (default):
python -m src.evaluation.evaluate

# Specific run:
python -m src.evaluation.evaluate --run-id run_20260504_202658_yolo11s

# Without writing to MongoDB:
python -m src.evaluation.evaluate --dry-run
```

See `DOCUMENTATION/IN DETAIL/evaluation.md` for the full evaluation protocol.

---

## 5. Data flow

```
train.py ──► MongoDB experiments ──► dashboard.py (reads via get_db())
evaluate.py ─────────────────────►  experiments.metrics.evaluation_val
                                           │
                                           └──► Validation page charts
```

The dashboard never writes to MongoDB.  All writes go through `train.py`,
`evaluate.py`, and `promote.py`.

---

## 6. CLI reference

### `streamlit run src/dashboard.py` — Launch the experiment tracking dashboard

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
streamlit run src/dashboard.py
```

**macOS / Linux**
```bash
source .venv/bin/activate
streamlit run src/dashboard.py
```

#### Flags

No configurable flags — the dashboard reads its data exclusively from MongoDB
(`MONGO_URI` in `.env`), `runs/train/*/results.csv`, and the local MLflow
`mlruns/` directory. All configuration is done through the sidebar inside the
running app.

The dashboard opens at **http://localhost:8501** by default. To change the port,
pass Streamlit's own flag:

```bash
streamlit run src/dashboard.py --server.port 8502
```

#### Full example

```powershell
# Windows — default port
streamlit run src/dashboard.py
```
```bash
# macOS / Linux — default port
streamlit run src/dashboard.py
```

#### Troubleshooting

**`streamlit: command not found` / exit code 127**
The venv is not active. Either activate it first (`.venv\Scripts\activate` on
Windows, `source .venv/bin/activate` on macOS/Linux) or call the binary by its
full path:

```powershell
.\.venv\Scripts\streamlit.exe run src/dashboard.py
```

**PowerShell prints `NativeCommandError` but the server still starts**
Windows PowerShell 5.1 wraps any stderr line from a native executable into an
ErrorRecord when you use `2>&1`. Streamlit logs its startup banner
(`Uvicorn server started...`) to stderr, so the redirect makes PS report a
fake error even though exit code is 0 and the server is reachable on
`http://localhost:8501`. Solution: do **not** add `2>&1` — let streamlit
write to stderr normally.

**Background / detached launches die silently on Windows**
On Windows, Streamlit will not survive if it is spawned without an attached
console — neither `Start-Process -WindowStyle Hidden/Minimized` nor
`cmd /c start /MIN` keeps it alive, and the same is true of any tool that
spawns it in the background (including Claude Code's background-task runner).
The process exits as soon as the parent shell hands off.

The reliable pattern is to keep it in the foreground of an interactive
PowerShell window:

```powershell
.\.venv\Scripts\activate
streamlit run src/dashboard.py    # leave this window open
```

If you want it truly detached as a service, wrap it with `nssm` or run it
inside WSL where standard `nohup` works.
