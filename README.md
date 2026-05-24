# Road Damage Detection System (RDDS)
Group 3 — Universidad Francisco de Vitoria

End-to-end road damage detection pipeline using deep learning on the RDD2022 dataset.

## Prerequisites
- **Python 3.12.x** — required. Download from
  https://www.python.org/downloads/ (any 3.12.x release works; pick the
  latest). On Windows, run the installer with **"Add python.exe to PATH"**
  and **"py launcher"** both checked.
  Newer versions (3.13, 3.14) are *not* supported as of April 2026 — Pillow
  and torch do not yet ship pre-built wheels for them and will fail to
  install. Older versions (3.10, 3.11) may also work but 3.12 is the
  reference.
- Git
- **NVIDIA GPU with CUDA 12.4-compatible drivers (strongly recommended).**
  Training (`train.py`, `retrain.py`) and the CRDDC2022 evaluation
  (`evaluate.py`) default to `device=0` (first CUDA GPU) and will fail
  outright with `Invalid CUDA 'device=0' requested` if no GPU is visible.
  Inference (`predict.py`) and the FastAPI web demo (`src/api`) fall back
  to CPU automatically, but a single 640-pixel image takes ~3-5 s on CPU
  versus ~50 ms on GPU. The Streamlit dashboard does not need a GPU.
  Without CUDA you must pass `--device cpu` to `evaluate.py` / `retrain.py`
  / `train.py`, accept that a Phase 0 sweep epoch will take 30-60 min
  instead of 1-2 min, and skip the smoke tests that exercise training
  (`tests/smoke_step7.py`).

  Verify your installation:
  ```
  py -3.12 --version          # Windows
  python3.12 --version        # Mac/Linux
  ```

## Setup

1. Clone the repository
   ```
   git clone https://github.com/Larriba02/rdds.git
   cd rdds
   ```

2. Create and activate a virtual environment **using Python 3.12 explicitly**
   ```
   py -3.12 -m venv .venv             # Windows
   python3.12 -m venv .venv           # Mac/Linux

   .venv\Scripts\activate             # Windows
   source .venv/bin/activate          # Mac/Linux

   python --version                   # must print Python 3.12.x
   ```

   Pinning the Python version at venv creation matters: a venv is just a
   thin wrapper around whichever interpreter you invoke. If your shell's
   `python` points to 3.14, `python -m venv .venv` will create a 3.14 venv
   and `setup.py` will then fail building Pillow.

3. Run the setup script
   ```
   python setup.py
   ```

   This will:
   - Check you are on Python 3.12 (aborts with instructions if not)
   - Detect your GPU and install `torch+cu124` (NVIDIA) or `torch` CPU automatically
   - Install all remaining dependencies from `requirements.txt`
   - Ask for your credentials and create your `.env` file (UTF-8)
   - Configure Ultralytics for the project
   - Verify the installation and report whether CUDA is available
   - Offer to run a quick smoke test (Steps 1–2) to verify MongoDB and the data pipeline end-to-end

4. Set RDD_DATA_ROOT in .env when the dataset is downloaded (Step 2)

5. Verify MongoDB Atlas connection (Step 1)
   ```
   python -m src.db.setup_atlas
   python -m src.db.test_connection
   ```

   First command creates the three collections and their indexes (idempotent).
   Second command inserts/reads/deletes a sentinel document in each collection.

   Note: from step 3 onwards the venv is active, so plain `python` already
   points to the 3.12 interpreter inside `.venv`. You do **not** need to use
   `py -3.12` for these commands — only at venv creation time.

6. Data ingestion (Step 2 — run once by M, teammates pull from cloud)
   ```
   python -m src.data.download            # download RDD2022 ZIPs
   python -m src.data.validate            # validate annotations, log discards
   python -m src.data.convert             # PascalVOC XML → YOLO .txt
   python -m src.data.analyse_distribution  # class distribution → logs/
   python -m src.data.split               # assign train/val/test splits
   python -m src.data.ingest              # write metadata to MongoDB
   python -m src.data.upload_to_cloud     # upload labels + logs to B2
   ```

   To smoke-test the pipeline in seconds (no real dataset needed):
   ```
   python -m src.data.validate   --data-root tests/data/tiny_rdd2022
   python -m src.data.convert    --data-root tests/data/tiny_rdd2022
   python -m src.data.analyse_distribution --data-root tests/data/tiny_rdd2022
   python -m src.data.split      --data-root tests/data/tiny_rdd2022
   ```

7. Training (Step 3 — Phase 0 laptop baseline)
   ```
   # Phase 0: grow from 10% to 100% to map F1-vs-data curve
   python -m src.training.train --model yolo11s --sample-ratio 0.10 --epochs 50 --batch 8 --patience 15
   python -m src.training.train --model yolo11s --sample-ratio 0.25 --epochs 50 --batch 8 --patience 15
   python -m src.training.train --model yolo11s --sample-ratio 0.50 --epochs 50 --batch 8 --patience 15
   python -m src.training.train --model yolo11s --sample-ratio 1.00 --epochs 50 --batch 8 --patience 15
   ```

   Prerequisites: Step 2 must be complete (splits.json and MongoDB images_metadata populated).
   Each run writes to MongoDB, uploads checkpoints to B2, logs to MLflow, and conditionally promotes.

   To smoke-test training without real data or B2 credentials:
   ```
   # Ingest tiny dataset into MongoDB first (requires MONGO_URI in .env)
   python -m src.data.split     --data-root tests/data/tiny_rdd2022
   python -m src.data.ingest    --data-root tests/data/tiny_rdd2022

   # 1-epoch run, no B2 upload, no promotion
   python -m src.training.train --model yolo11s --sample-ratio 1.0 --epochs 1 --batch 2 --skip-upload --skip-promote
   ```

8. Inference (Step 6)
   ```
   # Extract frames from a video at 1 fps:
   python -m src.inference.extract_frames --video path/to/video.mp4 --output-dir outputs/frames/

   # Predict on a single image (uses production model from MongoDB):
   python -m src.inference.predict --source path/to/image.jpg

   # Predict on a folder (e.g. extracted frames):
   python -m src.inference.predict --source outputs/frames/ --output-dir outputs/predictions/

   # Use a local .pt file instead of downloading from B2:
   python -m src.inference.predict --source path/to/image.jpg --model runs/train/.../best.pt
   ```

9. Web demo (Step 8 — optional)
   ```
   uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
   # Then open http://localhost:8000
   ```

## Troubleshooting

**`setup.py` installed CPU torch on a machine that does have an NVIDIA GPU.**
`setup.py` decides between `torch+cu124` and `torch` (CPU) based on whether
`nvidia-smi` runs successfully at install time. If you ran setup before
installing the NVIDIA drivers, or in a shell where `nvidia-smi` is not on
PATH, the CPU wheel was installed and any later attempt to train or
evaluate will fail with `Invalid CUDA 'device=0' requested` or a torch
DLL load error (`c10.dll`). Verify with:
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If it prints `+cpu` or `False`, force-reinstall the CUDA wheel:
```
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**`Invalid CUDA 'device=0' requested` on a CPU-only machine.**
Pass `--device cpu` to the script. `evaluate.py`, `train.py` and
`retrain.py` all accept it. The FastAPI demo and `predict.py` already
fall back to CPU automatically.

**`/predict` returns 503 / API can't load the production model.**
`_resolve_checkpoint` downloads `best.pt` from Backblaze B2 when no
local copy is found in `runs/train/<run_id>/weights/`. The bucket is
private, so `BACKBLAZE_KEY_ID` and `BACKBLAZE_APP_KEY` must be set in
`.env`. The same credentials are required for the dashboard's
`results.csv` fallback. Setup prompts for them; if you skipped, edit
`.env` manually and restart.

## No credentials yet?
Contact M to receive the MongoDB Atlas URI and Backblaze credentials.
In the meantime you can still clone the repo, set up the environment,
and follow the detailed guides in DOCUMENTATION/IN DETAIL/.

## Environment Variables
- MONGO_URI: MongoDB Atlas connection string
- RDD_DATA_ROOT: Local path to the processed RDD2022 dataset
- BACKBLAZE_KEY_ID / BACKBLAZE_APP_KEY: Backblaze B2 credentials
- BACKBLAZE_BUCKET: Bucket name for model checkpoints
- RANDOM_SEED: Fixed at 42 in all runs
- SAMPLE_RATIO: Set automatically (0.10 Phase 0 / 1.0 Phase 1)

## Documentation
- DOCUMENTATION/RDDS_Dev_Steps.md — step-by-step development guide
- DOCUMENTATION/RDDS_Pipeline.md — full pipeline reference
- DOCUMENTATION/IN DETAIL/ — detailed guides for each pipeline stage
  - [dashboard.md](DOCUMENTATION/IN%20DETAIL/dashboard.md) — experiment tracking dashboard and validation results viewer
  - [evaluation.md](DOCUMENTATION/IN%20DETAIL/evaluation.md) — CRDDC2022 evaluation protocol, dataset splits, reported metrics
  - [inference.md](DOCUMENTATION/IN%20DETAIL/inference.md) — inference module: extract_frames, predict, video workflow
  - [api.md](DOCUMENTATION/IN%20DETAIL/api.md) — web demo: FastAPI routes, frontend, how to run
  - [training.md](DOCUMENTATION/IN%20DETAIL/training.md) — training pipeline, Phase 0/1, hyperparameters
  - [mongo.md](DOCUMENTATION/IN%20DETAIL/mongo.md) — MongoDB schema, collections, atomic promotion
  - [data.md](DOCUMENTATION/IN%20DETAIL/data.md) — dataset download, conversion, ingestion
  - [setup.md](DOCUMENTATION/IN%20DETAIL/setup.md) — environment setup and credentials
  - [ai_assistance.md](DOCUMENTATION/IN%20DETAIL/ai_assistance.md) — AI tooling policy and configuration

## AI-assisted development
This project uses AI tooling as a development assistant for code scaffolding,
review, debugging, and documentation maintenance. All design decisions, metric
choices, and result interpretations are made by the human team — see
`DOCUMENTATION/IN DETAIL/ai_assistance.md` §10 ("Boundaries of Trust").

The configuration shipped in `CLAUDE.md` and `.claude/` (slash commands,
subagents, hooks) targets [Claude Code](https://www.anthropic.com/claude-code)
and **requires an active Claude subscription** on the developer's account.
Without a subscription the configuration files are inert; the codebase still
runs, the AI workflow simply does not.

The same configuration is partially portable to OpenAI's
[Codex CLI](https://developers.openai.com/codex/cli) (also a paid product):
`CLAUDE.md` can be reused as `AGENTS.md`, the hook script is portable, but
subagents and slash commands have to be recreated per developer (Codex
stores them in `~/.codex/`, not in the repo).

For the full AI usage policy, configuration details, and trust boundaries,
see `DOCUMENTATION/IN DETAIL/ai_assistance.md`.

## Team
- M — Project lead. Pipeline architecture
- L — MongoDB setup: Atlas cluster, collections, schemas.
- J — Training and support