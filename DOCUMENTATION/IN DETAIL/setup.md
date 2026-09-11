# Environment Setup — In Detail
**Road Damage Detection System · Group 3 · UFV**

This document explains every step taken to configure the development environment
for RDDS, including the reasoning behind each decision. It is designed so that
any team member can replicate the setup from scratch on a new machine.

---

## 1. Git

### What it is and why we use it
Git is a version control system that records every change made to the codebase.
For RDDS it solves three concrete problems: three people working on overlapping
files simultaneously, eight clearly separated pipeline stages that need to be
developed independently, and the requirement that every training run is
reproducible from a clean setup.

### Installation

**Windows**
Download the installer from https://git-scm.com/download/win and run it.
When asked about PATH configuration, select:
> Git from the command line and also from 3rd-party software

This option makes Git accessible from PowerShell and from VSCode without
switching terminals.

**Mac**
```bash
brew install git
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt update && sudo apt install git
```

### Verify
```bash
git --version
# Expected: git version 2.53.x or newer
```

### Identity configuration (once per machine)
Git attaches a name and email to every commit. Use the same email registered
on GitHub, otherwise commits appear as anonymous on the repository.
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### Configure VSCode as the default editor
By default Git opens Vim for merge messages, which is unfamiliar.
This sets VSCode instead:
```bash
git config --global core.editor "code --wait"
```

---

## 2. GitHub Repository

### Structure
The repository is hosted at https://github.com/Larriba02/rdds (public).
It uses a three-level branch strategy:

| Branch | Purpose | Rule |
|--------|---------|------|
| `main` | Stable code only | Never commit directly |
| `dev` | Active development | All feature branches merge here |
| `feature/step-N-name` | One pipeline stage | PR back to dev when done |

### Clone the repository
```bash
git clone https://github.com/Larriba02/rdds.git
cd rdds
```

### Create and push the dev branch (owner only, done once)
```bash
git checkout -b dev
git push -u origin dev
```

`-b` creates the branch. `-u` links the local branch to the remote so future
`git push` commands do not need to specify the destination.

---

## 3. Python Virtual Environment

### Why we use one
A virtual environment is an isolated Python installation for this project.
Libraries installed inside it do not affect other projects on the machine and
vice versa. This is critical for reproducibility — every machine runs the exact
same library versions.

### Create and activate

**Always create the venv with the explicit Python 3.12 interpreter.** If your
shell's `python` points to 3.13 or 3.14, a plain `python -m venv .venv` will
create a 3.13/3.14 venv and `pip install` will fail building Pillow and torch
wheels. Use the versioned invocation below.

**Windows**
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

If PowerShell blocks the activation script with a security error, run this once:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
This allows local scripts to run without changing the global security policy.

**Mac/Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Once active, the prompt shows `(.venv)` at the start. All `pip install` commands
from this point install into the virtual environment, not the global Python.

The `.venv` folder is in `.gitignore` and is never committed — each machine
creates its own.

---

## 4. Project Setup Script

### What it does
`setup.py` is a one-command setup script that:
1. Installs all dependencies from `requirements.txt` with pinned versions
2. Asks for credentials and creates the `.env` file
3. Configures Ultralytics settings for the project
4. Verifies that the three core libraries import correctly
5. Optionally runs `tests/smoke_all.py --steps 1 2` to confirm MongoDB connectivity and the data pipeline end-to-end (prompts for confirmation; safe to skip)

### Run it
```bash
python setup.py
```

The script will ask for four credentials that M shares separately:
- `MONGO_URI` — MongoDB Atlas connection string
- `BACKBLAZE_KEY_ID` — Backblaze B2 key ID
- `BACKBLAZE_APP_KEY` — Backblaze B2 application key
- `BACKBLAZE_BUCKET` — bucket name for model checkpoints

The remaining variables (`RANDOM_SEED=42`, `SAMPLE_RATIO=1.0`) are set
automatically. `SAMPLE_RATIO` controls the fraction of training images included
by `split.py` and defaults to `1.0` so the full pool is always available.
Per-run subsampling for Phase 0 iterations is passed as `--sample-ratio` to
`train.py` at run time — do not set this to a value less than 1.0 in `.env`.

### Why dependencies are pinned
`requirements.txt` uses `==` for every package. This ensures that every machine
used in the project (M's RTX 4050 laptop, J's RTX 4060, and — in the original
Phase 1 design — the A100 cluster) uses identical library versions. Without
pinning, `pip install` fetches the latest version available at that moment,
which can differ between machines and break reproducibility.

### Ultralytics configuration
By default Ultralytics tries to connect to external experiment trackers
(ClearML, Comet, WandB, Neptune, etc.) and sends telemetry. The setup script
disables all of these except MLflow, which is the tracker defined for this
project. It also redirects checkpoints and runs to the correct project folders.

### The .env file
The `.env` file is created by `setup.py` and contains real credentials.
It is listed in `.gitignore` and must never be committed to Git. Even in a
private repository, committing credentials is a security risk because Git
history is permanent and difficult to fully purge.

The `.env.example` file is committed instead — it contains placeholder values
and documents which variables are required.

### RDD_DATA_ROOT
This variable is left as a placeholder by the setup script because the dataset
is not downloaded until Step 2. Edit `.env` manually and set the full path to
the processed RDD2022 folder when Step 2 is complete.

---

## 5. VSCode Extensions

All team members use VSCode. The following extensions are required:

| Extension | Why |
|-----------|-----|
| Python (Microsoft) | Syntax highlighting, IntelliSense, linting for all pipeline scripts |
| Pylance (Microsoft) | Fast type checking and auto-imports |
| GitLens (GitKraken) | Inline blame, branch history, PR info |
| GitHub Pull Requests | Review and merge PRs without leaving the editor |

Recommended:

| Extension | Why |
|-----------|-----|
| MongoDB for VS Code | Browse Atlas collections and verify DB writes from the editor |
| Python Debugger | Step through scripts with breakpoints |

Install via `Ctrl+Shift+X`, search by name, click Install.

---

## 6. Daily Workflow

Every working session follows this sequence:
```bash
# 1. Navigate to the project
cd "path/to/rdds"

# 2. Activate the virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# 3. Get latest changes from teammates
git checkout dev
git pull origin dev

# 4. Create a feature branch for your task
git checkout -b feature/step-N-name

# 5. Work

# 6. Stage and commit
git add .
git commit -m "stepN: short description of what changed"

# 7. Push and open a Pull Request toward dev
git push origin feature/step-N-name
```

**Never commit `.env`** — verify with `git status` before every push.
If it appears, run `git reset HEAD .env` to unstage it.

---

## 7. Repository File Structure
```
rdds/
├── DOCUMENTATION/
│   ├── RDDS_Technical_Document_v3.docx
│   ├── RDDS_Dev_Steps.md
│   ├── RDDS_Pipeline.md
│   └── IN DETAIL/
│       ├── setup.md          ← this document
│       ├── mongo.md
│       ├── data.md
│       ├── training.md
│       ├── evaluation.md
│       ├── dashboard.md
│       ├── inference.md
│       ├── retraining.md
│       ├── api.md
│       └── ai_assistance.md
├── FOLLOW-UP/
│   └── Follow-up_Template.docx
├── src/
│   ├── db/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── api/                  # optional
├── scripts/
├── logs/
├── outputs/
├── setup.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```