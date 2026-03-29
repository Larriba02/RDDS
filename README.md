# Road Damage Detection System (RDDS)
Group 3 — Universidad Francisco de Vitoria

End-to-end road damage detection pipeline using deep learning on the RDD2022 dataset.

## Prerequisites
- Python 3.10+
- Git

## Setup

1. Clone the repository
   git clone https://github.com/Larriba02/rdds.git
   cd rdds

2. Create and activate a virtual environment
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Mac/Linux

3. Run the setup script
   python setup.py

   This will:
   - Install all dependencies
   - Ask for your credentials and create your .env file
   - Configure Ultralytics for the project
   - Verify the installation

4. Set RDD_DATA_ROOT in .env when the dataset is downloaded (Step 2)

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

## Team
- M — Project lead. Pipeline architecture
- L — MongoDB setup: Atlas cluster, collections, schemas.
- J — Baseline architect