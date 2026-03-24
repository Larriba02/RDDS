# Road Damage Detection System (RDDS)
Group 3 — Universidad Francisco de Vitoria

End-to-end road damage detection pipeline using deep learning on the RDD2022 dataset.

## Prerequisites
- Python 3.10+
- Git

## Setup

1. Clone the repository
   git clone https://github.com/tu_usuario/rdds.git
   cd rdds

2. Create your environment file
   cp .env.example .env
   Fill in the real values in .env (never commit this file)

3. Create and activate a virtual environment
   python -m venv .venv
   .venv\Scripts\activate

4. Install dependencies
   pip install -r requirements.txt

5. Configure Ultralytics settings
   python setup_env.py

6. Verify the installation
   python -c "import ultralytics, pymongo, mlflow; print('OK')"

## Environment Variables
- MONGO_URI: MongoDB Atlas connection string
- RDD_DATA_ROOT: Local path to the processed RDD2022 dataset
- BACKBLAZE_KEY_ID / BACKBLAZE_APP_KEY: Backblaze B2 credentials
- BACKBLAZE_BUCKET: Bucket name for model checkpoints
- SAMPLE_RATIO: Fraction of training data to use (0.10 for Phase 0, 1.0 for Phase 1)
- RANDOM_SEED: Fixed at 42 in all runs

## Team
- M — Project lead. Pipeline architecture, training, delivery.
- L — MongoDB setup: Atlas cluster, collections, schemas.
- J — TBD.

## Pipeline
See RDDS_Dev_Steps.md for the full step-by-step development guide.