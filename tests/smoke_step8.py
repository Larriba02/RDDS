"""
Step 8 smoke test — FastAPI web demo (GET / and GET /model via TestClient).

Prerequisites:
  - MongoDB accessible (MONGO_URI in .env)
  - is_production=True experiment exists in MongoDB (GET /model needs it)

POST /predict is not tested here because it loads the YOLO model on first
call, which may trigger a B2 download. Run the server manually to test that.

Run from repo root:
    python tests/smoke_step8.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def _check_prereqs() -> str | None:
    try:
        from src.db.connection import get_db
        get_db()
    except Exception as exc:
        return f"MongoDB unreachable: {exc}"
    return None


def main() -> bool:
    print("=== Step 8 smoke test — Web Demo API ===\n")

    skip_reason = _check_prereqs()
    if skip_reason:
        print(f"SKIP: {skip_reason}")
        return True

    try:
        from fastapi.testclient import TestClient
    except (ImportError, RuntimeError):
        print("SKIP: httpx not installed (required by TestClient). Run: pip install httpx")
        return True

    from src.api.main import app
    client = TestClient(app, raise_server_exceptions=False)

    print("[1/2] GET / (frontend HTML)...")
    resp = client.get("/")
    if resp.status_code != 200:
        print(f"FAIL: GET / returned {resp.status_code}")
        return False
    print("  OK — HTML frontend served\n")

    print("[2/2] GET /model (production model metadata)...")
    resp = client.get("/model")
    if resp.status_code == 503:
        print("  SKIP: no production model in MongoDB (503) — train a model first")
    elif resp.status_code == 200:
        data = resp.json()
        print(f"  OK — run_id={data.get('run_id')}, F1={data.get('F1_overall')}")
    else:
        print(f"FAIL: GET /model returned {resp.status_code}: {resp.text}")
        return False

    print("\nStep 8 smoke test PASSED")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
