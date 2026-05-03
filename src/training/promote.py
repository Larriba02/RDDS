"""
src/training/promote.py
-----------------------
Compare a newly finished training run against the current production model
and promote it if it clears the CRDDC2022 improvement threshold.

Promotion rule (CLAUDE.md §3, RDDS_Dev_Steps.md Appendix A):
    - F1_new > F1_current + 0.01  → promote.
    - F1_new ∈ [F1_current − 0.005, F1_current + 0.01]  → noise band,
      log as "completed", do NOT promote.
    - F1_new < F1_current − 0.005  → regression, log as "completed".

When no production model exists (first run ever), the new model is promoted
unconditionally.

``is_production`` is toggled inside a **MongoDB transaction** so that exactly
one document holds ``is_production=True`` at any instant (CLAUDE.md §2 rule 4).

Usage (programmatic — called by train.py):
    from src.training.promote import maybe_promote

    outcome = maybe_promote(run_id="run_20260310_001", f1_new=0.74)

Usage (standalone):
    python -m src.training.promote --run-id run_20260310_001 --f1 0.74
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import WriteConcern

from src.db.connection import DB_NAME, get_client, get_db

load_dotenv()

# CRDDC2022 promotion thresholds (CLAUDE.md §3, Appendix A).
PROMOTE_MARGIN = 0.01      # F1_new must exceed F1_current by this much.
NOISE_BAND = 0.005         # Within this below F1_current → still "completed".

PromoteOutcome = str  # "promoted" | "completed" | "regression"


def _get_production_experiment() -> dict[str, Any] | None:
    """Return the current production experiment document, or None.

    Returns:
        MongoDB document with ``is_production=True``, or ``None`` if no
        production model exists yet.
    """
    db = get_db()
    return db["experiments"].find_one({"is_production": True})


def maybe_promote(run_id: str, f1_new: float) -> PromoteOutcome:
    """Decide whether to promote the new run and apply the change atomically.

    The function:
    1. Finds the current production experiment (if any).
    2. Applies the threshold rule to decide the outcome.
    3. Executes the MongoDB promotion inside a transaction:
       - Sets the new experiment ``is_production=True``,
         ``status="promoted"``.
       - Sets the previous production experiment ``is_production=False``,
         ``status="superseded"``.
    4. Non-promoted runs are updated to ``status="completed"``.

    Args:
        run_id: ``run_id`` field of the candidate experiment document.
        f1_new: F1 score (overall, IoU=0.5) of the candidate model.

    Returns:
        Outcome string: ``"promoted"``, ``"completed"``, or ``"regression"``.

    Raises:
        ValueError: If ``run_id`` is not found in the experiments collection.
        RuntimeError: If the transaction cannot be committed after retries.
    """
    db = get_db()
    experiments = db["experiments"]

    # Validate that the run_id exists.
    candidate = experiments.find_one({"run_id": run_id})
    if candidate is None:
        raise ValueError(
            f"Experiment with run_id='{run_id}' not found in MongoDB. "
            "Make sure train.py has written the initial document before calling promote."
        )

    production = _get_production_experiment()

    # --- Decide outcome ---
    if production is None:
        # First run ever — promote unconditionally.
        outcome: PromoteOutcome = "promoted"
        f1_current = None
        delta = None
        print(f"No production model found. Promoting {run_id} unconditionally.")
    else:
        f1_current = production.get("metrics", {}).get("F1")
        if f1_current is None:
            # Current production has no F1 yet — cannot compare. Promote.
            outcome = "promoted"
            delta = None
            print(
                f"Current production model has no F1 metric. "
                f"Promoting {run_id} unconditionally."
            )
        else:
            delta = f1_new - f1_current
            if delta > PROMOTE_MARGIN:
                outcome = "promoted"
                print(
                    f"F1 {f1_new:.4f} > {f1_current:.4f} + {PROMOTE_MARGIN} "
                    f"(delta={delta:+.4f}). Promoting {run_id}."
                )
            elif delta >= -NOISE_BAND:
                outcome = "completed"
                print(
                    f"F1 {f1_new:.4f} within noise band of {f1_current:.4f} "
                    f"(delta={delta:+.4f}). Not promoting."
                )
            else:
                outcome = "regression"
                print(
                    f"F1 {f1_new:.4f} < {f1_current:.4f} − {NOISE_BAND} "
                    f"(delta={delta:+.4f}). Regression — not promoting."
                )

    now = datetime.now(timezone.utc).isoformat()

    if outcome == "promoted":
        _apply_promotion(
            run_id=run_id,
            old_run_id=production["run_id"] if production else None,
            timestamp=now,
        )
    else:
        # Just mark the run as completed (or regression — same field value).
        experiments.update_one(
            {"run_id": run_id},
            {
                "$set": {
                    "status": "completed",
                    "is_production": False,
                    "promotion_delta": delta,
                    "promotion_outcome": outcome,
                    "promotion_timestamp": now,
                }
            },
        )

    return outcome


def _apply_promotion(run_id: str, old_run_id: str | None, timestamp: str) -> None:
    """Atomically flip is_production flags using a MongoDB transaction.

    Args:
        run_id: ``run_id`` of the model to promote.
        old_run_id: ``run_id`` of the model currently in production, or
            ``None`` if there is no current production model.
        timestamp: ISO-8601 UTC timestamp string.

    Raises:
        RuntimeError: If the transaction fails.
    """
    client_obj = get_client()

    # Transactions require a replica set. Atlas free tier supports transactions.
    try:
        with client_obj.start_session() as session:
            def _txn(s):
                db = s.client[DB_NAME]
                exps = db.get_collection(
                    "experiments",
                    write_concern=WriteConcern("majority"),
                )

                # Promote the new model.
                exps.update_one(
                    {"run_id": run_id},
                    {
                        "$set": {
                            "is_production": True,
                            "status": "promoted",
                            "promotion_timestamp": timestamp,
                        }
                    },
                    session=s,
                )

                # Supersede the old model (if any).
                if old_run_id:
                    exps.update_one(
                        {"run_id": old_run_id},
                        {
                            "$set": {
                                "is_production": False,
                                "status": "superseded",
                                "superseded_at": timestamp,
                            }
                        },
                        session=s,
                    )

            session.with_transaction(_txn)

    except Exception as exc:
        raise RuntimeError(
            f"MongoDB transaction failed while promoting {run_id}: {exc}"
        ) from exc

    print(f"Promoted {run_id} to production.")
    if old_run_id:
        print(f"  Previous production model {old_run_id} marked superseded.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote a training run to production if F1 clears the threshold."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="run_id of the candidate experiment.",
    )
    parser.add_argument(
        "--f1",
        type=float,
        required=True,
        help="Overall F1 score (IoU=0.5) of the candidate model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = maybe_promote(run_id=args.run_id, f1_new=args.f1)
    print(f"\nOutcome: {result}")
