"""
src/evaluation/compare.py
-------------------------
Baseline vs fine-tuned comparison (WP9 R4).

Aggregates the R2 metric set (produced by ``evaluate.py`` into
``experiments.metrics.evaluation_val``) across several runs and emits a single
like-for-like comparison — same fixed val set, same CRDDC2022 protocol — so a
reader can see at a glance whether the larger / fine-tuned model beats the
baseline. This is the evidence the tribunal said was missing (P3).

What it compares
----------------
- **Baseline** — ``run_20260504_202658_yolo11s`` (first YOLO11s @ 100% data).
- **The YOLO11m run(s)** — the remediation main model (R3), once evaluated.
- **Any other run** that already has ``evaluation_val`` (e.g. J's YOLO11s
  experiments, the production retrain).
- **CRDDC2022 leaderboard** — an external reference column (see
  ``CRDDC2022_LEADERBOARD``; fill the numbers from the official results).

This script is **read-only** by default: it only reads what ``evaluate.py``
already wrote. Runs that lack the R2 metric set are listed so you know to run
``python -m src.evaluation.evaluate --run-id <id>`` on them first. Pass
``--evaluate-missing`` to trigger that automatically (expensive — a GPU job per
run).

Outputs (under ``outputs/``):
- ``comparison_<ts>.csv`` / ``.json`` — the overall + per-class table.
- ``comparison_per_country_<ts>.csv`` — F1 per country, one column per run.
- ``f1_vs_data_<ts>.csv`` — the Phase-0 F1-vs-data-fraction curve.

Usage:
    # Compare every run that has evaluation_val + the leaderboard reference:
    python -m src.evaluation.compare

    # Compare a specific set:
    python -m src.evaluation.compare --run-ids run_20260504_202658_yolo11s run_2026..._yolo11m

    # Evaluate any selected run that is missing the R2 metrics first (GPU):
    python -m src.evaluation.compare --run-ids ... --evaluate-missing --device 0
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.db.connection import get_db

load_dotenv()

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
COUNTRIES = [
    "China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States",
]
OUTPUT_DIR = Path("outputs")
BASELINE_RUN_ID = "run_20260504_202658_yolo11s"

# ---------------------------------------------------------------------------
# CRDDC2022 leaderboard reference
# ---------------------------------------------------------------------------
# External reference column so the absolute level is contextualised. These are
# NOT our numbers — they are the CRDDC2022 challenge winner's official scores,
# F1 @ IoU>0.5 on the TEST set. Source: Arya et al., "Crowdsensing-based Road
# Damage Detection Challenge (CRDDC'2022)", arXiv:2211.11362, Table II
# (https://arxiv.org/abs/2211.11362 ; challenge: https://crddc2022.sekilab.global/).
#
# CRDDC2022 had FIVE leaderboards: overall-6-countries + India + Japan + Norway +
# United States. There was NO dedicated Czech or China leaderboard, and China was
# not split into Drone/MotorBike — so those entries are None (not available),
# never fabricated.
#
# CAVEAT: these are TEST-set scores under the challenge protocol; our numbers are
# on the held-out VAL set. Use this as an absolute-level reference, not a
# same-split head-to-head.
CRDDC2022_LEADERBOARD: dict[str, Any] = {
    "run_id": "CRDDC2022 winner (ShiYu_SeaView)",
    "model": "ensemble (ref)",
    "F1_overall": 0.770,  # LeaderBoard-1, 6 countries combined
    "F1_per_country": {
        "India": 0.583,
        "Japan": 0.789,
        "Norway": 0.595,
        "United_States": 0.844,
        "Czech": None,            # no dedicated CRDDC2022 leaderboard
        "China_Drone": None,      # China not split; no dedicated leaderboard
        "China_MotorBike": None,
    },
    "top3_overall": {"ShiYu_SeaView": 0.770, "DongjunJeong": 0.743, "MDPT": 0.741},
    "source": "Arya et al., CRDDC'2022, arXiv:2211.11362, Table II.",
    "note": (
        "CRDDC2022 winner, F1 @ IoU>0.5 on the official TEST set (5 leaderboards: "
        "overall-6-countries + India/Japan/Norway/US; no separate Czech/China board). "
        "Our scores are on the held-out VAL set — absolute reference, not same-split."
    ),
}


# ---------------------------------------------------------------------------
# Fetch + flatten
# ---------------------------------------------------------------------------


def _fetch_runs(run_ids: list[str] | None) -> list[dict[str, Any]]:
    """Fetch experiment docs. With no ids, return every run that has evaluation_val."""
    db = get_db()
    if run_ids:
        docs = list(db["experiments"].find({"run_id": {"$in": run_ids}}))
        found = {d["run_id"] for d in docs}
        for missing in set(run_ids) - found:
            print(f"  [warn] run_id '{missing}' not found in MongoDB — skipped.")
        # Preserve the requested order.
        order = {rid: i for i, rid in enumerate(run_ids)}
        docs.sort(key=lambda d: order.get(d["run_id"], 1_000_000))
    else:
        docs = list(db["experiments"].find({"metrics.evaluation_val": {"$exists": True}}))
        docs.sort(key=lambda d: d.get("run_id", ""))
    return docs


def _row_from_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Flatten one experiment doc into a comparison row."""
    ev = (doc.get("metrics") or {}).get("evaluation_val") or {}
    per_class = ev.get("per_class") or {}
    f1_per_class = ev.get("F1_per_class") or {c: v.get("F1") for c, v in per_class.items()}

    row: dict[str, Any] = {
        "run_id": doc.get("run_id"),
        "model": doc.get("model"),
        "sample_ratio": doc.get("sample_ratio"),
        "status": doc.get("status"),
        "is_production": bool(doc.get("is_production")),
        "has_R2": bool(per_class),
        "F1": ev.get("F1_overall"),
        "precision": ev.get("precision_overall"),
        "recall": ev.get("recall_overall"),
        "mAP50": ev.get("mAP50_overall"),
        "mAP50-95": ev.get("mAP50_95_overall"),
    }
    for c in CLASS_NAMES:
        row[f"F1_{c}"] = f1_per_class.get(c)
    row["_F1_per_country"] = ev.get("F1_per_country") or {}
    return row


# ---------------------------------------------------------------------------
# F1-vs-data curve (Phase 0)
# ---------------------------------------------------------------------------


def _f1_vs_data() -> list[dict[str, Any]]:
    """Best F1 per training-data fraction across completed YOLO11s runs.

    Uses ``evaluation_val.F1_overall`` when present, else the training-time
    ``metrics.F1`` (flagged via the ``f1_source`` column so the provenance is
    explicit). Failed runs are skipped.
    """
    db = get_db()
    docs = db["experiments"].find(
        {"model": "yolo11s", "status": {"$ne": "failed"}},
        {"run_id": 1, "sample_ratio": 1, "metrics": 1, "_id": 0},
    )
    best: dict[float, dict[str, Any]] = {}
    for d in docs:
        # The data-fraction curve is about fresh-from-scratch training at each
        # ratio; retrains are a different regime (fine-tuning on new images) and
        # would contaminate the curve, so exclude them.
        if "_retrain" in d.get("run_id", ""):
            continue
        ratio = d.get("sample_ratio")
        if ratio is None:
            continue
        m = d.get("metrics") or {}
        ev = m.get("evaluation_val") or {}
        f1 = ev.get("F1_overall")
        source = "evaluation_val"
        if f1 is None:
            f1 = m.get("F1")
            source = "training-time"
        if f1 is None:
            continue
        cur = best.get(ratio)
        if cur is None or f1 > cur["F1"]:
            best[ratio] = {"sample_ratio": ratio, "F1": f1, "f1_source": source, "run_id": d["run_id"]}
    return [best[r] for r in sorted(best)]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def compare(
    run_ids: list[str] | None = None,
    include_leaderboard: bool = True,
    evaluate_missing: bool = False,
    device: str = "0",
    ts: str | None = None,
) -> dict[str, Any]:
    """Build the comparison table(s) and write CSV/JSON artefacts.

    Args:
        run_ids: Runs to compare. None ->every run that has evaluation_val.
        include_leaderboard: Append the CRDDC2022 reference row.
        evaluate_missing: Run evaluate.py on selected runs lacking R2 metrics first.
        device: CUDA device for ``--evaluate-missing``.
        ts: Timestamp string for output filenames (callers stamp it; defaults to
            a fixed label so the function stays deterministic for tests).

    Returns:
        Dict with ``rows``, ``per_country``, ``f1_vs_data``.
    """
    print("=" * 60)
    print("RDDS — Baseline vs fine-tuned comparison (R4)")
    print("=" * 60)

    docs = _fetch_runs(run_ids)
    if not docs:
        print(
            "\nNo runs to compare. Either pass --run-ids, or run "
            "`python -m src.evaluation.evaluate --run-id <id>` so a run has "
            "metrics.evaluation_val."
        )
        return {"rows": [], "per_country": [], "f1_vs_data": []}

    # Optionally evaluate runs that lack the R2 metric set.
    if evaluate_missing:
        missing = [d["run_id"] for d in docs if not ((d.get("metrics") or {}).get("evaluation_val") or {}).get("per_class")]
        if missing:
            from src.evaluation.evaluate import evaluate as _evaluate
            for rid in missing:
                print(f"\n[evaluate-missing] Evaluating {rid} (device={device}) ...")
                try:
                    _evaluate(run_id=rid, device=device)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] evaluation of {rid} failed: {exc}")
            docs = _fetch_runs(run_ids or [d["run_id"] for d in docs])

    rows = [_row_from_doc(d) for d in docs]

    missing_r2 = [r["run_id"] for r in rows if not r["has_R2"]]
    if missing_r2:
        print(
            f"\n[note] {len(missing_r2)} run(s) lack the R2 metric set "
            f"(per-class columns will be blank): {', '.join(missing_r2)}\n"
            "       Run evaluate.py on them (or pass --evaluate-missing)."
        )

    _print_table(rows, include_leaderboard)
    f1_curve = _f1_vs_data()
    _print_f1_vs_data(f1_curve)

    stamp = ts or "latest"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_outputs(rows, f1_curve, include_leaderboard, stamp)

    return {
        "rows": rows,
        "per_country": _per_country_rows(rows, include_leaderboard),
        "f1_vs_data": f1_curve,
    }


# ---------------------------------------------------------------------------
# Rendering + output
# ---------------------------------------------------------------------------

_MAIN_COLS = ["run_id", "model", "sample_ratio", "F1", "precision", "recall", "mAP50", "mAP50-95"]


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_table(rows: list[dict[str, Any]], include_leaderboard: bool) -> None:
    print("\nOverall + per-class F1 (val set, CRDDC2022 protocol, IoU>=0.5, conf=0.5)\n")
    headers = _MAIN_COLS + [f"F1_{c}" for c in CLASS_NAMES]
    widths = {h: max(len(h), 12 if h in ("run_id",) else 8) for h in headers}
    widths["run_id"] = max(widths["run_id"], *(len(str(r["run_id"])) for r in rows), 24)
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(_fmt(r.get(h)).ljust(widths[h]) for h in headers))
    if include_leaderboard:
        lb = {"run_id": CRDDC2022_LEADERBOARD["run_id"], "model": "reference",
              "F1": CRDDC2022_LEADERBOARD["F1_overall"]}
        print("  ".join(_fmt(lb.get(h)).ljust(widths[h]) for h in headers))
        print(f"\n  [ref] {CRDDC2022_LEADERBOARD['note']}")


def _per_country_rows(rows: list[dict[str, Any]], include_leaderboard: bool) -> list[dict[str, Any]]:
    out = []
    for c in COUNTRIES:
        rec: dict[str, Any] = {"country": c}
        for r in rows:
            rec[r["run_id"]] = r["_F1_per_country"].get(c)
        if include_leaderboard:
            rec[CRDDC2022_LEADERBOARD["run_id"]] = CRDDC2022_LEADERBOARD["F1_per_country"].get(c)
        out.append(rec)
    return out


def _print_f1_vs_data(curve: list[dict[str, Any]]) -> None:
    if not curve:
        return
    print("\nF1 vs training-data fraction (Phase 0, YOLO11s):")
    print(f"  {'ratio':>6}  {'F1':>8}  {'source':<14} run_id")
    for p in curve:
        print(f"  {p['sample_ratio']:>6}  {p['F1']:>8.4f}  {p['f1_source']:<14} {p['run_id']}")


def _write_outputs(
    rows: list[dict[str, Any]],
    f1_curve: list[dict[str, Any]],
    include_leaderboard: bool,
    stamp: str,
) -> None:
    main_cols = _MAIN_COLS + ["status", "is_production", "has_R2"] + [f"F1_{c}" for c in CLASS_NAMES]
    csv_path = OUTPUT_DIR / f"comparison_{stamp}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=main_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in main_cols})

    json_path = OUTPUT_DIR / f"comparison_{stamp}.json"
    payload = {
        "rows": [{k: v for k, v in r.items() if k != "_F1_per_country"} for r in rows],
        "per_country": _per_country_rows(rows, include_leaderboard),
        "f1_vs_data": f1_curve,
        "leaderboard": CRDDC2022_LEADERBOARD if include_leaderboard else None,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pc_path = OUTPUT_DIR / f"comparison_per_country_{stamp}.csv"
    pc_rows = _per_country_rows(rows, include_leaderboard)
    if pc_rows:
        with open(pc_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pc_rows[0].keys()))
            w.writeheader()
            w.writerows(pc_rows)

    fd_path = OUTPUT_DIR / f"f1_vs_data_{stamp}.csv"
    if f1_curve:
        with open(fd_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["sample_ratio", "F1", "f1_source", "run_id"])
            w.writeheader()
            w.writerows(f1_curve)

    print(f"\nWrote: {csv_path}\n       {json_path}\n       {pc_path}"
          + (f"\n       {fd_path}" if f1_curve else ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare runs on the val set (CRDDC2022 R2 metric set).")
    p.add_argument("--run-ids", nargs="+", default=None, help="Runs to compare (default: all with evaluation_val).")
    p.add_argument("--no-leaderboard", action="store_true", help="Omit the CRDDC2022 reference row.")
    p.add_argument("--evaluate-missing", action="store_true", help="Run evaluate.py on selected runs lacking R2 metrics first (GPU).")
    p.add_argument("--device", default="0", help="CUDA device for --evaluate-missing (default: 0).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    compare(
        run_ids=args.run_ids,
        include_leaderboard=not args.no_leaderboard,
        evaluate_missing=args.evaluate_missing,
        device=args.device,
        ts=stamp,
    )
