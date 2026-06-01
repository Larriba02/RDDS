"""
src/dashboard.py
----------------
Streamlit dashboard for RDDS experiment tracking.

Data sources:
  - MongoDB `experiments` collection -- canonical run results, hyperparams.
  - runs/train/{run_id}/results.csv  -- per-epoch training curves (Ultralytics).
  - MLflow mlruns/                   -- params and final metrics logged by train.py.

Run:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src.*` imports work regardless of CWD.
_REPO_ROOT = Path(__file__).parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import io

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from plotly.subplots import make_subplots

load_dotenv()

from src.db.connection import get_db  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUNS_DIR = _REPO_ROOT / "runs" / "train"
MLFLOW_URI = (_REPO_ROOT / "mlruns").resolve().as_uri()
CLASS_NAMES = ["D00", "D10", "D20", "D40"]

st.set_page_config(
    page_title="RDDS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loaders (cached so they don't re-query on every interaction)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def load_experiments() -> pd.DataFrame:
    try:
        db = get_db()
    except Exception as exc:
        st.error(f"Cannot connect to MongoDB: {exc}\n\nCheck that MONGO_URI is set in .env.")
        st.stop()
    docs = list(
        db["experiments"].find(
            {},
            {
                "_id": 0,
                "run_id": 1,
                "model": 1,
                "sample_ratio": 1,
                "status": 1,
                "is_production": 1,
                "metrics": 1,
                "hyperparams": 1,
                "checkpoints": 1,
                "timestamp": 1,
                "completed_at": 1,
                "dataset_countries": 1,
            },
        )
    )
    if not docs:
        return pd.DataFrame()

    rows = []
    for d in docs:
        m = d.get("metrics", {})
        rows.append(
            {
                "run_id": d.get("run_id", ""),
                "model": d.get("model", ""),
                "sample_ratio": d.get("sample_ratio"),
                "status": d.get("status", ""),
                "is_production": d.get("is_production", False),
                "F1": m.get("F1"),
                "mAP50": m.get("mAP50"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "timestamp": d.get("timestamp", ""),
                "completed_at": d.get("completed_at", ""),
                "hyperparams": d.get("hyperparams", {}),
                "checkpoints": d.get("checkpoints", {}),
                "countries": ", ".join(d.get("dataset_countries", [])),
            }
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["completed_at"] = pd.to_datetime(df["completed_at"], utc=True, errors="coerce")
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=30, show_spinner=False)
def load_validation_results() -> list[dict]:
    try:
        db = get_db()
    except Exception:
        return []
    return list(
        db["experiments"].find(
            {"metrics.evaluation_val": {"$exists": True}},
            {
                "_id": 0,
                "run_id": 1,
                "model": 1,
                "sample_ratio": 1,
                "is_production": 1,
                "status": 1,
                "metrics.evaluation_val": 1,
            },
        )
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_results_csv(run_id: str, b2_url: str | None = None) -> pd.DataFrame | None:
    csv_path = RUNS_DIR / run_id / "results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        return df
    if b2_url:
        try:
            csv_text = _fetch_b2_text(b2_url, timeout=10)
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as exc:
            st.warning(f"Could not fetch results.csv from B2: {exc}")
            return None
        df.columns = [c.strip() for c in df.columns]
        return df
    return None


def load_progress(run_id: str) -> dict:
    """Read the live per-epoch progress for a run straight from MongoDB.

    Deliberately NOT cached: the Live Training page polls this on a timer and
    must see fresh data each refresh. MongoDB is the cross-machine source of
    truth (CLAUDE.md §9), so this works for a run executing on a remote host
    (e.g. the A100 cluster) just as well as a local one.

    Returns the document fields needed to render live curves, or an empty dict
    if the run is missing or MongoDB is unreachable.
    """
    try:
        db = get_db()
    except Exception:
        return {}
    doc = db["experiments"].find_one(
        {"run_id": run_id},
        {
            "_id": 0,
            "model": 1,
            "status": 1,
            "is_production": 1,
            "progress": 1,
            "progress_current_epoch": 1,
            "progress_total_epochs": 1,
            "progress_updated_at": 1,
            "metrics": 1,
        },
    )
    return doc or {}


def _fetch_b2_text(url: str, timeout: int = 10) -> str:
    """Fetch a Backblaze B2 object as text using authenticated S3 API.

    The bucket is private, so anonymous requests return 401. Credentials are
    read from .env (BACKBLAZE_KEY_ID, BACKBLAZE_APP_KEY).
    """
    import os
    import urllib.parse

    import boto3
    from botocore.config import Config

    parsed = urllib.parse.urlparse(url)
    endpoint = f"{parsed.scheme}://{parsed.netloc}"
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) != 2:
        raise RuntimeError(f"Cannot parse bucket/key from URL: {url}")
    bucket, key = parts

    key_id = os.getenv("BACKBLAZE_KEY_ID")
    app_key = os.getenv("BACKBLAZE_APP_KEY")
    if not key_id or not app_key:
        raise RuntimeError(
            "BACKBLAZE_KEY_ID and BACKBLAZE_APP_KEY must be set in .env to "
            "fetch results.csv from the private B2 bucket."
        )

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
        config=Config(
            signature_version="s3v4",
            connect_timeout=timeout,
            read_timeout=timeout,
        ),
    )
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read().decode("utf-8")


@st.cache_data(ttl=60, show_spinner=False)
def load_mlflow_runs() -> pd.DataFrame:
    try:
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        exps = client.search_experiments()
        records = []
        for exp in exps:
            for run in client.search_runs(exp.experiment_id):
                records.append(
                    {
                        "mlflow_run_id": run.info.run_id,
                        "run_name": run.info.run_name,
                        "experiment": exp.name,
                        "status": run.info.status,
                        "params": run.data.params,
                        "metrics": run.data.metrics,
                    }
                )
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status_badge(status: str, is_prod: bool) -> str:
    if is_prod:
        return "PRODUCTION"
    color = {"completed": "green", "running": "orange", "failed": "red"}.get(
        status, "gray"
    )
    return f":{color}[{status}]"


def _fmt_pct(val) -> str:
    return f"{val * 100:.2f}%" if val is not None and not pd.isna(val) else "N/A"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("RDDS")
    st.caption("Road Damage Detection System")
    page = st.radio(
        "Navigate",
        ["Overview", "Live Training", "Experiments", "Validation", "Run Detail", "MLflow"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with st.spinner("Loading experiments..."):
    df_all = load_experiments()

_FINISHED = {"completed", "promoted", "superseded"}
df_completed = (
    df_all[df_all["status"].isin(_FINISHED)].copy()
    if not df_all.empty
    else pd.DataFrame()
)

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------

if page == "Overview":
    st.title("Overview")

    if df_all.empty:
        st.warning("No experiments found in MongoDB.")
        st.stop()

    # Production model card
    prod_rows = df_all[df_all["is_production"] == True]  # noqa: E712
    if not prod_rows.empty:
        prod = prod_rows.iloc[0]
        st.subheader("Production model")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Run ID", prod["run_id"].split("_", 1)[1] if "_" in prod["run_id"] else prod["run_id"])
        col2.metric("F1", _fmt_pct(prod["F1"]))
        col3.metric("mAP@0.5", _fmt_pct(prod["mAP50"]))
        col4.metric("Precision", _fmt_pct(prod["precision"]))
        col5.metric("Recall", _fmt_pct(prod["recall"]))

        with st.expander("Hyperparameters"):
            hp = prod["hyperparams"]
            if hp:
                hp_df = pd.DataFrame(
                    [{"param": str(k), "value": str(v)} for k, v in hp.items()]
                )
                st.dataframe(hp_df, width="stretch", hide_index=True)
            else:
                st.write("No hyperparameter data.")
    else:
        st.info("No production model set yet.")

    st.divider()

    # F1-vs-data curve
    st.subheader("Phase 0 — F1 vs data fraction")
    if not df_completed.empty and df_completed["F1"].notna().any():
        curve_df = (
            df_completed[df_completed["F1"].notna()]
            .groupby("sample_ratio", as_index=False)["F1"]
            .max()
            .sort_values("sample_ratio")
        )
        fig = px.line(
            curve_df,
            x="sample_ratio",
            y="F1",
            markers=True,
            labels={"sample_ratio": "Sample ratio", "F1": "F1 score"},
            title="Best F1 per sample ratio",
        )
        fig.update_traces(marker_size=10, line_width=2)
        fig.update_layout(
            xaxis=dict(tickformat=".0%"),
            yaxis=dict(range=[0, 1]),
            height=350,
        )
        # Annotate each point
        for _, row in curve_df.iterrows():
            fig.add_annotation(
                x=row["sample_ratio"],
                y=row["F1"],
                text=f"{row['F1']:.3f}",
                showarrow=False,
                yshift=14,
                font_size=12,
            )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No completed runs with F1 data yet.")

    st.divider()

    # Run count summary
    st.subheader("Run summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", len(df_all))
    c2.metric("Completed", len(df_completed))
    c3.metric("Running", len(df_all[df_all["status"] == "running"]))
    c4.metric("Failed", len(df_all[df_all["status"] == "failed"]))

# ---------------------------------------------------------------------------
# Page: Live Training
# ---------------------------------------------------------------------------

elif page == "Live Training":
    st.title("Live Training")
    st.caption(
        "Per-epoch metrics streamed to MongoDB by the training callback. "
        "Because MongoDB is the cross-machine source of truth, this follows a "
        "run executing on another host (e.g. the A100 cluster) in real time — "
        "no local results.csv required."
    )

    if df_all.empty:
        st.warning("No experiments found in MongoDB.")
        st.stop()

    running_ids = df_all[df_all["status"] == "running"]["run_id"].tolist()
    all_ids = df_all["run_id"].tolist()
    # Running runs first; fall back to any run (so finished runs can be replayed).
    options = running_ids + [r for r in all_ids if r not in running_ids]

    top = st.columns([3, 1, 1])
    selected_run = top[0].selectbox(
        "Run",
        options,
        help="Runs with status='running' are listed first.",
    )
    auto = top[1].checkbox("Auto-refresh", value=True)
    interval = top[2].selectbox("Every", [3, 5, 10, 30], index=1, format_func=lambda s: f"{s}s")

    if running_ids:
        st.success(f"{len(running_ids)} run(s) currently training: {', '.join(running_ids)}")
    else:
        st.info("No run is currently training. Showing recorded per-epoch history for the selected run.")

    _LIVE_COL_MAP = {
        "epoch": "Epoch",
        "train/box_loss": "Train box loss",
        "train/cls_loss": "Train cls loss",
        "train/dfl_loss": "Train dfl loss",
        "val/box_loss": "Val box loss",
        "val/cls_loss": "Val cls loss",
        "val/dfl_loss": "Val dfl loss",
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "metrics/mAP50(B)": "mAP@0.5",
        "metrics/mAP50-95(B)": "mAP@0.5:0.95",
        "F1": "F1",
    }

    @st.fragment(run_every=(f"{interval}s" if auto else None))
    def _live_view() -> None:
        doc = load_progress(selected_run)
        if not doc:
            st.error(f"Run '{selected_run}' not found in MongoDB.")
            return

        status = doc.get("status", "?")
        cur = doc.get("progress_current_epoch") or 0
        total = doc.get("progress_total_epochs") or 0
        updated = doc.get("progress_updated_at")

        # Status / progress header.
        h = st.columns([2, 2, 2])
        h[0].markdown(f"**Status:** {_status_badge(status, doc.get('is_production', False))}")
        h[1].markdown(f"**Model:** `{doc.get('model', '?')}`")
        h[2].markdown(f"**Last update:** {updated or 'n/a'}")

        if total:
            st.progress(min(cur / total, 1.0), text=f"Epoch {cur} / {total}")

        progress = doc.get("progress") or []
        if not progress:
            st.info(
                "No per-epoch records yet. The first row appears after epoch 1 "
                "finishes its validation pass."
            )
            return

        df_p = pd.DataFrame(progress)
        df_p = df_p.rename(columns={k: v for k, v in _LIVE_COL_MAP.items() if k in df_p.columns})
        df_p = df_p.sort_values("Epoch") if "Epoch" in df_p.columns else df_p
        mode = "lines+markers" if len(df_p) >= 2 else "markers"

        # Latest-epoch metric cards.
        last = df_p.iloc[-1]
        cards = st.columns(5)
        for col_box, name in zip(
            cards, ["F1", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
        ):
            val = last.get(name)
            col_box.metric(name, f"{val:.4f}" if pd.notna(val) else "N/A")

        tab_m, tab_l = st.tabs(["Metrics", "Losses"])

        with tab_m:
            metric_cols = [c for c in ["F1", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"] if c in df_p.columns]
            if metric_cols and "Epoch" in df_p.columns:
                fig = go.Figure()
                for col in metric_cols:
                    fig.add_trace(go.Scatter(x=df_p["Epoch"], y=df_p[col], name=col, mode=mode))
                fig.update_layout(
                    title="Validation metrics per epoch (live)",
                    xaxis_title="Epoch", yaxis_title="Value", yaxis_range=[0, 1],
                    height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, width="stretch")

        with tab_l:
            train_cols = [c for c in ["Train box loss", "Train cls loss", "Train dfl loss"] if c in df_p.columns]
            val_cols = [c for c in ["Val box loss", "Val cls loss", "Val dfl loss"] if c in df_p.columns]
            if (train_cols or val_cols) and "Epoch" in df_p.columns:
                fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Train losses", "Val losses"))
                for col in train_cols:
                    fig2.add_trace(go.Scatter(x=df_p["Epoch"], y=df_p[col], name=col, mode=mode), row=1, col=1)
                for col in val_cols:
                    fig2.add_trace(go.Scatter(x=df_p["Epoch"], y=df_p[col], name=col, mode=mode), row=1, col=2)
                fig2.update_layout(height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig2, width="stretch")
            else:
                st.info("No loss columns recorded yet.")

        with st.expander("Raw per-epoch records", expanded=False):
            st.dataframe(df_p, width="stretch", hide_index=True)

    _live_view()

# ---------------------------------------------------------------------------
# Page: Experiments
# ---------------------------------------------------------------------------

elif page == "Experiments":
    st.title("Experiments")

    if df_all.empty:
        st.warning("No experiments found in MongoDB.")
        st.stop()

    # Filters
    col_f1, col_f2 = st.columns(2)
    status_filter = col_f1.multiselect(
        "Status",
        df_all["status"].unique().tolist(),
        default=df_all["status"].unique().tolist(),
    )
    model_filter = col_f2.multiselect(
        "Model",
        df_all["model"].unique().tolist(),
        default=df_all["model"].unique().tolist(),
    )

    mask = df_all["status"].isin(status_filter) & df_all["model"].isin(model_filter)
    df_view = df_all[mask].copy()

    # Display table
    display_cols = ["run_id", "model", "sample_ratio", "status", "is_production", "F1", "mAP50", "precision", "recall", "completed_at"]
    for col in ["F1", "mAP50", "precision", "recall"]:
        if col in df_view.columns:
            df_view[col] = df_view[col].apply(lambda v: round(v, 4) if pd.notna(v) else None)

    st.dataframe(
        df_view[display_cols],
        width="stretch",
        hide_index=True,
        column_config={
            "is_production": st.column_config.CheckboxColumn("Prod"),
            "sample_ratio": st.column_config.NumberColumn("Sample ratio", format="%.2f"),
            "F1": st.column_config.NumberColumn("F1", format="%.4f"),
            "mAP50": st.column_config.NumberColumn("mAP@0.5", format="%.4f"),
            "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
        },
    )

    st.divider()

    # Comparison bar chart
    df_chart = df_view[df_view["F1"].notna()].sort_values("F1", ascending=True)
    if not df_chart.empty:
        fig = px.bar(
            df_chart,
            x="F1",
            y="run_id",
            orientation="h",
            color="model",
            hover_data=["sample_ratio", "mAP50", "precision", "recall"],
            title="F1 by run (completed only)",
            height=max(300, len(df_chart) * 32),
        )
        fig.update_layout(yaxis_title=None, xaxis_range=[0, 1])
        # Mark production run
        prod_ids = df_all[df_all["is_production"] == True]["run_id"].tolist()  # noqa: E712
        for run_id in prod_ids:
            if run_id not in df_chart["run_id"].values:
                continue
            fig.add_vline(
                x=df_chart.loc[df_chart["run_id"] == run_id, "F1"].values[0],
                line_dash="dot",
                line_color="gold",
                annotation_text="production",
            )
        st.plotly_chart(fig, width="stretch")

    # mAP50 vs F1 scatter
    df_scatter = df_view[df_view["F1"].notna() & df_view["mAP50"].notna()]
    if len(df_scatter) > 1:
        fig2 = px.scatter(
            df_scatter,
            x="mAP50",
            y="F1",
            color="model",
            size="sample_ratio",
            hover_data=["run_id", "sample_ratio"],
            title="mAP@0.5 vs F1",
        )
        fig2.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1], height=350)
        st.plotly_chart(fig2, width="stretch")

# ---------------------------------------------------------------------------
# Page: Validation
# ---------------------------------------------------------------------------

elif page == "Validation":
    st.title("Validation Results")
    st.caption("CRDDC2022 protocol — F1 @ IoU ≥ 0.5, fixed validation set (1 000 images/country)")

    val_docs = load_validation_results()
    if not val_docs:
        st.warning("No evaluation results found. Run `python -m src.evaluation.evaluate` first.")
        st.stop()

    COUNTRIES = ["China_Drone", "China_MotorBike", "Czech", "India", "Japan", "Norway", "United_States"]

    run_labels = []
    for doc in val_docs:
        label = doc["run_id"]
        if doc.get("is_production"):
            label += " ★"
        run_labels.append(label)

    label_to_doc = dict(zip(run_labels, val_docs))

    selected_labels = st.multiselect("Runs to compare", run_labels, default=run_labels)
    if not selected_labels:
        st.info("Select at least one run.")
        st.stop()

    selected_docs = [label_to_doc[lbl] for lbl in selected_labels]

    # Summary table
    st.subheader("Summary")
    summary_rows = []
    for doc, label in zip(selected_docs, selected_labels):
        ev = doc["metrics"]["evaluation_val"]
        row = {
            "Run": label,
            "Model": doc.get("model", ""),
            "Sample ratio": doc.get("sample_ratio"),
            "F1 overall": ev.get("F1_overall"),
            "Precision": ev.get("precision_overall"),
            "Recall": ev.get("recall_overall"),
            "mAP@0.5": ev.get("mAP50_overall"),
        }
        for c in COUNTRIES:
            row[c] = ev.get("F1_per_country", {}).get(c)
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    fmt_cols = ["F1 overall", "Precision", "Recall", "mAP@0.5"] + COUNTRIES
    for col in fmt_cols:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].apply(
                lambda v: round(v, 4) if v is not None and not pd.isna(v) else None
            )
    st.dataframe(
        df_summary,
        width="stretch",
        hide_index=True,
        column_config={
            "Sample ratio": st.column_config.NumberColumn("Sample ratio", format="%.2f"),
            **{c: st.column_config.NumberColumn(c, format="%.4f") for c in fmt_cols},
        },
    )

    st.divider()

    # F1 per country
    st.subheader("F1 per country")
    country_rows = []
    for doc, label in zip(selected_docs, selected_labels):
        for country, f1 in doc["metrics"]["evaluation_val"].get("F1_per_country", {}).items():
            country_rows.append({"Run": label, "Country": country, "F1": f1})

    if country_rows:
        fig_country = px.bar(
            pd.DataFrame(country_rows),
            x="Country", y="F1", color="Run", barmode="group",
            title="F1 per country", height=400,
        )
        fig_country.update_layout(yaxis_range=[0, 1], xaxis_title=None)
        st.plotly_chart(fig_country, width="stretch")

    st.divider()

    # F1 per class
    st.subheader("F1 per damage class")
    class_rows = []
    for doc, label in zip(selected_docs, selected_labels):
        for cls, f1 in doc["metrics"]["evaluation_val"].get("F1_per_class", {}).items():
            class_rows.append({"Run": label, "Class": cls, "F1": f1})

    if class_rows:
        fig_class = px.bar(
            pd.DataFrame(class_rows),
            x="Class", y="F1", color="Run", barmode="group",
            title="F1 per class  (D00=longitudinal crack, D10=transverse crack, D20=alligator crack, D40=pothole)",
            height=380,
        )
        fig_class.update_layout(yaxis_range=[0, 1], xaxis_title=None)
        st.plotly_chart(fig_class, width="stretch")
    else:
        st.info("No per-class F1 data available.")

    # Overall comparison (only meaningful with multiple runs)
    if len(selected_docs) > 1:
        st.divider()
        st.subheader("F1 overall comparison")
        overall_rows = [
            {"Run": label, "F1": doc["metrics"]["evaluation_val"].get("F1_overall", 0)}
            for doc, label in zip(selected_docs, selected_labels)
        ]
        df_overall = pd.DataFrame(overall_rows).sort_values("F1", ascending=True)
        fig_overall = px.bar(
            df_overall, x="F1", y="Run", orientation="h",
            title="F1 overall (CRDDC2022)",
            height=max(250, len(df_overall) * 45),
        )
        fig_overall.update_layout(xaxis_range=[0, 1], yaxis_title=None)
        st.plotly_chart(fig_overall, width="stretch")

    # -----------------------------------------------------------------------
    # Detailed error analysis (single run) — R2 detection-native metrics
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("Detailed error analysis (single run)")
    detail_label = st.selectbox("Run for the detailed view", selected_labels, key="val_detail")
    ev = label_to_doc[detail_label]["metrics"]["evaluation_val"]

    # Per-class precision / recall / AP table.
    per_class = ev.get("per_class") or {}
    if per_class:
        st.markdown("**Per-class precision / recall / AP**  (IoU ≥ 0.5, conf = 0.5)")
        pc_rows = []
        for cls in CLASS_NAMES:
            m = per_class.get(cls)
            if not m:
                continue
            pc_rows.append({
                "Class": cls,
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "AP@0.5": m.get("AP50"),
                "AP@0.5:0.95": m.get("AP50_95"),
                "F1": m.get("F1"),
            })
        num_cols = ["Precision", "Recall", "AP@0.5", "AP@0.5:0.95", "F1"]
        st.dataframe(
            pd.DataFrame(pc_rows), hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in num_cols},
        )
    else:
        st.info(
            "No per-class P/R/AP recorded for this run. Re-run "
            "`python -m src.evaluation.evaluate` to populate the R2 metrics."
        )

    col_cm, col_pr = st.columns(2)

    # Confusion matrix heatmap.
    with col_cm:
        st.markdown("**Confusion matrix**")
        cm = ev.get("confusion_matrix")
        if cm and cm.get("matrix"):
            labels = cm["labels"]
            mat = cm["matrix"]
            fig_cm = go.Figure(data=go.Heatmap(
                z=mat,
                x=[f"true {lbl}" for lbl in labels],
                y=[f"pred {lbl}" for lbl in labels],
                colorscale="Blues", text=mat, texttemplate="%{text}", showscale=False,
            ))
            fig_cm.update_layout(
                height=420, yaxis_autorange="reversed",
                xaxis_title="ground truth", yaxis_title="predicted",
                title=f"conf={cm.get('conf')}, IoU match={cm.get('iou_thres')}",
            )
            st.plotly_chart(fig_cm, width="stretch")
            st.caption(cm.get("note", ""))
        else:
            st.info("No confusion matrix recorded for this run.")

    # Precision–Recall curves per class.
    with col_pr:
        st.markdown("**Precision–Recall curves** (per class)")
        pr = ev.get("pr_curves")
        if pr and pr.get("per_class"):
            fig_pr = go.Figure()
            for cls, c in pr["per_class"].items():
                fig_pr.add_trace(go.Scatter(
                    x=c.get("recall", []), y=c.get("precision", []), mode="lines", name=cls,
                ))
                if c.get("recall_at_op") is not None:
                    fig_pr.add_trace(go.Scatter(
                        x=[c["recall_at_op"]], y=[c["precision_at_op"]],
                        mode="markers", marker=dict(size=10, symbol="x", color="black"),
                        showlegend=False, hovertext=f"{cls} @ conf={pr.get('operating_point_conf')}",
                    ))
            fig_pr.update_layout(
                height=420, xaxis_title="Recall", yaxis_title="Precision",
                xaxis_range=[0, 1], yaxis_range=[0, 1],
                title=f"×  = operating point at conf={pr.get('operating_point_conf')}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_pr, width="stretch")
        else:
            st.info("No PR curve data recorded for this run.")

    # False-positive / false-negative breakdown.
    fp_fn = (ev.get("fp_fn") or {}).get("overall") or {}
    if fp_fn:
        st.markdown("**False-positive / false-negative breakdown**  (from the confusion matrix, conf = 0.5)")
        ff_rows = []
        for cls in CLASS_NAMES:
            e = fp_fn.get(cls)
            if not e:
                continue
            ff_rows.append({"Class": cls, **{k: e.get(k) for k in ("TP", "FP", "FN", "FP_background", "FN_missed")}})
        df_ff = pd.DataFrame(ff_rows)
        melt = df_ff.melt(id_vars="Class", value_vars=["TP", "FP", "FN"], var_name="Kind", value_name="Count")
        fig_ff = px.bar(
            melt, x="Class", y="Count", color="Kind", barmode="group", height=360,
            title="TP / FP / FN per class",
        )
        st.plotly_chart(fig_ff, width="stretch")
        st.dataframe(df_ff, hide_index=True, width="stretch")

        per_country_ff = (ev.get("fp_fn") or {}).get("per_country") or {}
        if per_country_ff:
            with st.expander("FP / FN per country"):
                rows = []
                for ctry, classes in sorted(per_country_ff.items()):
                    for cls, e in classes.items():
                        rows.append({"Country": ctry, "Class": cls, **{k: e.get(k) for k in ("TP", "FP", "FN")}})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # Localization error — IoU distribution of class-correct detections.
    loc = ev.get("localization") or {}
    if loc.get("n_class_correct"):
        st.markdown("**Localization error**  (IoU of class-correct detections vs ground truth, conf = 0.5)")
        lo, hi = loc.get("poor_box_iou_range", [0.1, 0.5])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean IoU", f"{loc.get('mean_iou', 0):.3f}")
        m2.metric("Median IoU", f"{loc.get('median_iou', 0):.3f}")
        m3.metric(f"Poor-box share (IoU<{hi})", f"{loc.get('poor_box_share', 0):.1%}")
        m4.metric("Class-correct dets", f"{loc.get('n_class_correct', 0)}")

        hist = loc.get("iou_histogram") or {}
        edges = hist.get("bin_edges") or []
        counts = hist.get("counts") or []
        if edges and counts:
            centers = [round((edges[i] + edges[i + 1]) / 2, 2) for i in range(len(counts))]
            colors = ["#d62728" if edges[i + 1] <= hi else "#2ca02c" for i in range(len(counts))]
            fig_loc = go.Figure(go.Bar(x=centers, y=counts, marker_color=colors))
            fig_loc.update_layout(
                height=340, xaxis_title="best same-class IoU", yaxis_title="detections",
                title=f"IoU distribution  (red = poor box, IoU in [{lo}, {hi}))",
                bargap=0.05,
            )
            st.plotly_chart(fig_loc, width="stretch")

        loc_pc = loc.get("per_class") or {}
        if loc_pc:
            st.dataframe(
                pd.DataFrame([
                    {"Class": c, "Class-correct dets": d.get("n"), "Poor boxes": d.get("poor"),
                     "Poor-box share": d.get("poor_box_share"), "Mean IoU": d.get("mean_iou")}
                    for c, d in sorted(loc_pc.items())
                ]),
                hide_index=True, width="stretch",
                column_config={
                    "Poor-box share": st.column_config.NumberColumn("Poor-box share", format="%.3f"),
                    "Mean IoU": st.column_config.NumberColumn("Mean IoU", format="%.3f"),
                },
            )
        n_capped = sum(1 for c in (loc.get("per_country_cap") or {}).values() if c.get("available", 0) > c.get("used", 0))
        if n_capped:
            st.caption(
                f"Sampled up to {loc.get('sample_per_country')} images/country for this pass "
                f"({n_capped} countries were capped). Increase via `--loc-sample`."
            )


# ---------------------------------------------------------------------------
# Page: Run Detail
# ---------------------------------------------------------------------------

elif page == "Run Detail":
    st.title("Run Detail")

    if df_all.empty:
        st.warning("No experiments in MongoDB.")
        st.stop()

    run_options = df_all["run_id"].tolist()
    prod_ids = df_all[df_all["is_production"] == True]["run_id"].tolist()  # noqa: E712
    default_idx = run_options.index(prod_ids[0]) if prod_ids else 0

    selected_run = st.selectbox("Select run", run_options, index=default_idx)
    run_row = df_all[df_all["run_id"] == selected_run].iloc[0]

    # Header metrics
    badge = "[PRODUCTION]" if run_row["is_production"] else f"[{run_row['status']}]"
    st.subheader(f"{selected_run}  {badge}")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("F1", _fmt_pct(run_row["F1"]))
    mc2.metric("mAP@0.5", _fmt_pct(run_row["mAP50"]))
    mc3.metric("Precision", _fmt_pct(run_row["precision"]))
    mc4.metric("Recall", _fmt_pct(run_row["recall"]))

    # Hyperparams
    with st.expander("Hyperparameters", expanded=False):
        hp = run_row["hyperparams"]
        if hp:
            cols = st.columns(4)
            for i, (k, v) in enumerate(hp.items()):
                cols[i % 4].metric(k, str(v))
        else:
            st.write("No hyperparameter data.")

    # Checkpoints
    ckpts = run_row.get("checkpoints") or {}
    if ckpts:
        with st.expander("Checkpoint URLs (Backblaze B2)", expanded=False):
            for k, url in ckpts.items():
                st.markdown(f"**{k}**: `{url}`")

    st.divider()

    # Training curves from results.csv
    st.subheader("Training curves")
    results_url = (run_row.get("checkpoints") or {}).get("results_csv")
    df_csv = load_results_csv(selected_run, results_url)

    if df_csv is None:
        st.info(
            f"No results.csv available for {selected_run} "
            f"(checked runs/train/{selected_run}/results.csv and B2)."
        )
    else:
        # Map readable names
        col_map = {
            "epoch": "Epoch",
            "train/box_loss": "Train box loss",
            "train/cls_loss": "Train cls loss",
            "train/dfl_loss": "Train dfl loss",
            "metrics/precision(B)": "Precision",
            "metrics/recall(B)": "Recall",
            "metrics/mAP50(B)": "mAP@0.5",
            "metrics/mAP50-95(B)": "mAP@0.5:0.95",
            "val/box_loss": "Val box loss",
            "val/cls_loss": "Val cls loss",
            "val/dfl_loss": "Val dfl loss",
        }
        df_csv = df_csv.rename(columns={k: v for k, v in col_map.items() if k in df_csv.columns})

        if len(df_csv) < 2:
            st.warning(
                f"This run only has {len(df_csv)} epoch recorded — curves render "
                "as a single marker. Useful for verifying that training started, "
                "but not for tracking convergence."
            )
        _curve_mode = "lines+markers" if len(df_csv) >= 2 else "markers"

        tab1, tab2, tab3 = st.tabs(["Metrics", "Losses", "Learning rate"])

        with tab1:
            metric_cols = [c for c in ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"] if c in df_csv.columns]
            if metric_cols and "Epoch" in df_csv.columns:
                fig = go.Figure()
                for col in metric_cols:
                    fig.add_trace(go.Scatter(x=df_csv["Epoch"], y=df_csv[col], name=col, mode=_curve_mode))
                fig.update_layout(
                    title="Val metrics per epoch",
                    xaxis_title="Epoch",
                    yaxis_title="Value",
                    yaxis_range=[0, 1],
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, width="stretch")

        with tab2:
            loss_cols_train = [c for c in ["Train box loss", "Train cls loss", "Train dfl loss"] if c in df_csv.columns]
            loss_cols_val = [c for c in ["Val box loss", "Val cls loss", "Val dfl loss"] if c in df_csv.columns]
            if (loss_cols_train or loss_cols_val) and "Epoch" in df_csv.columns:
                fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Train losses", "Val losses"))
                for col in loss_cols_train:
                    fig2.add_trace(go.Scatter(x=df_csv["Epoch"], y=df_csv[col], name=col, mode=_curve_mode), row=1, col=1)
                for col in loss_cols_val:
                    fig2.add_trace(go.Scatter(x=df_csv["Epoch"], y=df_csv[col], name=col, mode=_curve_mode), row=1, col=2)
                fig2.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig2, width="stretch")

        with tab3:
            lr_cols = [c for c in df_csv.columns if c.startswith("lr/")]
            if lr_cols and "Epoch" in df_csv.columns:
                fig3 = go.Figure()
                for col in lr_cols:
                    fig3.add_trace(go.Scatter(x=df_csv["Epoch"], y=df_csv[col], name=col, mode=_curve_mode))
                fig3.update_layout(title="Learning rate schedule", xaxis_title="Epoch", height=300)
                st.plotly_chart(fig3, width="stretch")
            else:
                st.info("No learning rate data in results.csv.")

        with st.expander("Raw results.csv", expanded=False):
            st.dataframe(df_csv, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# Page: MLflow
# ---------------------------------------------------------------------------

elif page == "MLflow":
    st.title("MLflow")

    with st.spinner("Loading MLflow runs..."):
        df_mlflow = load_mlflow_runs()

    if df_mlflow.empty:
        st.warning(f"No MLflow runs found at `{MLFLOW_URI}`.")
        st.info("Runs are logged automatically during training. Make sure `mlruns/` exists in the repo root.")
        st.stop()

    st.caption(f"Tracking URI: `{MLFLOW_URI}`")

    # Table of runs
    display_mlflow = df_mlflow[["run_name", "experiment", "status"]].copy()
    for col in ["params", "metrics"]:
        display_mlflow[col] = df_mlflow[col].apply(lambda d: ", ".join(f"{k}={v}" for k, v in d.items()) if d else "")
    st.dataframe(display_mlflow, width="stretch", hide_index=True)

    st.divider()

    # Run detail
    st.subheader("Run detail")
    options = df_mlflow[df_mlflow["run_name"].notna()][["mlflow_run_id", "run_name"]].drop_duplicates("mlflow_run_id")
    id_to_name = dict(zip(options["mlflow_run_id"], options["run_name"]))
    selected_mlflow_id = st.selectbox(
        "Select MLflow run",
        options["mlflow_run_id"].tolist(),
        format_func=lambda rid: id_to_name.get(rid, rid),
    )
    row = df_mlflow[df_mlflow["mlflow_run_id"] == selected_mlflow_id].iloc[0]

    col_p, col_m = st.columns(2)
    with col_p:
        st.markdown("**Parameters**")
        if row["params"]:
            st.dataframe(
                pd.DataFrame([{"param": k, "value": v} for k, v in row["params"].items()]),
                hide_index=True,
                width="stretch",
            )
    with col_m:
        st.markdown("**Metrics**")
        if row["metrics"]:
            st.dataframe(
                pd.DataFrame([{"metric": k, "value": v} for k, v in row["metrics"].items()]),
                hide_index=True,
                width="stretch",
            )

    st.divider()
    st.info(
        "For full MLflow UI (per-epoch curves logged by Ultralytics' built-in callback): "
        "run `mlflow ui --backend-store-uri mlruns` in the repo root and open http://localhost:5000"
    )
