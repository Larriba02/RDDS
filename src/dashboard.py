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


@st.cache_data(ttl=60, show_spinner=False)
def load_results_csv(run_id: str) -> pd.DataFrame | None:
    csv_path = RUNS_DIR / run_id / "results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


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
        ["Overview", "Experiments", "Run Detail", "MLflow"],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("Refresh data", use_container_width=True):
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
                    [{"param": k, "value": v} for k, v in hp.items()]
                )
                st.dataframe(hp_df, use_container_width=True, hide_index=True)
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
        st.plotly_chart(fig, use_container_width=True)
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
        use_container_width=True,
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig2, use_container_width=True)

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
    df_csv = load_results_csv(selected_run)

    if df_csv is None:
        st.info(f"No results.csv found at runs/train/{selected_run}/results.csv")
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

        tab1, tab2, tab3 = st.tabs(["Metrics", "Losses", "Learning rate"])

        with tab1:
            metric_cols = [c for c in ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"] if c in df_csv.columns]
            if metric_cols and "Epoch" in df_csv.columns:
                fig = go.Figure()
                for col in metric_cols:
                    fig.add_trace(go.Scatter(x=df_csv["Epoch"], y=df_csv[col], name=col, mode="lines"))
                fig.update_layout(
                    title="Val metrics per epoch",
                    xaxis_title="Epoch",
                    yaxis_title="Value",
                    yaxis_range=[0, 1],
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            loss_cols_train = [c for c in ["Train box loss", "Train cls loss", "Train dfl loss"] if c in df_csv.columns]
            loss_cols_val = [c for c in ["Val box loss", "Val cls loss", "Val dfl loss"] if c in df_csv.columns]
            if (loss_cols_train or loss_cols_val) and "Epoch" in df_csv.columns:
                fig2 = make_subplots(rows=1, cols=2, subplot_titles=("Train losses", "Val losses"))
                for col in loss_cols_train:
                    fig2.add_trace(go.Scatter(x=df_csv["Epoch"], y=df_csv[col], name=col, mode="lines"), row=1, col=1)
                for col in loss_cols_val:
                    fig2.add_trace(go.Scatter(x=df_csv["Epoch"], y=df_csv[col], name=col, mode="lines"), row=1, col=2)
                fig2.update_layout(height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            lr_cols = [c for c in df_csv.columns if c.startswith("lr/")]
            if lr_cols and "Epoch" in df_csv.columns:
                fig3 = go.Figure()
                for col in lr_cols:
                    fig3.add_trace(go.Scatter(x=df_csv["Epoch"], y=df_csv[col], name=col, mode="lines"))
                fig3.update_layout(title="Learning rate schedule", xaxis_title="Epoch", height=300)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No learning rate data in results.csv.")

        with st.expander("Raw results.csv", expanded=False):
            st.dataframe(df_csv, use_container_width=True, hide_index=True)

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
    st.dataframe(display_mlflow, use_container_width=True, hide_index=True)

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
                use_container_width=True,
            )
    with col_m:
        st.markdown("**Metrics**")
        if row["metrics"]:
            st.dataframe(
                pd.DataFrame([{"metric": k, "value": v} for k, v in row["metrics"].items()]),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()
    st.info(
        "For full MLflow UI (per-epoch curves logged by Ultralytics' built-in callback): "
        "run `mlflow ui --backend-store-uri mlruns` in the repo root and open http://localhost:5000"
    )
