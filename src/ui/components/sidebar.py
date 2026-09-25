"""Sidebar component — file upload, target column, task type, and model selectors."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.ingestion.data_loader import DataLoader, DataLoadError
from src.ml.task_detector import TaskType


def render_sidebar() -> dict[str, Any]:
    """Render the sidebar and return the current user selections.

    Returns
    -------
    dict with keys:
        ``df``            — loaded :class:`pd.DataFrame` or ``None``
        ``filename``      — uploaded filename or ``None``
        ``target_column`` — selected target column or ``None``
        ``task_type``     — :class:`TaskType` (user override) or ``None``
        ``model_names``   — list[str] of selected model names (empty = all)
    """
    st.sidebar.title("⚙️ Configuration")

    # ------------------------------------------------------------------
    # 1. File upload
    # ------------------------------------------------------------------
    st.sidebar.header("1. Upload Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        help="Maximum 50 MB. Supported formats: CSV, Excel (.xlsx, .xls).",
    )

    df: pd.DataFrame | None = None
    filename: str | None = None

    if uploaded_file is not None:
        filename = uploaded_file.name
        loader = DataLoader()
        try:
            with st.sidebar.spinner("Loading dataset…"):
                df = loader.load(uploaded_file)
            st.sidebar.success(
                f"✅ Loaded **{filename}** — "
                f"{df.shape[0]:,} rows × {df.shape[1]} columns"
            )
        except DataLoadError as exc:
            st.sidebar.error(f"❌ Failed to load file: {exc}")
            df = None
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"❌ Unexpected error: {exc}")
            df = None

    # ------------------------------------------------------------------
    # 2. Target column selector
    # ------------------------------------------------------------------
    target_column: str | None = None

    if df is not None:
        st.sidebar.header("2. Target Column")
        col_options = ["(None — unsupervised)"] + list(df.columns)
        target_sel = st.sidebar.selectbox(
            "Select target column",
            options=col_options,
            index=0,
            help="Choose the column you want to predict. Leave as '(None)' for clustering.",
        )
        if target_sel != "(None — unsupervised)":
            target_column = target_sel

    # ------------------------------------------------------------------
    # 3. Task type override
    # ------------------------------------------------------------------
    task_type: TaskType | None = None

    if df is not None:
        st.sidebar.header("3. Task Type")
        task_options = {
            "Auto-detect": None,
            "Classification": TaskType.CLASSIFICATION,
            "Regression": TaskType.REGRESSION,
            "Clustering": TaskType.CLUSTERING,
        }
        task_sel = st.sidebar.selectbox(
            "ML task type",
            options=list(task_options.keys()),
            index=0,
            help="Override the automatically detected task type.",
        )
        task_type = task_options[task_sel]

    # ------------------------------------------------------------------
    # 4. Model selector (classification / regression only)
    # ------------------------------------------------------------------
    model_names: list[str] = []

    if df is not None and task_type != TaskType.CLUSTERING and target_column is not None:
        st.sidebar.header("4. Models")
        available_models = {
            "classification": ["Logistic Regression", "Random Forest", "XGBoost"],
            "regression": ["Linear Regression", "Random Forest Regressor", "XGBoost Regressor"],
        }
        task_key = task_type.value if task_type else "classification"
        if task_key == "clustering":
            task_key = "classification"
        model_choices = available_models.get(task_key, available_models["classification"])
        model_names = st.sidebar.multiselect(
            "Select models to train",
            options=model_choices,
            default=model_choices,
            help="All selected models will be trained and compared.",
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("AI Data Analyst • Powered by IBM watsonx.ai")

    return {
        "df": df,
        "filename": filename,
        "target_column": target_column,
        "task_type": task_type,
        "model_names": model_names,
    }
