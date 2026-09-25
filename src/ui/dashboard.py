"""Main Streamlit dashboard — AI-Powered Intelligent Data Analytics.

Entry point:
    streamlit run src/ui/dashboard.py
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

# Load .env before anything else (no-op if python-dotenv not installed)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass

# ---------------------------------------------------------------------------
# Page configuration — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local component imports (after page config)
# ---------------------------------------------------------------------------
from src.ui.components.eda_tab import render_eda_tab
from src.ui.components.insights_tab import render_insights_tab
from src.ui.components.ml_tab import render_ml_tab
from src.ui.components.report_tab import render_report_tab
from src.ui.components.sidebar import render_sidebar


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Ensure all required session-state keys are initialised."""
    defaults: dict[str, Any] = {
        # Core data
        "df": None,
        "filename": None,
        "target_column": None,
        "task_type": None,
        # Computed artefacts (populated by individual tabs)
        "profile": None,
        "eda_result": None,
        # ML artefacts
        "ml_trained": False,
        "ml_train_results": [],
        "ml_metrics": [],
        "ml_task_type": None,
        # Insights
        "nl_query_history": [],
        # Report
        "report_html_path": None,
        "report_html_str": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Helper: eager profile / EDA computation & caching in session state
# ---------------------------------------------------------------------------

def _compute_profile_if_needed(df: Any) -> dict[str, Any] | None:
    """Return cached profile, recomputing when df changes."""
    if df is None:
        return None
    cached_profile = st.session_state.get("profile")
    # Invalidate cache when the dataframe shape/columns change
    cached_sig = st.session_state.get("_profile_sig")
    current_sig = (df.shape, tuple(df.columns))
    if cached_profile is None or cached_sig != current_sig:
        from src.profiling.data_profiler import profile_dataframe  # noqa: PLC0415

        with st.spinner("Profiling dataset…"):
            cached_profile = profile_dataframe(df)
        st.session_state["profile"] = cached_profile
        st.session_state["_profile_sig"] = current_sig
    return cached_profile


def _compute_eda_if_needed(
    df: Any, target_column: str | None, task_type: Any
) -> dict[str, Any] | None:
    """Return cached EDA result, recomputing when inputs change."""
    if df is None:
        return None
    cached_eda = st.session_state.get("eda_result")
    task_val = task_type.value if task_type else None
    cached_sig = st.session_state.get("_eda_sig")
    current_sig = (df.shape, tuple(df.columns), target_column, task_val)
    if cached_eda is None or cached_sig != current_sig:
        from src.eda.eda_engine import run_eda  # noqa: PLC0415

        with st.spinner("Running EDA…"):
            cached_eda = run_eda(df, target_column=target_column, task_type=task_val)
        st.session_state["eda_result"] = cached_eda
        st.session_state["_eda_sig"] = current_sig
    return cached_eda


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the full Streamlit application."""
    _init_session_state()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.title("🔬 AI-Powered Data Analyst")
    st.caption(
        "Upload a CSV or Excel dataset, explore it with automated EDA, "
        "train ML models, and generate AI-driven insights — powered by IBM watsonx.ai."
    )

    # ------------------------------------------------------------------
    # Sidebar (file upload + configuration)
    # ------------------------------------------------------------------
    sidebar_state = render_sidebar()
    df = sidebar_state["df"]
    filename = sidebar_state["filename"]
    target_column = sidebar_state["target_column"]
    task_type_override = sidebar_state["task_type"]
    model_names = sidebar_state["model_names"]

    # Detect dataset change and reset ML artefacts so stale results aren't shown
    prev_sig = st.session_state.get("_dataset_sig")
    current_sig = (filename, df.shape if df is not None else None)
    if prev_sig != current_sig:
        st.session_state["ml_trained"] = False
        st.session_state["ml_train_results"] = []
        st.session_state["ml_metrics"] = []
        st.session_state["profile"] = None
        st.session_state["eda_result"] = None
        st.session_state["nl_query_history"] = []
        st.session_state["report_html_path"] = None
        st.session_state["report_html_str"] = None
        st.session_state["_dataset_sig"] = current_sig

    # ------------------------------------------------------------------
    # No file uploaded yet — landing state
    # ------------------------------------------------------------------
    if df is None:
        _render_landing()
        return

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    tab_data, tab_eda, tab_ml, tab_insights, tab_report = st.tabs(
        ["📋 Data", "📊 EDA", "🤖 ML", "💡 Insights", "📄 Report"]
    )

    # ---- Data tab (always cheap) ----
    with tab_data:
        _render_data_tab(df, filename)

    # ---- EDA tab ----
    with tab_eda:
        profile = _compute_profile_if_needed(df)
        eda_result = _compute_eda_if_needed(df, target_column, task_type_override)
        task_str = task_type_override.value if task_type_override else None
        render_eda_tab(df, target_column=target_column, task_type=task_str)

    # ---- ML tab ----
    with tab_ml:
        ml_state = render_ml_tab(
            df,
            target_column=target_column,
            task_type_override=task_type_override,
            selected_model_names=model_names,
        )
        best_result = ml_state.get("best_result")
        best_metrics = ml_state.get("best_metrics")

    # ---- Insights tab ----
    with tab_insights:
        profile = st.session_state.get("profile")
        eda_result = st.session_state.get("eda_result")
        best_result_ins = st.session_state.get("ml_train_results", [None])[0] if st.session_state.get("ml_trained") else None
        best_metrics_ins = st.session_state.get("ml_metrics", [None])[0] if st.session_state.get("ml_trained") else None

        # Use the best from the ML tab if available
        ml_results = st.session_state.get("ml_train_results", [])
        ml_metrics_list = st.session_state.get("ml_metrics", [])
        if ml_results and ml_metrics_list:
            from src.ui.components.ml_tab import _pick_best_model  # noqa: PLC0415
            ml_task = st.session_state.get("ml_task_type")
            if ml_task:
                best_result_ins, best_metrics_ins = _pick_best_model(ml_results, ml_metrics_list, ml_task)

        nl_history = render_insights_tab(
            df=df,
            profile=profile,
            eda_result=eda_result,
            best_metrics=best_metrics_ins,
            best_result=best_result_ins,
        )

    # ---- Report tab ----
    with tab_report:
        report_profile = st.session_state.get("profile")
        report_eda = st.session_state.get("eda_result")
        report_ml_results = st.session_state.get("ml_train_results", [])
        report_ml_metrics = st.session_state.get("ml_metrics", [])
        report_best_result = None
        report_best_metrics = None
        if report_ml_results and report_ml_metrics:
            from src.ui.components.ml_tab import _pick_best_model  # noqa: PLC0415
            report_task = st.session_state.get("ml_task_type")
            if report_task:
                report_best_result, report_best_metrics = _pick_best_model(
                    report_ml_results, report_ml_metrics, report_task
                )

        render_report_tab(
            df=df,
            filename=filename,
            profile=report_profile,
            eda_result=report_eda,
            best_metrics=report_best_metrics,
            best_result=report_best_result,
            nl_query_results=st.session_state.get("nl_query_history", []),
        )


# ---------------------------------------------------------------------------
# Data tab renderer
# ---------------------------------------------------------------------------


def _render_data_tab(df: Any, filename: str | None) -> None:
    """Render the raw data preview tab."""
    st.header("📋 Dataset Preview")

    if filename:
        st.caption(f"File: **{filename}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

    st.markdown("#### First 100 rows")
    st.dataframe(df.head(100), use_container_width=True)

    with st.expander("Column info"):
        col_info = []
        for col in df.columns:
            col_info.append(
                {
                    "Column": col,
                    "Dtype": str(df[col].dtype),
                    "Non-null": int(df[col].notna().sum()),
                    "Null": int(df[col].isna().sum()),
                    "Unique": int(df[col].nunique()),
                    "Sample": str(df[col].dropna().iloc[0]) if df[col].notna().any() else "—",
                }
            )
        import pandas as pd  # noqa: PLC0415

        st.dataframe(pd.DataFrame(col_info), use_container_width=True)


# ---------------------------------------------------------------------------
# Landing page (no file uploaded)
# ---------------------------------------------------------------------------


def _render_landing() -> None:
    """Render the welcome / landing state."""
    st.markdown("---")
    st.markdown(
        """
        ## Welcome! 👋

        This application provides end-to-end data analytics in five steps:

        | Tab | What it does |
        |---|---|
        | **📋 Data** | Preview the raw dataset and column info |
        | **📊 EDA** | Automated profiling, statistics, and interactive charts |
        | **🤖 ML** | Train & compare ML models; evaluate metrics & feature importance |
        | **💡 Insights** | Ask natural-language questions; generate AI narrative insights |
        | **📄 Report** | Build and download a self-contained HTML report |

        ### Getting Started
        1. **Upload a CSV or Excel file** using the sidebar on the left.
        2. *(Optional)* Select a **target column** if you want supervised ML.
        3. *(Optional)* Override the **task type** (classification / regression / clustering).
        4. Navigate through the tabs.

        ### Sample Datasets
        Three sample datasets are available in the `datasets/` folder:
        - `titanic_like.csv` — classification (survival prediction)
        - `house_prices.csv` — regression (price prediction)
        - `customers.csv` — clustering / mixed analysis
        """
    )

    st.info(
        "🔑 **GenAI Insights** require IBM watsonx.ai credentials. "
        "Set `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` in your environment or `.env` file. "
        "All deterministic analytics work without credentials."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
