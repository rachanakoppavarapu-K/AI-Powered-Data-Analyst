"""Report tab component — generate, preview, and download the HTML report."""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.reporting.report_generator import ReportGenerator


def render_report_tab(
    df: pd.DataFrame | None,
    filename: str | None,
    profile: dict[str, Any] | None,
    eda_result: dict[str, Any] | None,
    best_metrics: Any | None,
    best_result: Any | None,
    nl_query_results: list[Any],
) -> None:
    """Render the Report tab — generate, preview, and download the HTML report.

    Parameters
    ----------
    df:
        The loaded DataFrame, or ``None``.
    filename:
        Name of the uploaded file, or ``None``.
    profile:
        Pre-computed profiling dict, or ``None``.
    eda_result:
        Pre-computed EDA dict, or ``None``.
    best_metrics:
        Best :class:`~src.ml.evaluator.ModelMetrics`, or ``None``.
    best_result:
        Best :class:`~src.ml.trainer.TrainResult`, or ``None``.
    nl_query_results:
        List of :class:`~src.nl_query.nl_query_engine.NLQueryResult` objects.
    """
    st.header("📄 Report")

    if df is None:
        st.info("Upload a dataset to generate a report.")
        return

    st.markdown(
        "Generate a self-contained HTML report summarising all pipeline outputs. "
        "The report includes the data profile, EDA findings, ML evaluation, and "
        "any AI-generated insights produced in this session."
    )

    # ------------------------------------------------------------------
    # Configuration options
    # ------------------------------------------------------------------
    with st.expander("Report options", expanded=True):
        report_title = st.text_input(
            "Report title",
            value=f"Data Analysis Report — {filename or 'Dataset'}",
        )
        include_sections = st.multiselect(
            "Sections to include",
            options=["Data Profile", "EDA", "ML Results", "Insights", "NL Queries"],
            default=["Data Profile", "EDA", "ML Results", "Insights", "NL Queries"],
        )

    # ------------------------------------------------------------------
    # Generate button
    # ------------------------------------------------------------------
    if st.button("📋 Generate Report", key="btn_generate_report"):
        with st.spinner("Assembling report…"):
            context = _build_report_context(
                df=df,
                filename=filename,
                report_title=report_title,
                include_sections=include_sections,
                profile=profile,
                eda_result=eda_result,
                best_metrics=best_metrics,
                best_result=best_result,
                nl_query_results=nl_query_results,
            )
            try:
                generator = ReportGenerator(output_dir=tempfile.gettempdir())
                html_path = generator.generate(context)
                st.session_state["report_html_path"] = html_path
                st.session_state["report_html_str"] = Path(html_path).read_text(encoding="utf-8")
                st.success(f"✅ Report generated: `{html_path}`")
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Report generation failed: {exc}")
                return

    # ------------------------------------------------------------------
    # Preview & Download (shown when a report has been generated)
    # ------------------------------------------------------------------
    html_str: str | None = st.session_state.get("report_html_str")
    html_path: str | None = st.session_state.get("report_html_path")

    if html_str:
        st.markdown("---")

        # Download button
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"report_{ts}.html"
        st.download_button(
            label="⬇️ Download HTML Report",
            data=html_str.encode("utf-8"),
            file_name=download_name,
            mime="text/html",
            key="btn_download_report",
        )

        # Preview (in a collapsible expander to avoid cluttering the page)
        with st.expander("📖 Preview Report", expanded=False):
            st.components.v1.html(html_str, height=800, scrolling=True)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def _build_report_context(
    *,
    df: pd.DataFrame,
    filename: str | None,
    report_title: str,
    include_sections: list[str],
    profile: dict[str, Any] | None,
    eda_result: dict[str, Any] | None,
    best_metrics: Any | None,
    best_result: Any | None,
    nl_query_results: list[Any],
) -> dict[str, Any]:
    """Assemble the context dict for ReportGenerator."""
    context: dict[str, Any] = {
        "title": report_title,
        "dataset_name": filename or "Uploaded Dataset",
    }

    if "Data Profile" in include_sections and profile:
        context["profile"] = profile
        shape = profile.get("shape", {})
        n_rows = shape.get("rows", 0)
        n_cols = shape.get("columns", 0)
        null_counts = profile.get("null_counts", {})
        context["data_quality"] = {
            "null_percentages": profile.get("null_percentages", {}),
            "outlier_info": {},
            "notes": [
                f"Dataset has {n_rows:,} rows and {n_cols} columns.",
                f"Total missing cells: {sum(null_counts.values()):,}",
            ],
        }

    if "EDA" in include_sections and eda_result:
        bivariate = eda_result.get("bivariate", {})
        corr_pearson = bivariate.get("correlation", {}).get("pearson", {})
        top_corrs = _extract_top_correlations(corr_pearson)

        univariate = eda_result.get("univariate", {})
        numeric_uni = univariate.get("numeric", {})
        skewed = [
            {"column": col, "skewness": stats.get("skewness", 0)}
            for col, stats in numeric_uni.items()
            if stats.get("skewness") is not None and abs(stats.get("skewness", 0)) > 1
        ]

        context["eda"] = {
            "top_correlations": top_corrs,
            "skewed_columns": skewed,
            "charts": [],
        }

    if "ML Results" in include_sections and best_metrics:
        ml_dict = best_metrics.to_dict()
        context["ml_metrics"] = ml_dict

        if best_result is not None:
            try:
                from src.ml.feature_importance import compute_feature_importance  # noqa: PLC0415
                from src.visualization.chart_builder import feature_importance_chart  # noqa: PLC0415

                imp = compute_feature_importance(best_result, method="auto", n_repeats=5)
                imp_entries = [
                    {"feature": e.feature, "importance": e.importance}
                    for e in imp.ranked[:20]
                ]
                names = [e.feature for e in imp.ranked[:20]]
                scores = [e.importance for e in imp.ranked[:20]]
                imp_fig = feature_importance_chart(names, scores, top_n=20)
                context["feature_importance"] = {
                    "entries": imp_entries,
                    "chart": imp_fig,
                }
            except Exception:
                pass

    if "Insights" in include_sections:
        # Include any cached GenAI insights from session state
        genai_insights: dict[str, str] = {}
        for section in ["Dataset Summary", "Key Findings", "EDA Explanation", "Model Explanation", "Recommendations"]:
            cached = st.session_state.get(f"genai_{section}")
            if cached:
                genai_insights[section] = cached
        if genai_insights:
            context["genai_insights"] = genai_insights

    if "NL Queries" in include_sections and nl_query_results:
        context["nl_query_results"] = nl_query_results

    return context


def _extract_top_correlations(pearson: dict[str, Any]) -> list[dict[str, Any]]:
    """Return top correlation pairs from a nested Pearson dict."""
    seen: set[frozenset[str]] = set()
    pairs = []
    for col_a, others in pearson.items():
        for col_b, val in others.items():
            if col_a == col_b:
                continue
            key = frozenset({col_a, col_b})
            if key in seen:
                continue
            seen.add(key)
            if val is not None:
                pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": val})
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs[:10]
