"""Insights tab component — NL query, GenAI insights, and verified analysis."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import streamlit as st

from src.genai.prompt_builder import (
    build_dataset_summary_prompt,
    build_eda_explanation_prompt,
    build_key_findings_prompt,
    build_model_explanation_prompt,
    build_recommendations_prompt,
)
from src.genai.watsonx_client import WatsonxClient
from src.nl_query.nl_query_engine import NLQueryEngine


def _get_watsonx_client() -> WatsonxClient | None:
    """Instantiate WatsonxClient, returning None when credentials are absent."""
    client = WatsonxClient()
    if not client.is_configured:
        return None
    return client


def render_insights_tab(
    df: pd.DataFrame,
    profile: dict[str, Any] | None,
    eda_result: dict[str, Any] | None,
    best_metrics: Any | None,
    best_result: Any | None,
) -> list[Any]:
    """Render the AI Insights tab.

    Parameters
    ----------
    df:
        The loaded DataFrame.
    profile:
        Pre-computed profiling dict (from ``DataProfiler.profile()``), or ``None``.
    eda_result:
        Pre-computed EDA dict (from ``run_eda()``), or ``None``.
    best_metrics:
        Best :class:`~src.ml.evaluator.ModelMetrics`, or ``None``.
    best_result:
        Best :class:`~src.ml.trainer.TrainResult`, or ``None``.

    Returns
    -------
    list of :class:`~src.nl_query.nl_query_engine.NLQueryResult`
        NL query results accumulated during this session.
    """
    st.header("💡 AI Insights")

    # ------------------------------------------------------------------
    # GenAI availability check
    # ------------------------------------------------------------------
    client = _get_watsonx_client()
    genai_available = client is not None

    if not genai_available:
        st.warning(
            "⚠️ **GenAI unavailable** — IBM watsonx.ai credentials are not configured.\n\n"
            "Set `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` environment variables to enable "
            "AI-generated narrative insights. Deterministic computed results are still shown."
        )

    # ------------------------------------------------------------------
    # 1. Natural Language Query (deterministic + optional LLM explanation)
    # ------------------------------------------------------------------
    st.subheader("🔍 Ask a Question About Your Data")
    st.caption(
        "Questions are answered deterministically from the data. "
        "If GenAI is available, an additional plain-language explanation is generated."
    )

    nl_history: list[Any] = st.session_state.get("nl_query_history", [])

    with st.form("nl_query_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What is the average age? How many rows? What is the distribution of salary?",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Querying data…"):
            engine = NLQueryEngine(df, watsonx_client=client)
            result = engine.ask(question)
        nl_history.append(result)
        st.session_state["nl_query_history"] = nl_history

    if nl_history:
        for res in reversed(nl_history[-5:]):  # show last 5
            _render_nl_result(res)

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2. GenAI narrative insights (only when GenAI is available)
    # ------------------------------------------------------------------
    st.subheader("📝 AI-Generated Insights")

    if not genai_available:
        st.info("Configure watsonx.ai credentials to generate narrative insights.")
    else:
        _render_genai_insights(client, profile, eda_result, best_metrics, best_result)

    return nl_history


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_nl_result(result: Any) -> None:
    """Render a single NL query result."""
    from src.nl_query.nl_query_engine import NLQueryResult  # local import avoids cycles

    with st.container():
        st.markdown(f"**Q: {result.question}**")
        if result.error:
            st.error(f"Error: {result.error}")
        else:
            st.markdown(f"*Query type:* `{result.query_type}`")
            if result.computed_result is not None:
                if isinstance(result.computed_result, dict):
                    rows = [{"Key": k, "Value": v} for k, v in result.computed_result.items()]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                elif isinstance(result.computed_result, list):
                    if result.computed_result and isinstance(result.computed_result[0], dict):
                        st.dataframe(pd.DataFrame(result.computed_result), use_container_width=True)
                    else:
                        st.write(result.computed_result)
                else:
                    st.write(result.computed_result)
            if result.explanation:
                st.success(result.explanation)
        st.divider()


def _render_genai_insights(
    client: WatsonxClient,
    profile: dict[str, Any] | None,
    eda_result: dict[str, Any] | None,
    best_metrics: Any | None,
    best_result: Any | None,
) -> None:
    """Render GenAI insight sections using pre-computed numbers."""

    sections = [
        "Dataset Summary",
        "Key Findings",
        "EDA Explanation",
        "Model Explanation",
        "Recommendations",
    ]

    selected_section = st.selectbox(
        "Generate insight for:",
        options=sections,
        key="genai_section_sel",
    )

    if st.button("✨ Generate Insight", key="btn_gen_insight"):
        insight_text = _generate_section(
            selected_section, client, profile, eda_result, best_metrics, best_result
        )
        if insight_text:
            st.session_state[f"genai_{selected_section}"] = insight_text
        else:
            st.warning("GenAI returned no response. Please check credentials or try again.")

    cached = st.session_state.get(f"genai_{selected_section}")
    if cached:
        st.markdown(f"**{selected_section}**")
        st.info(cached)


def _generate_section(
    section: str,
    client: WatsonxClient,
    profile: dict[str, Any] | None,
    eda_result: dict[str, Any] | None,
    best_metrics: Any | None,
    best_result: Any | None,
) -> str | None:
    """Build a prompt for *section* and call the watsonx client."""
    with st.spinner(f"Generating {section}…"):
        try:
            prompt = _build_prompt(section, profile, eda_result, best_metrics, best_result)
            if prompt is None:
                st.warning(f"Cannot generate '{section}': required data is not yet available. "
                           "Please run the EDA and/or ML pipeline first.")
                return None
            return client.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            st.error(f"GenAI call failed: {exc}")
            return None


def _build_prompt(
    section: str,
    profile: dict[str, Any] | None,
    eda_result: dict[str, Any] | None,
    best_metrics: Any | None,
    best_result: Any | None,
) -> str | None:
    """Return the appropriate prompt for *section*, or ``None`` when data is missing."""

    if section == "Dataset Summary":
        if profile is None:
            return None
        return build_dataset_summary_prompt(_flatten_profile(profile))

    if section == "Key Findings":
        if profile is None or eda_result is None:
            return None
        return build_key_findings_prompt(
            _flatten_profile(profile),
            _flatten_eda(eda_result),
        )

    if section == "EDA Explanation":
        if eda_result is None:
            return None
        return build_eda_explanation_prompt(_flatten_eda(eda_result))

    if section == "Model Explanation":
        if best_metrics is None:
            return None
        metrics_dict = best_metrics.to_dict()
        imp_list: list[dict] = []
        if best_result is not None:
            try:
                from src.ml.feature_importance import compute_feature_importance  # noqa: PLC0415

                imp = compute_feature_importance(best_result, method="auto", n_repeats=5)
                imp_list = [{"feature": e.feature, "importance": e.importance} for e in imp.ranked[:10]]
            except Exception:
                pass
        metrics_dict["n_samples"] = len(best_result.X_train) if best_result else "N/A"
        metrics_dict["n_features"] = len(best_result.feature_names) if best_result else "N/A"
        return build_model_explanation_prompt(metrics_dict, imp_list)

    if section == "Recommendations":
        context: dict[str, Any] = {}
        if profile:
            context["dataset_summary"] = _flatten_profile(profile)
        if eda_result:
            corr_data = _flatten_eda(eda_result)
            highlights = [
                f"Correlation: {e['feature_a']} ↔ {e['feature_b']} = {e['correlation']:.4f}"
                for e in corr_data.get("top_correlations", [])[:3]
            ]
            context["eda_highlights"] = highlights
        if best_metrics:
            context["model_metrics"] = best_metrics.to_dict()
        return build_recommendations_prompt(context)

    return None


def _flatten_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Convert DataProfiler output to the format expected by prompt_builder."""
    shape = profile.get("shape", {})
    n_rows = shape.get("rows", 0)
    n_cols = shape.get("columns", 0)
    null_counts = profile.get("null_counts", {})
    total_nulls = sum(null_counts.values())
    mem_bytes = profile.get("memory_usage_bytes", 0)

    num_stats = profile.get("numeric_stats", {})
    numeric_summary = {
        col: {
            "mean": s.get("mean"),
            "std": s.get("std"),
            "min": s.get("min"),
            "max": s.get("max"),
        }
        for col, s in num_stats.items()
        if s.get("mean") is not None
    }

    cat_stats = profile.get("categorical_stats", {})
    top_categories = {
        col: list(s.get("top_values", {}).items())[:3]
        for col, s in cat_stats.items()
    }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": len(num_stats),
        "n_categorical": len(cat_stats),
        "missing_cells": total_nulls,
        "missing_pct": total_nulls / max(n_rows * n_cols, 1) * 100,
        "duplicate_rows": 0,  # not tracked at insight stage
        "memory_usage_mb": mem_bytes / (1024 * 1024),
        "numeric_summary": numeric_summary,
        "top_categories": top_categories,
    }


def _flatten_eda(eda_result: dict[str, Any]) -> dict[str, Any]:
    """Convert EDAEngine output to the format expected by prompt_builder."""
    bivariate = eda_result.get("bivariate", {})
    corr = bivariate.get("correlation", {})
    pearson = corr.get("pearson", {})

    top_corrs: list[dict[str, Any]] = []
    seen: set[frozenset[str]] = set()
    for col_a, others in pearson.items():
        for col_b, val in others.items():
            if col_a == col_b:
                continue
            key = frozenset({col_a, col_b})
            if key in seen:
                continue
            seen.add(key)
            if val is not None:
                top_corrs.append({"feature_a": col_a, "feature_b": col_b, "correlation": val})
    top_corrs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    univariate = eda_result.get("univariate", {})
    numeric_uni = univariate.get("numeric", {})
    skewed = [
        {"column": col, "skewness": stats.get("skewness", 0)}
        for col, stats in numeric_uni.items()
        if stats.get("skewness") is not None and abs(stats.get("skewness", 0)) > 1
    ]
    skewed.sort(key=lambda x: abs(x["skewness"]), reverse=True)

    return {
        "top_correlations": top_corrs[:10],
        "skewed_columns": skewed[:10],
        "outlier_counts": {},
        "distribution_notes": [
            f"{col}: skewness={stats.get('skewness', 0):.2f}"
            for col, stats in numeric_uni.items()
            if stats.get("skewness") is not None
        ][:10],
        "group_stats": {},
    }
