"""
Prompt template library for watsonx.ai text generation.

Design invariant (deterministic firewall)
-----------------------------------------
Every template function receives *pre-computed* numerical values as arguments
and embeds them verbatim into the prompt string.  No template ever asks the LLM
to calculate, estimate, or infer a number.  The LLM's only job is to translate
already-computed statistics into clear, non-technical natural language.

Six template functions are provided:
    build_dataset_summary_prompt      — high-level dataset overview
    build_key_findings_prompt         — top findings from profiling + EDA
    build_eda_explanation_prompt      — plain-language EDA explanation
    build_model_explanation_prompt    — ML model results explanation
    build_anomaly_explanation_prompt  — anomaly / outlier explanation
    build_recommendations_prompt      — actionable recommendations

Each function returns a ``str`` ready to pass to ``WatsonxClient.generate()``.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# 1. Dataset Summary
# ---------------------------------------------------------------------------


def build_dataset_summary_prompt(profile_report: dict[str, Any]) -> str:
    """Return a prompt asking for a plain-language dataset summary.

    Parameters
    ----------
    profile_report:
        Pre-computed profiling dict with keys:
          n_rows, n_cols, n_numeric, n_categorical, missing_cells,
          missing_pct, duplicate_rows, memory_usage_mb,
          numeric_summary (dict col → {mean, std, min, max}),
          top_categories  (dict col → list of (value, count) pairs)
    """
    n_rows = profile_report.get("n_rows", 0)
    n_cols = profile_report.get("n_cols", 0)
    n_numeric = profile_report.get("n_numeric", 0)
    n_categorical = profile_report.get("n_categorical", 0)
    missing_cells = profile_report.get("missing_cells", 0)
    missing_pct = profile_report.get("missing_pct", 0.0)
    duplicate_rows = profile_report.get("duplicate_rows", 0)
    memory_usage_mb = profile_report.get("memory_usage_mb", 0.0)

    numeric_lines: list[str] = []
    for col, stats in (profile_report.get("numeric_summary") or {}).items():
        numeric_lines.append(
            f"  - {col}: mean={stats.get('mean', 'N/A'):.4g}, "
            f"std={stats.get('std', 'N/A'):.4g}, "
            f"min={stats.get('min', 'N/A'):.4g}, "
            f"max={stats.get('max', 'N/A'):.4g}"
        )

    category_lines: list[str] = []
    for col, top_vals in (profile_report.get("top_categories") or {}).items():
        pairs = ", ".join(f"{v}({c})" for v, c in (top_vals or [])[:3])
        category_lines.append(f"  - {col}: {pairs}")

    numeric_block = "\n".join(numeric_lines) if numeric_lines else "  (none)"
    category_block = "\n".join(category_lines) if category_lines else "  (none)"

    return (
        "You are a data analyst writing for a non-technical business audience.\n\n"
        "The following statistics were computed by a data-profiling tool. "
        "Do not recalculate or change any number. "
        "Write a concise, plain-English summary (3–5 sentences) of what this dataset looks like.\n\n"
        "=== Dataset Statistics (pre-computed) ===\n"
        f"Rows: {n_rows}\n"
        f"Columns: {n_cols}\n"
        f"Numeric columns: {n_numeric}\n"
        f"Categorical columns: {n_categorical}\n"
        f"Missing cells: {missing_cells} ({missing_pct:.2f}%)\n"
        f"Duplicate rows: {duplicate_rows}\n"
        f"Memory usage: {memory_usage_mb:.2f} MB\n\n"
        "Numeric column statistics:\n"
        f"{numeric_block}\n\n"
        "Top categories per categorical column:\n"
        f"{category_block}\n\n"
        "=== Task ===\n"
        "Write a clear, non-technical paragraph summarising the dataset. "
        "Do not introduce any numbers that are not listed above."
    )


# ---------------------------------------------------------------------------
# 2. Key Findings
# ---------------------------------------------------------------------------


def build_key_findings_prompt(profile_report: dict[str, Any], eda_results: dict[str, Any]) -> str:
    """Return a prompt asking for the top key findings.

    Parameters
    ----------
    profile_report:
        Same structure as for ``build_dataset_summary_prompt``.
    eda_results:
        Pre-computed EDA dict with keys:
          top_correlations (list of {feature_a, feature_b, correlation}),
          skewed_columns   (list of {column, skewness}),
          outlier_counts   (dict col → int)
    """
    n_rows = profile_report.get("n_rows", 0)
    n_cols = profile_report.get("n_cols", 0)
    missing_pct = profile_report.get("missing_pct", 0.0)

    corr_lines: list[str] = []
    for entry in (eda_results.get("top_correlations") or [])[:5]:
        corr_lines.append(
            f"  - {entry['feature_a']} / {entry['feature_b']}: "
            f"{entry['correlation']:.4f}"
        )

    skew_lines: list[str] = []
    for entry in (eda_results.get("skewed_columns") or [])[:5]:
        skew_lines.append(f"  - {entry['column']}: skewness={entry['skewness']:.4f}")

    outlier_lines: list[str] = []
    for col, count in (eda_results.get("outlier_counts") or {}).items():
        outlier_lines.append(f"  - {col}: {count} outliers")

    corr_block = "\n".join(corr_lines) if corr_lines else "  (none)"
    skew_block = "\n".join(skew_lines) if skew_lines else "  (none)"
    outlier_block = "\n".join(outlier_lines) if outlier_lines else "  (none)"

    return (
        "You are a data analyst writing for a non-technical business audience.\n\n"
        "The numbers below were computed deterministically by statistical tools. "
        "Do not change, recalculate, or introduce any number not listed here. "
        "Summarise the three to five most important findings in bullet-point form.\n\n"
        "=== Pre-Computed Statistics ===\n"
        f"Dataset: {n_rows} rows, {n_cols} columns\n"
        f"Missing data: {missing_pct:.2f}%\n\n"
        "Top correlations:\n"
        f"{corr_block}\n\n"
        "Highly skewed columns:\n"
        f"{skew_block}\n\n"
        "Outlier counts:\n"
        f"{outlier_block}\n\n"
        "=== Task ===\n"
        "List the key findings a business analyst should know. "
        "Use only the numbers shown above. Do not invent or estimate values."
    )


# ---------------------------------------------------------------------------
# 3. EDA Explanation
# ---------------------------------------------------------------------------


def build_eda_explanation_prompt(eda_results: dict[str, Any]) -> str:
    """Return a prompt asking for a plain-language EDA explanation.

    Parameters
    ----------
    eda_results:
        Pre-computed EDA dict with keys:
          top_correlations (list of {feature_a, feature_b, correlation}),
          distribution_notes (list of str — already-written observations),
          group_stats (dict group_col → list of {group, metric, value})
    """
    corr_lines: list[str] = []
    for entry in (eda_results.get("top_correlations") or [])[:10]:
        corr_lines.append(
            f"  - {entry['feature_a']} ↔ {entry['feature_b']}: "
            f"r = {entry['correlation']:.4f}"
        )

    dist_lines: list[str] = []
    for note in (eda_results.get("distribution_notes") or [])[:10]:
        dist_lines.append(f"  - {note}")

    group_lines: list[str] = []
    for group_col, stats_list in (eda_results.get("group_stats") or {}).items():
        for entry in (stats_list or [])[:5]:
            group_lines.append(
                f"  - {group_col}={entry['group']}: "
                f"{entry['metric']}={entry['value']:.4g}"
            )

    corr_block = "\n".join(corr_lines) if corr_lines else "  (none)"
    dist_block = "\n".join(dist_lines) if dist_lines else "  (none)"
    group_block = "\n".join(group_lines) if group_lines else "  (none)"

    return (
        "You are a data analyst writing for a non-technical business audience.\n\n"
        "All numbers below are pre-computed. Do not recalculate, alter, or add any value. "
        "Explain what the exploratory data analysis reveals in plain language (4–6 sentences).\n\n"
        "=== Exploratory Data Analysis Results (pre-computed) ===\n"
        "Correlations:\n"
        f"{corr_block}\n\n"
        "Distribution observations:\n"
        f"{dist_block}\n\n"
        "Group-level statistics:\n"
        f"{group_block}\n\n"
        "=== Task ===\n"
        "Explain what these EDA results mean for a business analyst. "
        "Do not introduce any numbers not listed above."
    )


# ---------------------------------------------------------------------------
# 4. Machine Learning Result Explanation
# ---------------------------------------------------------------------------


def build_model_explanation_prompt(
    metrics: dict[str, Any],
    feature_importance: list[dict[str, Any]],
) -> str:
    """Return a prompt asking for an ML result explanation.

    Parameters
    ----------
    metrics:
        Pre-computed evaluation metrics.  Recognised keys (all optional):
          task_type, accuracy, precision, recall, f1, roc_auc,
          rmse, mae, r2, silhouette_score, n_samples, n_features, model_name
    feature_importance:
        Ordered list of {feature, importance} dicts (descending importance).
    """
    task_type = metrics.get("task_type", "unknown")
    model_name = metrics.get("model_name", "unknown")
    n_samples = metrics.get("n_samples", "N/A")
    n_features = metrics.get("n_features", "N/A")

    # Build metrics lines from whichever keys are present
    metric_pairs = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("roc_auc", "ROC-AUC"),
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("r2", "R²"),
        ("silhouette_score", "Silhouette Score"),
    ]
    metric_lines: list[str] = []
    for key, label in metric_pairs:
        if key in metrics:
            metric_lines.append(f"  - {label}: {metrics[key]:.6g}")

    importance_lines: list[str] = []
    for entry in (feature_importance or [])[:10]:
        importance_lines.append(
            f"  - {entry['feature']}: {entry['importance']:.6g}"
        )

    metric_block = "\n".join(metric_lines) if metric_lines else "  (none)"
    importance_block = "\n".join(importance_lines) if importance_lines else "  (none)"

    return (
        "You are a data scientist writing for a non-technical business audience.\n\n"
        "The evaluation metrics and feature importances below were computed by a "
        "machine-learning evaluation framework. Do not change, recalculate, or add "
        "any number. Explain the model performance and key drivers in 4–6 sentences.\n\n"
        "=== Model Evaluation (pre-computed) ===\n"
        f"Task type: {task_type}\n"
        f"Model: {model_name}\n"
        f"Training samples: {n_samples}\n"
        f"Features: {n_features}\n\n"
        "Performance metrics:\n"
        f"{metric_block}\n\n"
        "Top feature importances:\n"
        f"{importance_block}\n\n"
        "=== Task ===\n"
        "Interpret what these results mean for the business. "
        "Do not introduce any numbers not listed above."
    )


# ---------------------------------------------------------------------------
# 5. Anomaly Explanation
# ---------------------------------------------------------------------------


def build_anomaly_explanation_prompt(anomaly_report: dict[str, Any]) -> str:
    """Return a prompt asking for a plain-language anomaly/outlier explanation.

    Parameters
    ----------
    anomaly_report:
        Pre-computed anomaly detection results with keys:
          total_anomalies (int),
          anomaly_pct     (float),
          affected_columns (list of {column, n_outliers, lower_bound, upper_bound}),
          method          (str — detection method used, e.g. 'IQR')
    """
    total_anomalies = anomaly_report.get("total_anomalies", 0)
    anomaly_pct = anomaly_report.get("anomaly_pct", 0.0)
    method = anomaly_report.get("method", "IQR")

    col_lines: list[str] = []
    for entry in (anomaly_report.get("affected_columns") or []):
        lb = entry.get("lower_bound")
        ub = entry.get("upper_bound")
        bounds = ""
        if lb is not None and ub is not None:
            bounds = f", expected range [{lb:.4g}, {ub:.4g}]"
        col_lines.append(
            f"  - {entry['column']}: {entry['n_outliers']} outliers{bounds}"
        )

    col_block = "\n".join(col_lines) if col_lines else "  (none)"

    return (
        "You are a data analyst writing for a non-technical business audience.\n\n"
        "The anomaly counts and bounds below were computed by a statistical outlier "
        "detection algorithm. Do not change, recalculate, or add any number. "
        "Explain what the anomalies might mean and why they matter (3–5 sentences).\n\n"
        "=== Anomaly Detection Results (pre-computed) ===\n"
        f"Detection method: {method}\n"
        f"Total anomalous records: {total_anomalies} ({anomaly_pct:.2f}%)\n\n"
        "Affected columns:\n"
        f"{col_block}\n\n"
        "=== Task ===\n"
        "Explain what these anomalies might indicate and whether they warrant "
        "investigation. Do not introduce any numbers not listed above."
    )


# ---------------------------------------------------------------------------
# 6. Recommendations
# ---------------------------------------------------------------------------


def build_recommendations_prompt(full_context: dict[str, Any]) -> str:
    """Return a prompt asking for actionable recommendations.

    Parameters
    ----------
    full_context:
        Aggregated pre-computed context dict with keys (all optional):
          dataset_summary  (dict — same shape as profile_report),
          eda_highlights   (list of str — key EDA observations),
          model_metrics    (dict — same shape as metrics in build_model_explanation_prompt),
          anomaly_summary  (dict — same shape as anomaly_report),
          business_goal    (str — optional framing, e.g. 'reduce customer churn')
    """
    dataset_summary = full_context.get("dataset_summary") or {}
    eda_highlights = full_context.get("eda_highlights") or []
    model_metrics = full_context.get("model_metrics") or {}
    anomaly_summary = full_context.get("anomaly_summary") or {}
    business_goal = full_context.get("business_goal", "improve data-driven decision making")

    # Dataset snapshot
    n_rows = dataset_summary.get("n_rows", "N/A")
    n_cols = dataset_summary.get("n_cols", "N/A")
    missing_pct = dataset_summary.get("missing_pct", "N/A")

    # EDA highlights
    eda_lines = "\n".join(f"  - {h}" for h in eda_highlights[:5]) or "  (none)"

    # Model performance snapshot
    metric_pairs = [
        ("accuracy", "Accuracy"),
        ("f1", "F1 Score"),
        ("rmse", "RMSE"),
        ("r2", "R²"),
    ]
    metric_lines: list[str] = []
    for key, label in metric_pairs:
        if key in model_metrics:
            metric_lines.append(f"  - {label}: {model_metrics[key]:.6g}")
    metric_block = "\n".join(metric_lines) if metric_lines else "  (none)"

    # Anomaly snapshot
    total_anomalies = anomaly_summary.get("total_anomalies", "N/A")
    anomaly_pct = anomaly_summary.get("anomaly_pct", "N/A")
    anomaly_str = (
        f"{total_anomalies} anomalous records ({anomaly_pct:.2f}%)"
        if isinstance(anomaly_pct, float)
        else str(total_anomalies)
    )

    return (
        "You are a senior data scientist writing for a non-technical business audience.\n\n"
        "All numbers below were computed by data science tools. "
        "Do not change, recalculate, or add any number. "
        "Provide 3–5 concrete, actionable recommendations based solely on these findings.\n\n"
        "=== Analysis Summary (pre-computed) ===\n"
        f"Business goal: {business_goal}\n"
        f"Dataset: {n_rows} rows, {n_cols} columns, {missing_pct}% missing\n\n"
        "Key EDA observations:\n"
        f"{eda_lines}\n\n"
        "Model performance:\n"
        f"{metric_block}\n\n"
        f"Anomalies detected: {anomaly_str}\n\n"
        "=== Task ===\n"
        "Give 3–5 bullet-point recommendations. Each must reference at least one "
        "number from the list above. Do not introduce any numbers not listed above."
    )
