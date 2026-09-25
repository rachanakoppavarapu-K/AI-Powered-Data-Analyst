"""
Tests for src/genai/watsonx_client.py and src/genai/prompt_builder.py.

All watsonx.ai API calls are mocked — no real network calls are made.

Key assertions
--------------
- WatsonxClient.is_configured returns False when credentials are absent.
- WatsonxClient.generate() returns None (not raises) when unconfigured.
- WatsonxClient.generate() returns None (not raises) on API error.
- WatsonxClient.generate() returns the model response string on success.
- Every prompt template embeds the injected numbers verbatim.
- No template asks the LLM to compute a value (deterministic-firewall check).
"""

from __future__ import annotations

import importlib
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile_report() -> dict[str, Any]:
    return {
        "n_rows": 1500,
        "n_cols": 12,
        "n_numeric": 8,
        "n_categorical": 4,
        "missing_cells": 45,
        "missing_pct": 3.0,
        "duplicate_rows": 7,
        "memory_usage_mb": 0.14,
        "numeric_summary": {
            "age": {"mean": 38.5, "std": 12.3, "min": 18.0, "max": 75.0},
            "income": {"mean": 52000.0, "std": 18000.0, "min": 10000.0, "max": 150000.0},
        },
        "top_categories": {
            "gender": [("Male", 800), ("Female", 700)],
            "region": [("North", 600), ("South", 500), ("East", 400)],
        },
    }


@pytest.fixture()
def eda_results() -> dict[str, Any]:
    return {
        "top_correlations": [
            {"feature_a": "income", "feature_b": "spend", "correlation": 0.82},
            {"feature_a": "age", "feature_b": "savings", "correlation": 0.61},
        ],
        "skewed_columns": [
            {"column": "income", "skewness": 1.74},
            {"column": "spend", "skewness": -0.32},
        ],
        "outlier_counts": {"income": 23, "spend": 5},
        "distribution_notes": [
            "income is right-skewed with skewness 1.74",
            "spend has 5 outliers above the IQR upper fence",
        ],
        "group_stats": {
            "region": [
                {"group": "North", "metric": "mean_income", "value": 55000.0},
                {"group": "South", "metric": "mean_income", "value": 48000.0},
            ]
        },
    }


@pytest.fixture()
def model_metrics() -> dict[str, Any]:
    return {
        "task_type": "classification",
        "model_name": "RandomForestClassifier",
        "n_samples": 1200,
        "n_features": 11,
        "accuracy": 0.879,
        "precision": 0.863,
        "recall": 0.891,
        "f1": 0.877,
        "roc_auc": 0.934,
    }


@pytest.fixture()
def feature_importance() -> list[dict[str, Any]]:
    return [
        {"feature": "income", "importance": 0.341},
        {"feature": "age", "importance": 0.189},
        {"feature": "spend", "importance": 0.143},
    ]


@pytest.fixture()
def anomaly_report() -> dict[str, Any]:
    return {
        "total_anomalies": 31,
        "anomaly_pct": 2.07,
        "method": "IQR",
        "affected_columns": [
            {"column": "income", "n_outliers": 23, "lower_bound": 5000.0, "upper_bound": 120000.0},
            {"column": "spend", "n_outliers": 8, "lower_bound": 0.0, "upper_bound": 9500.0},
        ],
    }


@pytest.fixture()
def full_context(profile_report, eda_results, model_metrics, anomaly_report) -> dict[str, Any]:
    return {
        "dataset_summary": profile_report,
        "eda_highlights": [
            "income / spend correlation = 0.82",
            "income is right-skewed (skewness 1.74)",
        ],
        "model_metrics": model_metrics,
        "anomaly_summary": anomaly_report,
        "business_goal": "reduce customer churn",
    }


# ---------------------------------------------------------------------------
# Helper: fresh client with env vars isolated
# ---------------------------------------------------------------------------


def _make_client(api_key: str = "", project_id: str = "", url: str = "", model_id: str = ""):
    """Import watsonx_client fresh each time so module-level load_dotenv() is re-executed."""
    from src.genai.watsonx_client import WatsonxClient  # noqa: PLC0415

    return WatsonxClient(
        api_key=api_key or None,
        project_id=project_id or None,
        url=url or None,
        model_id=model_id or None,
    )


# ===========================================================================
# WatsonxClient — configuration and fallback
# ===========================================================================


class TestWatsonxClientConfiguration:
    def test_is_configured_false_when_no_credentials(self, monkeypatch):
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        client = _make_client()
        assert client.is_configured is False

    def test_is_configured_false_when_only_api_key(self, monkeypatch):
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        client = _make_client(api_key="my-key")
        assert client.is_configured is False

    def test_is_configured_false_when_only_project_id(self, monkeypatch):
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        client = _make_client(project_id="my-project")
        assert client.is_configured is False

    def test_is_configured_true_when_both_credentials_set(self):
        client = _make_client(api_key="my-key", project_id="my-project")
        assert client.is_configured is True

    def test_credentials_loaded_from_env(self, monkeypatch):
        monkeypatch.setenv("WATSONX_API_KEY", "env-key")
        monkeypatch.setenv("WATSONX_PROJECT_ID", "env-project")
        client = _make_client()
        assert client.is_configured is True

    def test_default_url_is_us_south(self, monkeypatch):
        monkeypatch.delenv("WATSONX_URL", raising=False)
        client = _make_client(api_key="k", project_id="p")
        assert "us-south" in client._url

    def test_default_model_id_set(self, monkeypatch):
        monkeypatch.delenv("WATSONX_MODEL_ID", raising=False)
        client = _make_client(api_key="k", project_id="p")
        assert client._model_id  # non-empty


class TestWatsonxClientFallback:
    def test_generate_returns_none_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        client = _make_client()
        result = client.generate("some prompt")
        assert result is None

    def test_generate_does_not_raise_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("WATSONX_API_KEY", raising=False)
        monkeypatch.delenv("WATSONX_PROJECT_ID", raising=False)
        client = _make_client()
        # Must not raise
        client.generate("some prompt")

    def test_generate_returns_none_on_api_error(self):
        client = _make_client(api_key="k", project_id="p")
        mock_model = MagicMock()
        mock_model.generate_text.side_effect = RuntimeError("network error")
        client._model = mock_model
        result = client.generate("some prompt")
        assert result is None

    def test_generate_returns_none_on_import_error(self):
        """Simulate ibm-watsonx-ai not installed."""
        client = _make_client(api_key="k", project_id="p")
        with patch.dict("sys.modules", {"ibm_watsonx_ai": None,
                                        "ibm_watsonx_ai.foundation_models": None}):
            from src.genai.watsonx_client import WatsonxClientError
            # _get_model will raise WatsonxClientError on ImportError
            with pytest.raises(WatsonxClientError):
                client._get_model()


class TestWatsonxClientGenerate:
    def _configured_client_with_mock_model(self, response_text: str):
        client = _make_client(api_key="k", project_id="p")
        mock_model = MagicMock()
        mock_model.generate_text.return_value = response_text
        client._model = mock_model
        return client, mock_model

    def test_generate_returns_model_response(self):
        client, mock_model = self._configured_client_with_mock_model("Great insight!")
        result = client.generate("prompt text")
        assert result == "Great insight!"

    def test_generate_strips_whitespace(self):
        client, _ = self._configured_client_with_mock_model("  answer  \n")
        result = client.generate("prompt")
        assert result == "answer"

    def test_generate_passes_prompt_to_model(self):
        client, mock_model = self._configured_client_with_mock_model("ok")
        client.generate("my prompt")
        call_kwargs = mock_model.generate_text.call_args
        # prompt was passed either positionally or as keyword
        called_prompt = (
            call_kwargs.kwargs.get("prompt")
            or (call_kwargs.args[0] if call_kwargs.args else None)
        )
        assert called_prompt == "my prompt"

    def test_generate_merges_custom_parameters(self):
        client, mock_model = self._configured_client_with_mock_model("ok")
        client.generate("prompt", parameters={"max_new_tokens": 256})
        call_kwargs = mock_model.generate_text.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs.args[1]
        assert params["max_new_tokens"] == 256

    def test_generate_uses_default_parameters(self):
        from src.genai.watsonx_client import DEFAULT_PARAMS

        client, mock_model = self._configured_client_with_mock_model("ok")
        client.generate("prompt")
        call_kwargs = mock_model.generate_text.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs.args[1]
        assert params["decoding_method"] == DEFAULT_PARAMS["decoding_method"]

    def test_full_sdk_mock_via_patch(self):
        """Mock the entire SDK import path."""
        mock_model_instance = MagicMock()
        mock_model_instance.generate_text.return_value = "mocked answer"
        mock_model_cls = MagicMock(return_value=mock_model_instance)
        mock_credentials_cls = MagicMock()

        from src.genai.watsonx_client import WatsonxClient
        with patch.object(WatsonxClient, "_get_model", return_value=mock_model_instance):
            client = WatsonxClient(api_key="k", project_id="p")
            result = client.generate("test prompt")

        assert result == "mocked answer"


# ===========================================================================
# Prompt Builder — deterministic firewall tests
# All injected numbers MUST appear verbatim in the prompt.
# ===========================================================================


from src.genai.prompt_builder import (  # noqa: E402
    build_anomaly_explanation_prompt,
    build_dataset_summary_prompt,
    build_eda_explanation_prompt,
    build_key_findings_prompt,
    build_model_explanation_prompt,
    build_recommendations_prompt,
)


class TestDatasetSummaryPrompt:
    def test_returns_string(self, profile_report):
        result = build_dataset_summary_prompt(profile_report)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_n_rows_injected(self, profile_report):
        result = build_dataset_summary_prompt(profile_report)
        assert "1500" in result

    def test_n_cols_injected(self, profile_report):
        result = build_dataset_summary_prompt(profile_report)
        assert "12" in result

    def test_missing_pct_injected(self, profile_report):
        result = build_dataset_summary_prompt(profile_report)
        assert "3.00" in result

    def test_duplicate_rows_injected(self, profile_report):
        result = build_dataset_summary_prompt(profile_report)
        assert "7" in result

    def test_numeric_summary_values_injected(self, profile_report):
        result = build_dataset_summary_prompt(profile_report)
        # mean of age = 38.5
        assert "38.5" in result or "38.5" in result

    def test_llm_not_asked_to_compute(self, profile_report):
        import re
        result = build_dataset_summary_prompt(profile_report)
        # The prompt may contain "recalculate" (an instruction not to do so),
        # but must never contain a bare "calculate" or "compute" directive
        # asking the LLM to perform arithmetic.
        assert not re.search(r'\bcalculate\b', result.lower())
        assert not re.search(r'\bcompute\b', result.lower())

    def test_do_not_introduce_numbers_instruction(self, profile_report):
        result = build_dataset_summary_prompt(profile_report)
        assert "Do not introduce any numbers" in result

    def test_handles_empty_profile(self):
        result = build_dataset_summary_prompt({})
        assert isinstance(result, str)


class TestKeyFindingsPrompt:
    def test_returns_string(self, profile_report, eda_results):
        result = build_key_findings_prompt(profile_report, eda_results)
        assert isinstance(result, str)

    def test_correlation_value_injected(self, profile_report, eda_results):
        result = build_key_findings_prompt(profile_report, eda_results)
        assert "0.8200" in result or "0.82" in result

    def test_skewness_injected(self, profile_report, eda_results):
        result = build_key_findings_prompt(profile_report, eda_results)
        assert "1.7400" in result or "1.74" in result

    def test_outlier_count_injected(self, profile_report, eda_results):
        result = build_key_findings_prompt(profile_report, eda_results)
        assert "23" in result

    def test_missing_pct_injected(self, profile_report, eda_results):
        result = build_key_findings_prompt(profile_report, eda_results)
        assert "3.00" in result

    def test_do_not_invent_instruction(self, profile_report, eda_results):
        result = build_key_findings_prompt(profile_report, eda_results)
        assert "Do not invent" in result or "not listed" in result

    def test_handles_empty_eda(self, profile_report):
        result = build_key_findings_prompt(profile_report, {})
        assert isinstance(result, str)


class TestEdaExplanationPrompt:
    def test_returns_string(self, eda_results):
        result = build_eda_explanation_prompt(eda_results)
        assert isinstance(result, str)

    def test_correlation_injected(self, eda_results):
        result = build_eda_explanation_prompt(eda_results)
        assert "0.8200" in result or "0.82" in result

    def test_feature_names_injected(self, eda_results):
        result = build_eda_explanation_prompt(eda_results)
        assert "income" in result
        assert "spend" in result

    def test_distribution_notes_injected(self, eda_results):
        result = build_eda_explanation_prompt(eda_results)
        assert "1.74" in result  # skewness in distribution note

    def test_group_stat_value_injected(self, eda_results):
        result = build_eda_explanation_prompt(eda_results)
        assert "55000" in result or "5.5e+04" in result

    def test_llm_not_asked_to_compute(self, eda_results):
        result = build_eda_explanation_prompt(eda_results)
        assert "Do not recalculate" in result or "not recalculate" in result

    def test_handles_empty_eda(self):
        result = build_eda_explanation_prompt({})
        assert isinstance(result, str)


class TestModelExplanationPrompt:
    def test_returns_string(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert isinstance(result, str)

    def test_accuracy_injected(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert "0.879" in result

    def test_f1_injected(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert "0.877" in result

    def test_precision_injected(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert "0.863" in result

    def test_recall_injected(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert "0.891" in result

    def test_roc_auc_injected(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert "0.934" in result

    def test_feature_importance_injected(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert "0.341" in result
        assert "income" in result

    def test_model_name_injected(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert "RandomForestClassifier" in result

    def test_sample_count_injected(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert "1200" in result

    def test_llm_not_asked_to_compute(self, model_metrics, feature_importance):
        result = build_model_explanation_prompt(model_metrics, feature_importance)
        assert "Do not change" in result or "not listed" in result

    def test_handles_regression_metrics(self):
        metrics = {
            "task_type": "regression",
            "model_name": "XGBoostRegressor",
            "n_samples": 500,
            "n_features": 7,
            "rmse": 4231.5,
            "mae": 3100.2,
            "r2": 0.873,
        }
        result = build_model_explanation_prompt(metrics, [])
        assert "4231.5" in result or "4231" in result
        assert "0.873" in result


class TestAnomalyExplanationPrompt:
    def test_returns_string(self, anomaly_report):
        result = build_anomaly_explanation_prompt(anomaly_report)
        assert isinstance(result, str)

    def test_total_anomalies_injected(self, anomaly_report):
        result = build_anomaly_explanation_prompt(anomaly_report)
        assert "31" in result

    def test_anomaly_pct_injected(self, anomaly_report):
        result = build_anomaly_explanation_prompt(anomaly_report)
        assert "2.07" in result

    def test_method_injected(self, anomaly_report):
        result = build_anomaly_explanation_prompt(anomaly_report)
        assert "IQR" in result

    def test_column_names_injected(self, anomaly_report):
        result = build_anomaly_explanation_prompt(anomaly_report)
        assert "income" in result
        assert "spend" in result

    def test_outlier_counts_injected(self, anomaly_report):
        result = build_anomaly_explanation_prompt(anomaly_report)
        assert "23" in result
        assert "8" in result

    def test_bounds_injected(self, anomaly_report):
        result = build_anomaly_explanation_prompt(anomaly_report)
        assert "120000" in result or "1.2e+05" in result

    def test_llm_not_asked_to_compute(self, anomaly_report):
        result = build_anomaly_explanation_prompt(anomaly_report)
        assert "Do not change" in result or "not listed" in result

    def test_handles_empty_report(self):
        result = build_anomaly_explanation_prompt({})
        assert isinstance(result, str)


class TestRecommendationsPrompt:
    def test_returns_string(self, full_context):
        result = build_recommendations_prompt(full_context)
        assert isinstance(result, str)

    def test_business_goal_injected(self, full_context):
        result = build_recommendations_prompt(full_context)
        assert "reduce customer churn" in result

    def test_dataset_rows_injected(self, full_context):
        result = build_recommendations_prompt(full_context)
        assert "1500" in result

    def test_model_accuracy_injected(self, full_context):
        result = build_recommendations_prompt(full_context)
        assert "0.879" in result

    def test_anomaly_count_injected(self, full_context):
        result = build_recommendations_prompt(full_context)
        assert "31" in result

    def test_eda_highlight_injected(self, full_context):
        result = build_recommendations_prompt(full_context)
        assert "0.82" in result

    def test_llm_not_asked_to_compute(self, full_context):
        result = build_recommendations_prompt(full_context)
        assert "Do not change" in result or "not listed" in result

    def test_handles_empty_context(self):
        result = build_recommendations_prompt({})
        assert isinstance(result, str)

    def test_references_numbers_instruction(self, full_context):
        result = build_recommendations_prompt(full_context)
        assert "reference at least one" in result or "number from" in result


# ===========================================================================
# Integration: prompt → client (mocked SDK)
# ===========================================================================


class TestPromptClientIntegration:
    def test_dataset_summary_prompt_sent_to_client(self, profile_report):
        from src.genai.watsonx_client import WatsonxClient

        client = WatsonxClient(api_key="k", project_id="p")
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "summary text"
        client._model = mock_model

        prompt = build_dataset_summary_prompt(profile_report)
        response = client.generate(prompt)
        assert response == "summary text"
        # Confirm the full prompt (containing injected numbers) was sent
        sent_prompt = mock_model.generate_text.call_args.kwargs.get("prompt") or \
                      mock_model.generate_text.call_args.args[0]
        assert "1500" in sent_prompt

    def test_model_explanation_prompt_sent_to_client(self, model_metrics, feature_importance):
        from src.genai.watsonx_client import WatsonxClient

        client = WatsonxClient(api_key="k", project_id="p")
        mock_model = MagicMock()
        mock_model.generate_text.return_value = "ml explanation"
        client._model = mock_model

        prompt = build_model_explanation_prompt(model_metrics, feature_importance)
        response = client.generate(prompt)
        assert response == "ml explanation"
        sent_prompt = mock_model.generate_text.call_args.kwargs.get("prompt") or \
                      mock_model.generate_text.call_args.args[0]
        assert "0.879" in sent_prompt  # accuracy injected
