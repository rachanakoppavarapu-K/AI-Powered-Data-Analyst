"""
Thin wrapper around the IBM watsonx.ai SDK.

Credentials are loaded exclusively from environment variables:
  WATSONX_API_KEY      — IBM Cloud IAM API key
  WATSONX_PROJECT_ID   — watsonx.ai project ID
  WATSONX_URL          — Regional endpoint URL
  WATSONX_MODEL_ID     — (optional) model ID override

When credentials are absent the client enters *fallback mode*: every call to
``generate()`` returns ``None`` instead of raising, allowing the rest of the
application to continue without AI-generated text.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass  # python-dotenv is optional; env vars may be set by other means

logger = logging.getLogger(__name__)

# Default model used when WATSONX_MODEL_ID is not set.
DEFAULT_MODEL_ID = "ibm/granite-13b-instruct-v2"

# Default generation parameters (can be overridden per-call).
DEFAULT_PARAMS: dict[str, Any] = {
    "decoding_method": "greedy",
    "max_new_tokens": 512,
    "min_new_tokens": 1,
    "repetition_penalty": 1.05,
}


class WatsonxClientError(Exception):
    """Raised for unrecoverable errors from the watsonx.ai API."""


class WatsonxClient:
    """Wraps the IBM watsonx.ai text-generation API.

    Parameters
    ----------
    api_key:
        IBM Cloud IAM API key.  Defaults to ``WATSONX_API_KEY`` env var.
    project_id:
        watsonx.ai project ID.  Defaults to ``WATSONX_PROJECT_ID`` env var.
    url:
        Regional endpoint URL.  Defaults to ``WATSONX_URL`` env var, then
        ``https://us-south.ml.cloud.ibm.com``.
    model_id:
        Model identifier.  Defaults to ``WATSONX_MODEL_ID`` env var, then
        ``DEFAULT_MODEL_ID``.
    """

    def __init__(
        self,
        api_key: str | None = None,
        project_id: str | None = None,
        url: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("WATSONX_API_KEY") or ""
        self._project_id = project_id or os.getenv("WATSONX_PROJECT_ID") or ""
        self._url = (
            url
            or os.getenv("WATSONX_URL")
            or "https://us-south.ml.cloud.ibm.com"
        )
        self._model_id = (
            model_id or os.getenv("WATSONX_MODEL_ID") or DEFAULT_MODEL_ID
        )
        self._model: Any = None  # lazy-initialised on first generate() call

    @property
    def is_configured(self) -> bool:
        """Return True when all mandatory credentials are present."""
        return bool(self._api_key and self._project_id)

    def _get_model(self) -> Any:
        """Lazily initialise and cache the SDK model object."""
        if self._model is not None:
            return self._model

        try:
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
        except ImportError as exc:
            raise WatsonxClientError(
                "ibm-watsonx-ai package is not installed.  "
                "Run: pip install ibm-watsonx-ai"
            ) from exc

        credentials = Credentials(url=self._url, api_key=self._api_key)
        self._model = ModelInference(
            model_id=self._model_id,
            credentials=credentials,
            project_id=self._project_id,
        )
        return self._model

    def generate(
        self,
        prompt: str,
        parameters: dict[str, Any] | None = None,
    ) -> str | None:
        """Generate text from *prompt* and return the response string.

        Returns ``None`` (instead of raising) when:
        - Credentials are not configured.
        - The API returns an error or times out.

        Parameters
        ----------
        prompt:
            The fully-formed prompt text.  Must already contain all numerical
            context — the LLM is never asked to compute values.
        parameters:
            Optional generation parameters that override ``DEFAULT_PARAMS``.
        """
        if not self.is_configured:
            logger.warning(
                "WatsonxClient: credentials not configured — "
                "GenAI insights unavailable. "
                "Set WATSONX_API_KEY and WATSONX_PROJECT_ID to enable."
            )
            return None

        params = {**DEFAULT_PARAMS, **(parameters or {})}

        try:
            model = self._get_model()
            response = model.generate_text(prompt=prompt, params=params)
            return str(response).strip()
        except WatsonxClientError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("WatsonxClient: API call failed — %s", exc)
            return None
