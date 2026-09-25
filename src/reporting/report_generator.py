"""report_generator.py — ST-10: Assemble pipeline outputs into an HTML/PDF report.

Design principles
-----------------
* All section data is supplied in the *context* dict — this module never
  computes or invents statistics.
* Each section is optional; missing keys are handled gracefully.
* The generated HTML is self-contained (no external URLs required).
* Plotly figures supplied in the context are converted to base64-encoded PNG
  images and embedded inline.
* Optional PDF export is attempted via ``weasyprint``; if the library is not
  available, HTML-only export is returned and a warning is emitted.

Expected context keys (all optional)
-------------------------------------
title : str
    Report title.
dataset_name : str
    Name of the uploaded dataset.
generated_at : str
    Timestamp string (auto-filled if absent).
profile : dict
    Output of ``DataProfiler.profile()``.
data_quality : dict
    Keys: ``null_percentages`` (dict), ``outlier_info`` (dict), ``notes`` (list[str]).
eda : dict
    Keys: ``top_correlations`` (list), ``skewed_columns`` (list),
    ``charts`` (list[plotly.Figure]).
ml_metrics : dict
    ``ModelMetrics.to_dict()`` output, plus optional ``charts`` key
    (list[plotly.Figure]).
feature_importance : dict
    Keys: ``entries`` (list[{feature, importance}]),
    ``chart`` (plotly.Figure | None).
genai_insights : dict[str, str]
    Mapping of section label → generated text.
nl_query_results : list[NLQueryResult | dict]
    Each item must have ``.question``, ``.explanation``, ``.computed_result``,
    ``.error`` (or equivalent dict keys).
recommendations : str | list[str]
    A single block of text or a list of bullet-point recommendations.
"""

from __future__ import annotations

import base64
import io
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

__all__ = ["ReportGenerator"]

# ---------------------------------------------------------------------------
# Path to the Jinja2 template
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
_TEMPLATE_NAME = "report_template.html"

# ---------------------------------------------------------------------------
# Jinja2 custom filters
# ---------------------------------------------------------------------------


def _int_fmt(value: Any) -> str:
    """Format an integer with thousands separator."""
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _num_fmt(value: Any) -> str:
    """Format a float to 4 significant figures; return '—' for None."""
    if value is None:
        return "—"
    try:
        f = float(value)
        if f != f:  # NaN
            return "—"
        # 4 significant figures, strip trailing zeros
        return f"{f:.4g}"
    except (TypeError, ValueError):
        return str(value)


def _bytes_fmt(value: Any) -> str:
    """Human-readable file-size string."""
    if value is None:
        return "—"
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _build_jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_ASSETS_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["int_fmt"] = _int_fmt
    env.filters["num_fmt"] = _num_fmt
    env.filters["bytes_fmt"] = _bytes_fmt
    return env


# ---------------------------------------------------------------------------
# Figure → base64 PNG helper
# ---------------------------------------------------------------------------


def _figure_to_b64(fig: Any) -> str | None:
    """Convert a Plotly Figure to a base64-encoded PNG string.

    Returns ``None`` when the figure cannot be converted (e.g. kaleido not
    installed).
    """
    try:
        import plotly.graph_objects as go  # noqa: PLC0415

        if not isinstance(fig, go.Figure):
            return None
        png_bytes: bytes = fig.to_image(format="png", width=800, height=400)
        return base64.b64encode(png_bytes).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _convert_figures(items: list[Any]) -> list[str]:
    """Convert a list of Plotly figures to a list of base64 PNG strings.

    Items that cannot be converted are silently dropped.
    """
    out: list[str] = []
    for item in items:
        b64 = _figure_to_b64(item)
        if b64:
            out.append(b64)
    return out


# ---------------------------------------------------------------------------
# Context normalisation helpers
# ---------------------------------------------------------------------------


def _normalise_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *ctx* with charts converted to base64 and defaults set."""
    c = dict(ctx)

    # Timestamp
    if not c.get("generated_at"):
        c["generated_at"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # EDA charts
    eda = c.get("eda")
    if isinstance(eda, dict):
        eda = dict(eda)
        raw_charts = eda.get("charts", [])
        if raw_charts:
            eda["charts"] = _convert_figures(raw_charts)
        c["eda"] = eda

    # ML charts
    ml = c.get("ml_metrics")
    if isinstance(ml, dict):
        ml = dict(ml)
        raw_charts = ml.get("charts", [])
        if raw_charts:
            ml["charts"] = _convert_figures(raw_charts)
        c["ml_metrics"] = ml

    # Feature importance chart
    fi = c.get("feature_importance")
    if isinstance(fi, dict):
        fi = dict(fi)
        raw_chart = fi.get("chart")
        if raw_chart is not None:
            fi["chart"] = _figure_to_b64(raw_chart)
        c["feature_importance"] = fi

    # NL query results: normalise dataclass-like objects to dicts
    nl_results = c.get("nl_query_results")
    if nl_results:
        normalised: list[dict[str, Any]] = []
        for r in nl_results:
            if isinstance(r, dict):
                normalised.append(r)
            else:
                # Assume dataclass / object with matching attributes
                normalised.append({
                    "question": getattr(r, "question", ""),
                    "explanation": getattr(r, "explanation", None),
                    "computed_result": getattr(r, "computed_result", None),
                    "error": getattr(r, "error", None),
                })
        c["nl_query_results"] = normalised

    return c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Assemble pipeline outputs into a self-contained HTML (or PDF) report.

    Parameters
    ----------
    template_name:
        Filename of the Jinja2 template relative to ``assets/``.
        Defaults to ``"report_template.html"``.
    output_dir:
        Directory to write generated reports.  Defaults to the OS temp
        directory.
    """

    def __init__(
        self,
        template_name: str = _TEMPLATE_NAME,
        output_dir: str | Path | None = None,
    ) -> None:
        self._template_name = template_name
        self._output_dir = Path(output_dir) if output_dir else Path(
            os.environ.get("TMPDIR", os.environ.get("TEMP", "/tmp"))
        )
        self._env = _build_jinja_env()

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        context: dict[str, Any],
        *,
        output_path: str | Path | None = None,
        export_pdf: bool = False,
    ) -> str:
        """Render the report and write it to disk.

        Parameters
        ----------
        context:
            Dictionary of pipeline outputs (see module docstring for expected
            keys).  All keys are optional; missing sections are gracefully
            omitted from the report.
        output_path:
            Explicit output file path.  If *None*, a timestamped file is
            created inside ``self.output_dir``.
        export_pdf:
            When *True*, additionally attempt to export a PDF alongside the
            HTML file.  Requires ``weasyprint``.  If ``weasyprint`` is not
            available, a warning is emitted and HTML-only export proceeds.

        Returns
        -------
        str
            Absolute path of the generated HTML file.
        """
        ctx = _normalise_context(context)
        html_str = self._render(ctx)

        # Determine output path
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{ts}.html"
            html_path = self._output_dir / filename
        else:
            html_path = Path(output_path)

        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_str, encoding="utf-8")

        if export_pdf:
            self._export_pdf(html_str, html_path)

        return str(html_path)

    # ------------------------------------------------------------------
    # render → str (useful for Streamlit st.components.v1.html)
    # ------------------------------------------------------------------

    def render_html(self, context: dict[str, Any]) -> str:
        """Render the report to an HTML string without writing to disk.

        Parameters
        ----------
        context:
            Dictionary of pipeline outputs.

        Returns
        -------
        str
            Fully self-contained HTML document.
        """
        ctx = _normalise_context(context)
        return self._render(ctx)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render(self, ctx: dict[str, Any]) -> str:
        template = self._env.get_template(self._template_name)
        return template.render(**ctx)

    def _export_pdf(self, html_str: str, html_path: Path) -> None:
        pdf_path = html_path.with_suffix(".pdf")
        try:
            from weasyprint import HTML  # noqa: PLC0415

            HTML(string=html_str).write_pdf(str(pdf_path))
        except ImportError:
            warnings.warn(
                "weasyprint is not installed; PDF export skipped. "
                "Install it with: pip install weasyprint",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"PDF export failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
