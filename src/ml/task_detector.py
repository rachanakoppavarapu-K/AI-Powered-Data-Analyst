"""ML task detector — infers the appropriate ML task from dataset characteristics.

The detector inspects the target column's dtype and cardinality to return one of
three :class:`TaskType` values: ``CLASSIFICATION``, ``REGRESSION``, or
``CLUSTERING`` (when no target is provided).

The caller may always override the inferred type via ``user_override``.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd

__all__ = [
    "TaskType",
    "TaskDetectionResult",
    "detect_task",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum fraction of unique values in a numeric target column before the
#: task is inferred as regression rather than classification.
_CLASSIFICATION_UNIQUE_FRAC = 0.05

#: Hard cap: if unique values ≤ this, always prefer classification regardless
#: of column dtype.
_CLASSIFICATION_UNIQUE_MAX = 20

#: Minimum number of rows required to use a supervised task.  Datasets with
#: fewer rows fall back to CLUSTERING.
_MIN_SUPERVISED_ROWS = 10


# ---------------------------------------------------------------------------
# Enums / data classes
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """Supported ML task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class TaskDetectionResult:
    """Result returned by :func:`detect_task`.

    Attributes
    ----------
    task_type:
        The inferred (or overridden) :class:`TaskType`.
    reason:
        Human-readable explanation of why this task was chosen.
    n_classes:
        Number of unique target classes (``None`` for regression/clustering).
    is_override:
        ``True`` if the task was set by ``user_override`` rather than inferred.
    """

    def __init__(
        self,
        task_type: TaskType,
        reason: str,
        n_classes: Optional[int],
        is_override: bool,
    ) -> None:
        self.task_type = task_type
        self.reason = reason
        self.n_classes = n_classes
        self.is_override = is_override

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TaskDetectionResult(task_type={self.task_type!r}, "
            f"n_classes={self.n_classes}, is_override={self.is_override})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_task(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    user_override: Optional[TaskType] = None,
) -> TaskDetectionResult:
    """Infer the ML task from the dataset and target column.

    Parameters
    ----------
    df:
        Input DataFrame (features + optional target).
    target_column:
        Name of the target column.  When ``None``, the task is inferred as
        :attr:`TaskType.CLUSTERING`.
    user_override:
        When provided, the inferred logic is bypassed and this task type is
        returned directly (with ``is_override=True``).

    Returns
    -------
    TaskDetectionResult
    """
    if user_override is not None:
        n_classes: Optional[int] = None
        if target_column and target_column in df.columns:
            n_classes = int(df[target_column].nunique(dropna=True))
        return TaskDetectionResult(
            task_type=user_override,
            reason=f"Task overridden by user to '{user_override.value}'.",
            n_classes=n_classes,
            is_override=True,
        )

    # No target → clustering
    if target_column is None or target_column not in df.columns:
        return TaskDetectionResult(
            task_type=TaskType.CLUSTERING,
            reason="No target column provided; defaulting to clustering.",
            n_classes=None,
            is_override=False,
        )

    # Too few rows for supervised learning
    if len(df) < _MIN_SUPERVISED_ROWS:
        return TaskDetectionResult(
            task_type=TaskType.CLUSTERING,
            reason=(
                f"Dataset has only {len(df)} rows (< {_MIN_SUPERVISED_ROWS}); "
                "too small for supervised learning."
            ),
            n_classes=None,
            is_override=False,
        )

    target = df[target_column].dropna()
    n_unique = int(target.nunique())
    n_rows = len(target)

    # Categorical / boolean / object dtype → classification
    if pd.api.types.is_bool_dtype(target) or isinstance(target.dtype, pd.CategoricalDtype):
        return TaskDetectionResult(
            task_type=TaskType.CLASSIFICATION,
            reason=f"Target '{target_column}' is boolean/categorical → classification.",
            n_classes=n_unique,
            is_override=False,
        )

    if pd.api.types.is_object_dtype(target):
        return TaskDetectionResult(
            task_type=TaskType.CLASSIFICATION,
            reason=f"Target '{target_column}' is string/object dtype → classification.",
            n_classes=n_unique,
            is_override=False,
        )

    # Numeric target: decide by cardinality
    unique_frac = n_unique / n_rows if n_rows > 0 else 0.0

    if n_unique <= _CLASSIFICATION_UNIQUE_MAX or unique_frac <= _CLASSIFICATION_UNIQUE_FRAC:
        return TaskDetectionResult(
            task_type=TaskType.CLASSIFICATION,
            reason=(
                f"Numeric target '{target_column}' has {n_unique} unique values "
                f"({unique_frac:.1%} of rows) → classified as classification."
            ),
            n_classes=n_unique,
            is_override=False,
        )

    return TaskDetectionResult(
        task_type=TaskType.REGRESSION,
        reason=(
            f"Numeric target '{target_column}' has {n_unique} unique values "
            f"({unique_frac:.1%} of rows) → classified as regression."
        ),
        n_classes=None,
        is_override=False,
    )
