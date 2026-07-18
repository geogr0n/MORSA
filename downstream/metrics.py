from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .constants import OUTPUT_COLUMNS, SOURCE_BY_NAME


def metric_row(
    *,
    cohort: str,
    cancer: str,
    source: str,
    task: str,
    metric: str,
    value: float | int | str | None,
    n_patients: int,
    n_genes: int,
    fold: int | str,
    **extra: object,
) -> dict[str, object]:
    if source not in SOURCE_BY_NAME:
        raise ValueError(f"Unknown source: {source}")
    role = SOURCE_BY_NAME[source].role
    row: dict[str, object] = {
        "cohort": cohort,
        "cancer": cancer,
        "source": source,
        "source_role": role,
        "task": task,
        "metric": metric,
        "value": value,
        "n_patients": n_patients,
        "n_genes": n_genes,
        "fold": fold,
    }
    row.update(extra)
    return row


def order_columns(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    first = [col for col in OUTPUT_COLUMNS if col in df.columns]
    rest = [col for col in df.columns if col not in first]
    return df[first + rest]


def safe_pearson(x: Iterable[float], y: Iterable[float]) -> float:
    a, b = _clean_pair(x, y)
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return math.nan
    return float(stats.pearsonr(a, b).statistic)


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float:
    a, b = _clean_pair(x, y)
    if len(a) < 3 or np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return math.nan
    return float(stats.spearmanr(a, b).statistic)


def safe_mae(x: Iterable[float], y: Iterable[float]) -> float:
    a, b = _clean_pair(x, y)
    if len(a) == 0:
        return math.nan
    return float(np.mean(np.abs(a - b)))


def binary_metrics(y_true: Iterable[int], score: Iterable[float], hard_threshold: float | None = None) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    s = np.asarray(list(score), dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask].astype(int)
    s = s[mask].astype(float)
    out = {
        "AUROC": math.nan,
        "AUPRC": math.nan,
        "balanced_accuracy": math.nan,
        "sensitivity_at_90_specificity": math.nan,
        "PPV@top10%": math.nan,
    }
    if len(y) == 0 or len(np.unique(y)) < 2:
        return out
    out["AUROC"] = float(roc_auc_score(y, s))
    out["AUPRC"] = float(average_precision_score(y, s))
    threshold = hard_threshold if hard_threshold is not None else float(np.quantile(s, 0.75))
    pred = (s >= threshold).astype(int)
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    out["sensitivity_at_90_specificity"] = sensitivity_at_specificity(y, s, min_specificity=0.90)
    out["PPV@top10%"] = ppv_at_top_fraction(y, s, fraction=0.10)
    return out


def threshold_for_balanced_accuracy(y_true: Iterable[int], score: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(score), dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0:
        return math.nan
    thresholds = np.unique(s)
    if len(thresholds) > 200:
        thresholds = np.quantile(s, np.linspace(0, 1, 200))
    best_threshold = float(np.median(s))
    best_score = -1.0
    for threshold in thresholds:
        score_value = balanced_accuracy_score(y, (s >= threshold).astype(int))
        if score_value > best_score:
            best_score = score_value
            best_threshold = float(threshold)
    return best_threshold


def threshold_for_specificity(y_true: Iterable[int], score: Iterable[float], min_specificity: float = 0.90) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(score), dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    fpr, _, thresholds = roc_curve(y, s)
    specificity = 1.0 - fpr
    ok = np.where(specificity >= min_specificity)[0]
    if len(ok) == 0:
        return float(np.max(s) + 1e-12)
    return float(thresholds[ok[-1]])


def sensitivity_at_specificity(y_true: Iterable[int], score: Iterable[float], min_specificity: float = 0.90) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(score), dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return math.nan
    fpr, tpr, _ = roc_curve(y, s)
    specificity = 1.0 - fpr
    valid = tpr[specificity >= min_specificity]
    if len(valid) == 0:
        return 0.0
    return float(np.max(valid))


def ppv_at_top_fraction(y_true: Iterable[int], score: Iterable[float], fraction: float = 0.10) -> float:
    y = np.asarray(list(y_true), dtype=float)
    s = np.asarray(list(score), dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask].astype(int)
    s = s[mask]
    if len(y) == 0:
        return math.nan
    n_top = max(1, int(math.ceil(len(y) * fraction)))
    idx = np.argsort(-s)[:n_top]
    return float(np.mean(y[idx]))


def classification_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, float]:
    truth = np.asarray(list(y_true), dtype=object)
    pred = np.asarray(list(y_pred), dtype=object)
    mask = pd.notna(truth) & pd.notna(pred)
    truth = truth[mask]
    pred = pred[mask]
    if len(truth) == 0:
        return {"accuracy": math.nan, "balanced_accuracy": math.nan, "macro_F1": math.nan, "Cohen_kappa": math.nan}
    labels = sorted(set(truth) | set(pred))
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_F1": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "Cohen_kappa": float(cohen_kappa_score(truth, pred, labels=labels)),
    }


def confusion_rows(y_true: Iterable[str], y_pred: Iterable[str]) -> list[dict[str, object]]:
    truth = np.asarray(list(y_true), dtype=object)
    pred = np.asarray(list(y_pred), dtype=object)
    mask = pd.notna(truth) & pd.notna(pred)
    truth = truth[mask]
    pred = pred[mask]
    labels = sorted(set(truth) | set(pred))
    if not labels:
        return []
    cm = confusion_matrix(truth, pred, labels=labels)
    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append({"true_label": true_label, "pred_label": pred_label, "count": int(cm[i, j])})
    return rows


def _clean_pair(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]
