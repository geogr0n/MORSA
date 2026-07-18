from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .cli_utils import add_common_args, run_cli_task
from .clinical import load_msi_table
from .constants import MSI_CANCERS, SOURCE_NAMES
from .metrics import (
    binary_metrics,
    metric_row,
    order_columns,
    ppv_at_top_fraction,
    threshold_for_balanced_accuracy,
    threshold_for_specificity,
)
from .source_loader import (
    align_source_to_reference,
    load_all_source_splits,
    load_real_expression_from_ref,
    load_tcga_fold_patients,
)


TASK = "msi_triage"
MAX_GENES = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description="MSI-H triage from RNA-seq-like digital molecular reports.")
    add_common_args(parser)
    args = parser.parse_args()

    def _run() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        return run_msi_triage(
            data_root=args.data_root,
            resource_root=args.resource_root,
            cancers=MSI_CANCERS,
            sources=SOURCE_NAMES,
            max_genes=MAX_GENES,
        )

    def _save(result: tuple[list[dict[str, object]], list[dict[str, object]]], out_dir: Path) -> None:
        rows, sensor_rows = result
        order_columns(rows).to_csv(out_dir / "msi_triage_metrics.csv", index=False)
        order_columns(sensor_rows).to_csv(out_dir / "msi_sensor_sensitivity_check.csv", index=False)

    run_cli_task(task=TASK, args=args, run_fn=_run, save_fn=_save)


def run_msi_triage(
    *,
    data_root: str,
    resource_root: str,
    cancers: tuple[str, ...],
    sources: tuple[str, ...],
    max_genes: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    sensor_rows: list[dict[str, object]] = []
    for cancer in cancers:
        labels = load_msi_table(resource_root, cancer)
        if "MSI_H_MSIsensor" not in labels.columns:
            raise ValueError(f"Missing MSI_H_MSIsensor column for cancer={cancer}.")
        fold_frames = load_all_source_splits(data_root, cancer, "TCGA", sources=sources)
        for fold, source_frames in fold_frames.items():
            train_patients, _ = load_tcga_fold_patients(data_root, cancer, fold)
            train_expr = load_real_expression_from_ref(data_root, cancer, patients=train_patients)
            train_labels = labels["MSI_H_MANTIS"].dropna()
            train_common = train_expr.index.intersection(train_labels.index)
            train_expr = train_expr.loc[train_common]
            y_train = train_labels.loc[train_common].astype(int)
            if len(np.unique(y_train)) < 2:
                raise ValueError(f"MSI-H train labels contain fewer than two classes for cancer={cancer}, fold={fold}.")
            reference = source_frames["TrueRNA"]
            model_cache = {}
            for source in sources:
                current = source_frames[source]
                _, test_expr = align_source_to_reference(reference.expr, current.expr)
                test_labels = labels["MSI_H_MANTIS"].dropna()
                test_common = test_expr.index.intersection(test_labels.index)
                if len(test_common) == 0:
                    raise ValueError(f"No test patients with MSI labels for cancer={cancer}, source={source}, fold={fold}.")
                test_expr = test_expr.loc[test_common]
                y_test = test_labels.loc[test_common].astype(int)
                genes = train_expr.columns.intersection(test_expr.columns)
                selected_genes = _select_top_variance_genes(train_expr.loc[:, genes], max_genes)
                cache_key = tuple(selected_genes)
                if cache_key not in model_cache:
                    model = _fit_logistic(train_expr.loc[:, selected_genes], y_train)
                    train_prob = model.predict_proba(train_expr.loc[:, selected_genes])[:, 1]
                    threshold_bal = threshold_for_balanced_accuracy(y_train, train_prob)
                    threshold_spec = threshold_for_specificity(y_train, train_prob, min_specificity=0.90)
                    model_cache[cache_key] = (model, threshold_bal, threshold_spec)
                model, threshold_bal, threshold_spec = model_cache[cache_key]
                test_prob = model.predict_proba(test_expr.loc[:, selected_genes])[:, 1]
                for metric, value in _msi_metrics(y_test, test_prob, threshold_bal, threshold_spec).items():
                    rows.append(
                        metric_row(
                            cohort="TCGA",
                            cancer=cancer,
                            source=source,
                            task="MSI-H triage",
                            metric=metric,
                            value=value,
                            n_patients=len(y_test),
                            n_genes=len(selected_genes),
                            fold=fold,
                            label_source="MANTIS>0.4",
                        )
                    )
                sensor = labels["MSI_H_MSIsensor"].dropna()
                sensor_common = test_expr.index.intersection(sensor.index)
                if len(sensor_common) == 0:
                    raise ValueError(f"No test patients with MSIsensor labels for cancer={cancer}, source={source}, fold={fold}.")
                sensor_prob = pd.Series(test_prob, index=test_expr.index).loc[sensor_common]
                for metric, value in binary_metrics(sensor.loc[sensor_common].astype(int), sensor_prob).items():
                    sensor_rows.append(
                        metric_row(
                            cohort="TCGA",
                            cancer=cancer,
                            source=source,
                            task="MSI-H triage sensitivity check",
                            metric=metric,
                            value=value,
                            n_patients=len(sensor_common),
                            n_genes=len(selected_genes),
                            fold=fold,
                            label_source="MSIsensor>=10",
                        )
                    )
    return rows, sensor_rows


def _fit_logistic(train_x: pd.DataFrame, y_train: pd.Series):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = imputer.fit_transform(train_x)
    x = scaler.fit_transform(x)
    clf = LogisticRegression(
        l1_ratio=0.5,
        solver="saga",
        class_weight="balanced",
        max_iter=10000,
        tol=1e-3,
        random_state=29,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(x, y_train)

    class Model:
        def predict_proba(self, expr: pd.DataFrame):
            return clf.predict_proba(scaler.transform(imputer.transform(expr)))

    return Model()


def _select_top_variance_genes(expr: pd.DataFrame, max_genes: int) -> list[str]:
    variances = expr.var(axis=0, skipna=True).sort_values(ascending=False)
    variances = variances[np.isfinite(variances)]
    if max_genes and len(variances) > max_genes:
        variances = variances.iloc[:max_genes]
    return list(variances.index)


def _msi_metrics(y_true: pd.Series, score: np.ndarray, threshold_bal: float, threshold_spec: float) -> dict[str, float]:
    metrics = binary_metrics(y_true, score, hard_threshold=threshold_bal)
    y = np.asarray(y_true, dtype=int)
    pred_spec = (np.asarray(score, dtype=float) >= threshold_spec).astype(int)
    positives = y == 1
    metrics["sensitivity_at_90_specificity"] = float(np.mean(pred_spec[positives] == 1)) if positives.any() else math.nan
    metrics["PPV@top10%"] = ppv_at_top_fraction(y, score, fraction=0.10)
    return metrics


if __name__ == "__main__":
    main()
