from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .cli_utils import add_common_args, run_cli_task
from .constants import CPTAC_PROGRAM_CANCERS, SOURCE_NAMES, TCGA_PROGRAM_CANCERS
from .metrics import binary_metrics, metric_row, order_columns, safe_mae, safe_pearson, safe_spearman
from .resources import estimate_common_genes, estimate_gmt, hallmark_gmt, read_gmt
from .scoring import estimate_scores, score_ssgsea
from .source_loader import align_source_to_reference, load_all_source_splits


TASK = "program_report"


def main() -> None:
    parser = argparse.ArgumentParser(description="Molecular program report recovery for digital molecular reporting.")
    add_common_args(parser)
    args = parser.parse_args()

    def _run() -> list[dict[str, object]]:
        return run_program_report(
            data_root=args.data_root,
            resource_root=args.resource_root,
            tcga_cancers=TCGA_PROGRAM_CANCERS,
            cptac_cancers=CPTAC_PROGRAM_CANCERS,
            sources=SOURCE_NAMES,
        )

    def _save(rows: list[dict[str, object]], out_dir: Path) -> None:
        order_columns(rows).to_csv(out_dir / "program_report_metrics.csv", index=False)

    run_cli_task(task=TASK, args=args, run_fn=_run, save_fn=_save)


def run_program_report(
    *,
    data_root: str,
    resource_root: str,
    tcga_cancers: tuple[str, ...],
    cptac_cancers: tuple[str, ...],
    sources: tuple[str, ...],
) -> list[dict[str, object]]:
    hallmark_sets = read_gmt(hallmark_gmt(resource_root))
    estimate_sets = read_gmt(estimate_gmt(resource_root))
    estimate_common = _read_estimate_common_genes(estimate_common_genes(resource_root))
    rows: list[dict[str, object]] = []
    for cohort, cancers in (("TCGA", tcga_cancers), ("CPTAC", cptac_cancers)):
        for cancer in cancers:
            fold_frames = load_all_source_splits(data_root, cancer, cohort, sources=sources)
            estimate_score_cache = _precompute_estimate_scores(fold_frames=fold_frames, sources=sources, estimate_sets=estimate_sets, estimate_common=estimate_common)
            for fold, source_frames in fold_frames.items():
                reference = source_frames["TrueRNA"]
                reference_score_cache: dict[tuple[str, tuple[str, ...]], object] = {}
                for source in sources:
                    current = source_frames[source]
                    ref_expr, src_expr = align_source_to_reference(reference.expr, current.expr)
                    if ref_expr.empty or src_expr.empty:
                        raise ValueError(f"Empty aligned expression for cohort={cohort}, cancer={cancer}, source={source}, fold={fold}")
                    n_patients = int(ref_expr.shape[0])
                    n_genes = int(ref_expr.shape[1])
                    hallmark_key = ("hallmark", tuple(ref_expr.columns))
                    if hallmark_key not in reference_score_cache:
                        reference_score_cache[hallmark_key] = score_ssgsea(ref_expr, hallmark_sets)
                    ref_hallmark = reference_score_cache[hallmark_key]
                    src_hallmark = ref_hallmark if source == "TrueRNA" else score_ssgsea(src_expr, hallmark_sets)
                    rows.extend(
                        _report_metric_rows(
                            cohort=cohort,
                            cancer=cancer,
                            source=source,
                            fold=fold,
                            task="Hallmark program report",
                            reference_scores=ref_hallmark,
                            source_scores=src_hallmark,
                            n_patients=n_patients,
                            n_genes=n_genes,
                            include_mae=False,
                        )
                    )
                    ref_estimate = estimate_score_cache[(fold, "TrueRNA")]
                    src_estimate = ref_estimate if source == "TrueRNA" else estimate_score_cache[(fold, source)]
                    rows.extend(
                        _report_metric_rows(
                            cohort=cohort,
                            cancer=cancer,
                            source=source,
                            fold=fold,
                            task="ESTIMATE report",
                            reference_scores=ref_estimate,
                            source_scores=src_estimate,
                            n_patients=n_patients,
                            n_genes=n_genes,
                            include_mae=True,
                        )
                    )
    return rows


def _precompute_estimate_scores(
    *,
    fold_frames: dict[int, dict[str, object]],
    sources: tuple[str, ...],
    estimate_sets: dict[str, list[str]],
    estimate_common: list[str],
) -> dict[tuple[int, str], pd.DataFrame]:
    score_cache: dict[tuple[int, str], pd.DataFrame] = {}
    for source in sources:
        frames = []
        key_map: dict[int, tuple[list[str], list[str]]] = {}
        common_genes: set[str] | None = None
        for fold, source_frames in sorted(fold_frames.items()):
            expr = source_frames[source].expr
            genes = set(map(str, expr.columns))
            common_genes = genes if common_genes is None else common_genes.intersection(genes)
        if not common_genes:
            raise ValueError(f"No common genes available for official ESTIMATE scoring source={source}.")
        genes_order = sorted(common_genes)

        for fold, source_frames in sorted(fold_frames.items()):
            expr = source_frames[source].expr.loc[:, genes_order].copy()
            patient_ids = [str(item) for item in expr.index]
            sample_ids = [f"fold{fold:02d}__{idx:05d}" for idx in range(len(patient_ids))]
            expr.index = sample_ids
            key_map[fold] = (sample_ids, patient_ids)
            frames.append(expr)

        combined = pd.concat(frames, axis=0)
        scored = estimate_scores(combined, estimate_sets, estimate_common)
        for fold, (sample_ids, patient_ids) in key_map.items():
            sub = scored.loc[sample_ids].copy()
            sub.index = pd.Index(patient_ids, name="patient_id")
            score_cache[(fold, source)] = sub
    return score_cache


def _read_estimate_common_genes(path: Path) -> list[str]:
    table = pd.read_csv(path, sep="\t", usecols=["GeneSymbol"])
    return table["GeneSymbol"].astype(str).str.upper().dropna().drop_duplicates().tolist()


def _report_metric_rows(
    *,
    cohort: str,
    cancer: str,
    source: str,
    fold: int,
    task: str,
    reference_scores,
    source_scores,
    n_patients: int,
    n_genes: int,
    include_mae: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    patients = reference_scores.index.intersection(source_scores.index)
    reports = reference_scores.columns.intersection(source_scores.columns)
    for report in reports:
        ref = reference_scores.loc[patients, report].astype(float)
        pred = source_scores.loc[patients, report].astype(float)
        rows.append(metric_row(cohort=cohort, cancer=cancer, source=source, task=task, metric="Pearson", value=safe_pearson(ref, pred), n_patients=n_patients, n_genes=n_genes, fold=fold, report=report))
        rows.append(metric_row(cohort=cohort, cancer=cancer, source=source, task=task, metric="Spearman", value=safe_spearman(ref, pred), n_patients=n_patients, n_genes=n_genes, fold=fold, report=report))
        if include_mae:
            rows.append(metric_row(cohort=cohort, cancer=cancer, source=source, task=task, metric="MAE", value=safe_mae(ref, pred), n_patients=n_patients, n_genes=n_genes, fold=fold, report=report))
        threshold = float(np.nanquantile(ref, 0.75))
        labels = (ref >= threshold).astype(int)
        hard_threshold = float(np.nanquantile(pred, 0.75))
        for metric, value in binary_metrics(labels, pred, hard_threshold=hard_threshold).items():
            rows.append(metric_row(cohort=cohort, cancer=cancer, source=source, task=f"{task} triage", metric=metric, value=value, n_patients=n_patients, n_genes=n_genes, fold=fold, report=report))
    return rows


if __name__ == "__main__":
    main()
