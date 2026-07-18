from __future__ import annotations

import argparse
from pathlib import Path

from .cli_utils import add_common_args, run_cli_task
from .constants import SOURCE_NAMES, SUBTYPE_CANCERS
from .metrics import classification_metrics, confusion_rows, metric_row, order_columns
from .resources import load_cms_templates, load_pam50_centroids, read_gmt, verhaak_gmt
from .scoring import nearest_centroid_labels, top_signature_labels
from .source_loader import align_source_to_reference, load_all_source_splits


TASK = "subtype_report"


def main() -> None:
    parser = argparse.ArgumentParser(description="RNA-seq-derived molecular subtype report recovery.")
    add_common_args(parser)
    args = parser.parse_args()

    def _run() -> list[dict[str, object]]:
        return run_subtype_report(
            data_root=args.data_root,
            resource_root=args.resource_root,
            cancers=SUBTYPE_CANCERS,
            cohorts=("TCGA", "CPTAC"),
            sources=SOURCE_NAMES,
        )

    def _save(rows: list[dict[str, object]], out_dir: Path) -> None:
        order_columns(rows).to_csv(out_dir / "subtype_report_metrics.csv", index=False)

    run_cli_task(task=TASK, args=args, run_fn=_run, save_fn=_save)


def run_subtype_report(
    *,
    data_root: str,
    resource_root: str,
    cancers: tuple[str, ...],
    cohorts: tuple[str, ...],
    sources: tuple[str, ...],
) -> list[dict[str, object]]:
    resources: dict[str, object] = {}
    rows: list[dict[str, object]] = []
    for cohort in cohorts:
        for cancer in cancers:
            fold_frames = load_all_source_splits(data_root, cancer, cohort, sources=sources)
            for fold, source_frames in fold_frames.items():
                reference = source_frames["TrueRNA"]
                subtype_name = _subtype_name(cancer)
                for source in sources:
                    current = source_frames[source]
                    ref_expr, src_expr = align_source_to_reference(reference.expr, current.expr)
                    if ref_expr.empty or src_expr.empty:
                        raise ValueError(f"Empty aligned expression for cohort={cohort}, cancer={cancer}, source={source}, fold={fold}")
                    ref_labels = _labels(cancer, ref_expr, resource_root, resources)
                    src_labels = _labels(cancer, src_expr, resource_root, resources)
                    patients = ref_labels.index.intersection(src_labels.index)
                    metrics = classification_metrics(ref_labels.loc[patients], src_labels.loc[patients])
                    for metric, value in metrics.items():
                        rows.append(
                            metric_row(
                                cohort=cohort,
                                cancer=cancer,
                                source=source,
                                task="Molecular subtype report recovery",
                                metric=metric,
                                value=value,
                                n_patients=len(patients),
                                n_genes=ref_expr.shape[1],
                                fold=fold,
                                subtype_report=subtype_name,
                            )
                        )
                    for item in confusion_rows(ref_labels.loc[patients], src_labels.loc[patients]):
                        rows.append(
                            metric_row(
                                cohort=cohort,
                                cancer=cancer,
                                source=source,
                                task="Molecular subtype report recovery",
                                metric="confusion_matrix",
                                value=item["count"],
                                n_patients=len(patients),
                                n_genes=ref_expr.shape[1],
                                fold=fold,
                                subtype_report=subtype_name,
                                true_label=item["true_label"],
                                pred_label=item["pred_label"],
                            )
                        )
    return rows


def _labels(cancer: str, expr, resource_root: str, cache: dict[str, object]):
    if cancer == "BRCA":
        if "pam50" not in cache:
            cache["pam50"] = load_pam50_centroids(resource_root)
        return nearest_centroid_labels(expr, cache["pam50"])
    if cancer == "COAD":
        if "cms" not in cache:
            cache["cms"] = load_cms_templates(resource_root)
        return nearest_centroid_labels(expr, cache["cms"])
    if cancer == "GBM":
        if "verhaak" not in cache:
            cache["verhaak"] = read_gmt(verhaak_gmt(resource_root))
        return top_signature_labels(expr, cache["verhaak"])
    raise ValueError(f"Unsupported subtype cancer: {cancer}")


def _subtype_name(cancer: str) -> str:
    return {"BRCA": "PAM50", "COAD": "CMS", "GBM": "Verhaak"}.get(cancer, cancer)


if __name__ == "__main__":
    main()
