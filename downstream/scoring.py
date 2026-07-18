from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def score_ssgsea(expr: pd.DataFrame, gene_sets: dict[str, list[str]]) -> pd.DataFrame:
    import gseapy as gp

    clean = _clean_expression(expr)
    usable = {
        name: [gene for gene in genes if gene in clean.columns]
        for name, genes in gene_sets.items()
    }
    usable = {name: genes for name, genes in usable.items() if len(genes) >= 1}
    if not usable:
        raise ValueError("No gene sets overlap the expression matrix")
    result = gp.ssgsea(
        data=clean.T,
        gene_sets=usable,
        outdir=None,
        sample_norm_method="rank",
        no_plot=True,
        threads=1,
        min_size=1,
        max_size=10000,
        verbose=False,
    )
    res = result.res2d.copy()
    value_col = "NES" if "NES" in res.columns else "ES"
    scores = res.pivot(index="Name", columns="Term", values=value_col)
    scores = scores.reindex(index=clean.index, columns=list(usable.keys()))
    return scores.apply(pd.to_numeric, errors="coerce")


def estimate_scores(
    expr: pd.DataFrame,
    estimate_gene_sets: dict[str, list[str]],
    common_genes: list[str] | None = None,
) -> pd.DataFrame:
    if common_genes is None:
        scores = _estimate_scores_with_official_r(expr)
    else:
        scores = _estimate_scores_official_python(expr, estimate_gene_sets, common_genes)
    stromal = scores["StromalScore"].astype(float)
    immune = scores["ImmuneScore"].astype(float)
    estimate = scores["ESTIMATEScore"].astype(float)
    purity = scores["TumorPurity"].astype(float)
    return pd.DataFrame(
        {
            "stromal_score": stromal,
            "immune_score": immune,
            "ESTIMATE_score": estimate,
            "tumor_purity": purity,
        },
        index=scores.index,
    )


def _estimate_scores_official_python(
    expr: pd.DataFrame,
    estimate_gene_sets: dict[str, list[str]],
    common_genes: list[str],
) -> pd.DataFrame:
    clean = _clean_expression(expr)
    common = sorted(set(str(gene).upper() for gene in common_genes).intersection(clean.columns))
    if not common:
        raise ValueError("No expression genes overlap the official ESTIMATE common-gene universe.")
    matrix = clean.loc[:, common].T.apply(pd.to_numeric, errors="coerce")
    ranked = matrix.rank(axis=0, method="average", na_option="bottom", ascending=True)
    ranked = ranked * (10000.0 / float(ranked.shape[0]))
    gene_names = np.asarray(ranked.index.astype(str))
    values = ranked.to_numpy(dtype=float)

    stromal_key = _find_gene_set_key(estimate_gene_sets, "stromal")
    immune_key = _find_gene_set_key(estimate_gene_sets, "immune")
    score_rows: dict[str, np.ndarray] = {}
    for output_name, key in (("StromalScore", stromal_key), ("ImmuneScore", immune_key)):
        gene_set = {str(gene).upper() for gene in estimate_gene_sets[key]}
        tag_mask = np.asarray([gene in gene_set for gene in gene_names], dtype=bool)
        if not tag_mask.any():
            score_rows[output_name] = np.full(values.shape[1], np.nan, dtype=float)
            continue
        score_rows[output_name] = _official_estimate_es(values, tag_mask)

    estimate_score = score_rows["StromalScore"] + score_rows["ImmuneScore"]
    tumor_purity = np.cos(0.6049872018 + 0.0001467884 * estimate_score)
    tumor_purity = np.where(tumor_purity < 0, np.nan, tumor_purity)
    return pd.DataFrame(
        {
            "ImmuneScore": score_rows["ImmuneScore"],
            "StromalScore": score_rows["StromalScore"],
            "ESTIMATEScore": estimate_score,
            "TumorPurity": tumor_purity,
        },
        index=ranked.columns.astype(str),
    )


def _official_estimate_es(values: np.ndarray, tag_mask: np.ndarray) -> np.ndarray:
    n_genes, n_samples = values.shape
    nh = int(tag_mask.sum())
    nm = int(n_genes - nh)
    if nh == 0 or nm <= 0:
        return np.full(n_samples, np.nan, dtype=float)
    out = np.empty(n_samples, dtype=float)
    for sample_ix in range(n_samples):
        sample_values = values[:, sample_ix]
        order = np.argsort(-sample_values, kind="mergesort")
        tag = tag_mask[order].astype(float)
        no_tag = 1.0 - tag
        correl = np.abs(sample_values[order]) ** 0.25
        sum_correl = float(correl[tag == 1].sum())
        if sum_correl == 0:
            out[sample_ix] = np.nan
            continue
        f0 = np.cumsum(no_tag / float(nm))
        fn = np.cumsum(tag * correl / sum_correl)
        out[sample_ix] = float(np.sum(fn - f0))
    return out


def _find_gene_set_key(gene_sets: dict[str, list[str]], needle: str) -> str:
    needle = needle.lower()
    for key in gene_sets:
        if needle in str(key).lower():
            return str(key)
    raise ValueError(f"Could not find ESTIMATE gene set containing {needle!r}")


def _estimate_scores_with_official_r(expr: pd.DataFrame) -> pd.DataFrame:
    clean = _clean_expression(expr)
    clean = clean.loc[:, ~pd.Index(clean.columns).duplicated()]
    clean = clean.dropna(axis=1, how="all")
    if clean.empty:
        raise ValueError("ESTIMATE scoring received an empty expression matrix.")
    rscript = _find_rscript()
    script = Path(__file__).resolve().parent / "R" / "run_estimate_scores.R"
    if not script.is_file():
        raise FileNotFoundError(f"Missing ESTIMATE R bridge script: {script}")

    with tempfile.TemporaryDirectory(prefix="morsa_estimate_") as tmp:
        tmp_dir = Path(tmp)
        input_csv = tmp_dir / "estimate_input.csv"
        output_csv = tmp_dir / "estimate_output.csv"
        r_input = clean.T.reset_index().rename(columns={"index": "gene_symbol"})
        r_input.to_csv(input_csv, index=False)
        cmd = [
            rscript,
            str(script),
            "--input_csv",
            str(input_csv),
            "--output_csv",
            str(output_csv),
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Official ESTIMATE R scoring failed with exit code {result.returncode}: {message}")
        scores = pd.read_csv(output_csv)
    required = {"sample_id", "ImmuneScore", "StromalScore", "ESTIMATEScore", "TumorPurity"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Official ESTIMATE output missing columns: {sorted(missing)}")
    scores = scores.set_index("sample_id")
    scores.index = scores.index.astype(str)
    scores = scores.reindex(clean.index.astype(str))
    return scores[["ImmuneScore", "StromalScore", "ESTIMATEScore", "TumorPurity"]].apply(pd.to_numeric, errors="coerce")


def _find_rscript() -> str:
    candidates = []
    env_path = os.environ.get("MORSA_RSCRIPT") or os.environ.get("RSCRIPT")
    if env_path:
        candidates.append(env_path)
    found = shutil.which("Rscript")
    if found:
        candidates.append(found)
    if os.name == "nt":
        program_files = [os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"), os.environ.get("ProgramFiles(x86)")]
        for root in program_files:
            if not root:
                continue
            r_root = Path(root) / "R"
            if not r_root.is_dir():
                continue
            for path in sorted(r_root.glob("R-*/*/Rscript.exe"), reverse=True):
                candidates.append(str(path))
            for path in sorted(r_root.glob("R-*/bin/x64/Rscript.exe"), reverse=True):
                candidates.append(str(path))
            for path in sorted(r_root.glob("R-*/bin/Rscript.exe"), reverse=True):
                candidates.append(str(path))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("Rscript was not found. Set MORSA_RSCRIPT to the Rscript executable that has the R `estimate` package installed.")


def nearest_centroid_labels(expr: pd.DataFrame, centroids: pd.DataFrame) -> pd.Series:
    clean = _clean_expression(expr)
    cent = centroids.copy()
    cent.index = cent.index.astype(str).str.upper()
    genes = clean.columns.intersection(cent.index)
    if len(genes) == 0:
        raise ValueError("No centroid genes overlap the expression matrix")
    x = clean.loc[:, genes]
    c = cent.loc[genes]
    x = _zscore_rows(x)
    c = _zscore_columns(c)
    scores = pd.DataFrame(index=x.index, columns=c.columns, dtype=float)
    for label in c.columns:
        template = c[label].to_numpy(dtype=float)
        denom_template = np.linalg.norm(template)
        if denom_template == 0:
            scores[label] = np.nan
            continue
        denom = np.linalg.norm(x.to_numpy(dtype=float), axis=1) * denom_template
        numerator = x.to_numpy(dtype=float).dot(template)
        scores[label] = np.divide(numerator, denom, out=np.full(len(x), np.nan), where=denom != 0)
    return scores.idxmax(axis=1).astype(str)


def top_signature_labels(expr: pd.DataFrame, signatures: dict[str, list[str]]) -> pd.Series:
    scores = score_ssgsea(expr, signatures)
    return scores.idxmax(axis=1).astype(str)


def _clean_expression(expr: pd.DataFrame) -> pd.DataFrame:
    clean = expr.copy()
    clean.columns = [str(col).upper() for col in clean.columns]
    clean.index = clean.index.astype(str)
    clean = clean.loc[:, ~pd.Index(clean.columns).duplicated()]
    return clean.apply(pd.to_numeric, errors="coerce")


def _find_column(df: pd.DataFrame, needle: str) -> str:
    needle = needle.lower()
    for column in df.columns:
        if needle in str(column).lower():
            return str(column)
    raise ValueError(f"Could not find ESTIMATE column containing {needle!r}")


def _zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    means = np.nanmean(values, axis=1, keepdims=True)
    stds = np.nanstd(values, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return pd.DataFrame((values - means) / stds, index=df.index, columns=df.columns)


def _zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    values = df.to_numpy(dtype=float)
    means = np.nanmean(values, axis=0, keepdims=True)
    stds = np.nanstd(values, axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    return pd.DataFrame((values - means) / stds, index=df.index, columns=df.columns)
