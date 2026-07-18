from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import MissingItem, ResourceError


def require_file(path: Path, context: str) -> Path:
    if not path.is_file():
        raise ResourceError(f"Missing required resource: {path}", [MissingItem(str(path), "missing file", context)])
    return path


def read_gmt(path: Path) -> dict[str, list[str]]:
    require_file(path, "gene set resource")
    gene_sets: dict[str, list[str]] = {}
    for line in _read_text_lines(path):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        name = parts[0]
        genes = [gene.strip().upper() for gene in parts[2:] if gene.strip()]
        if genes:
            gene_sets[name] = sorted(set(genes))
    if not gene_sets:
        raise ResourceError(f"No gene sets were parsed from {path}", [MissingItem(str(path), "empty GMT", "gene set resource")])
    return gene_sets


def _read_text_lines(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            with path.open("r", encoding=encoding) as handle:
                lines = handle.readlines()
            if lines and "\t" in lines[0]:
                return lines
        except UnicodeError:
            continue
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.readlines()


def hallmark_gmt(resource_root: str | Path) -> Path:
    return require_file(Path(resource_root) / "prepared" / "hallmark.gmt", "Hallmark gene sets")


def estimate_gmt(resource_root: str | Path) -> Path:
    return require_file(Path(resource_root) / "prepared" / "estimate_si_genesets.gmt", "ESTIMATE stromal/immune signatures")


def estimate_common_genes(resource_root: str | Path) -> Path:
    return require_file(Path(resource_root) / "prepared" / "estimate_common_genes.txt", "ESTIMATE common genes")


def verhaak_gmt(resource_root: str | Path) -> Path:
    return require_file(Path(resource_root) / "prepared" / "verhaak_gbm_subtypes.gmt", "Verhaak GBM subtype signatures")


def load_pam50_centroids(resource_root: str | Path) -> pd.DataFrame:
    path = require_file(Path(resource_root) / "prepared" / "pam50_centroids.csv", "PAM50 centroids")
    centroids = pd.read_csv(path)
    if "gene_symbol" not in centroids.columns:
        raise ResourceError(
            f"PAM50 centroid file is missing required column `gene_symbol`: {path}",
            [MissingItem(str(path), "missing column gene_symbol", "PAM50")],
        )
    centroids["gene_symbol"] = centroids["gene_symbol"].astype(str).str.upper()
    centroids = centroids.set_index("gene_symbol")
    centroids = centroids.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if centroids.empty or centroids.shape[1] < 2:
        raise ResourceError(f"PAM50 centroid file is not usable: {path}", [MissingItem(str(path), "invalid centroid table", "PAM50")])
    return centroids


def load_cms_templates(resource_root: str | Path) -> pd.DataFrame:
    path = require_file(Path(resource_root) / "prepared" / "cms_templates.csv", "CMS templates")
    templates = pd.read_csv(path)
    if "gene_symbol" not in templates.columns:
        raise ResourceError(
            f"CMS template file is missing required column `gene_symbol`: {path}",
            [MissingItem(str(path), "missing column gene_symbol", "CMS")],
        )
    wide = templates.copy()
    wide["gene_symbol"] = wide["gene_symbol"].astype(str).str.upper()
    wide = wide.set_index("gene_symbol")
    wide = wide.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if wide.empty or wide.shape[1] < 2:
        raise ResourceError(f"CMS template file is not usable: {path}", [MissingItem(str(path), "invalid template table", "CMS")])
    return wide
