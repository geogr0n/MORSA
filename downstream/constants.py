from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceSpec:
    name: str
    role: str
    experiment: str | None


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("TrueRNA", "reference", None),
    SourceSpec("Mean", "baseline", "mean"),
    SourceSpec("HE2RNA", "baseline", "he2rna"),
    SourceSpec("ViS", "baseline", "vis"),
    SourceSpec("MORSA-Enc", "model", "morsa_enc"),
    SourceSpec("MORSA-Mean", "model", "morsa_mean"),
    SourceSpec("MORSA-HE2RNA", "model", "morsa_he2rna"),
    SourceSpec("MORSA-ViS", "model", "morsa_vis"),
    SourceSpec("MORSA", "model", "morsa"),
)

SOURCE_BY_NAME = {source.name: source for source in SOURCES}
SOURCE_NAMES = tuple(source.name for source in SOURCES)

TCGA_PROGRAM_CANCERS = (
    "BRCA",
    "LUAD",
    "LUSC",
    "COAD",
    "KIRC",
    "KIRP",
    "GBM",
    "HNSC",
    "LIHC",
    "PAAD",
    "PRAD",
    "SKCM",
    "STAD",
    "THCA",
    "UCEC",
    "BLCA",
)

CPTAC_PROGRAM_CANCERS = ("BRCA", "COAD", "GBM", "HNSC", "KIRC", "LUAD", "LUSC", "PAAD")
MSI_CANCERS = ("COAD", "STAD", "UCEC")
SUBTYPE_CANCERS = ("BRCA", "COAD", "GBM")

OUTPUT_COLUMNS = (
    "cohort",
    "cancer",
    "source",
    "source_role",
    "task",
    "metric",
    "value",
    "n_patients",
    "n_genes",
    "fold",
)
