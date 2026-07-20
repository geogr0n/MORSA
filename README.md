# MORSA

MORSA (Morphology-Omics Representation via Structural Alignment) is a code package for morphology-to-transcriptomic-coordinate mapping from H&E whole-slide images. The repository contains preprocessing, model training, evaluation, cross-cohort validation and downstream molecular-reporting code.

The training entry point composes an encoder with an output head. Supported
encoders are:

- `mean`
- `he2rna`
- `vis`
- `morsa_enc` (the full SPD morphology encoder)
- `diag_spd` (the matched diagonal-covariance control)

Supported output heads are `linear`, `spex`, `learned_rank` and
`covnull_spex`. Experiment directory names are generated from this structured
configuration; callers do not supply free-form experiment names.

The primary MORSA comparisons use one global rank, `K=16`. The complete
experiment matrix also evaluates `K={8,16,32,48,64}` on held-out validation
predictions. It contains the backbone benchmark, matched output-head
experiments, rank sensitivity and the DiagSPD morphology-covariance ablation
in one reproducible batch.

## Repository Contents

```text
src/                 Core model definitions, heads, training utilities and data readers
evaluation/          TCGA and cross-cohort transcriptome-recovery evaluation
analysis/            Structural analysis and experiment-summary entry points
configs/             Experiment matrix
downstream/          Molecular program, ESTIMATE, MSI-H and subtype reporting workflows
  R/                  Official ESTIMATE scoring bridge
preprocessing/       Patch extraction, UNI feature extraction and patch-feature clustering
scripts/             Shell entry points for training, evaluation and downstream workflows
```

Large manuscript outputs, raw data, local environments and checkpoints are intentionally excluded from version control.

## Installation

Create an isolated Python environment. Install PyTorch and torchvision first,
using the official PyTorch selector for the CUDA runtime, driver and operating
system of the target machine:

```text
https://pytorch.org/get-started/locally/
```

Then install the remaining dependency list:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-linux.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-linux.txt
```

WSI patch extraction additionally requires the native OpenSlide library. Install
it through the package manager appropriate for the target operating system.

See [ENVIRONMENT.md](ENVIRONMENT.md) for the reference Python/PyTorch
environment, PyTorch/CUDA installation boundary and runtime-measurement scope.

## Data Boundary

This repository does not redistribute raw third-party H&E whole-slide images, RNA-seq files, controlled-access data, or the gated UNI checkpoint. Obtain raw data from the original TCGA/GDC/CPTAC and external project repositories under their own data-use terms.

The processed results and analysis source data supporting the reported experiments are deposited separately in Zenodo. The current release (v1.1.0) has DOI `10.5281/zenodo.21436404`; the Concept DOI for all versions is `10.5281/zenodo.21195955`. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

The expected local project layout is:

```text
project-root/
  data/
    BLCA/
    BRCA/
    ...
    final_result/
    downstream_data/
    cross-cohort/
  morsa/
```

Use environment variables to point the scripts to your local data:

```bash
export MORSA_DATA_ROOT=/path/to/data
export DATA_ROOT=/path/to/data
```

## Checkpoints

MORSA uses frozen UNI features for WSI representation. The UNI checkpoint is not included. Request access to `MahmoodLab/UNI` from Hugging Face, place the downloaded `pytorch_model.bin` outside version control, and pass the path with `--uni_model_path` or `UNI_MODEL_PATH`.

See [UNI_CHECKPOINT.md](UNI_CHECKPOINT.md) for the checkpoint access route, expected file name and checksum used in the reported experiments.

## WSI Preprocessing

Each cancer directory must contain `ref_file.csv` and an `HE/` directory with
the whole-slide images named in the reference file. The standard preprocessing
sequence is:

```bash
DATA_DIR=./data/COAD bash scripts/extract_patches.sh
DATA_DIR=./data/COAD UNI_MODEL_PATH=/path/to/pytorch_model.bin bash scripts/extract_uni_features.sh
DATA_DIR=./data/COAD bash scripts/cluster_patch_features.sh
```

The resulting `features/{PROJECT}/{WSI}/{WSI}.h5` files contain the
`cluster_features` consumed by model training. Patch limits, worker counts,
UNI batch size, cluster count and random seeds can be overridden through the
environment variables defined in the three shell scripts.

## Running the Main Pipeline

The shell runners use portable defaults under `./data` and can be overridden with environment variables.

Train the seed-29 benchmark models for one cohort:

```bash
DATA_DIR=./data/COAD bash scripts/run_train.sh
```

Evaluate trained models:

```bash
DATA_DIR=./data/COAD MODEL_DIR=./data/COAD/output/TCGA bash scripts/run_evaluate.sh
```

Run multi-cancer orchestration:

```bash
DATA_ROOT=./data CANCER_TYPES="BRCA LUAD COAD" EVENT_TYPES="run_train run_evaluate" bash scripts/run_all.sh
```

Run cross-cohort evaluation:

```bash
DATA_ROOT=./data CROSS_COHORT_ROOT=./data/cross-cohort CANCER=LUAD bash scripts/run_cross_cohort_eval.sh
```

Fit and evaluate analytical MORSA-Mean for one cohort:

```bash
DATA_DIR=./data/COAD bash scripts/run_closed_form.sh
```

This entry point computes the fold-specific RNA basis, mean-pools the
morphology prototypes, fits all 13 ridge candidates on the inner-training
subset, selects the penalty on inner-validation all-gene PCC and evaluates the
selected map on the outer test fold. It writes the same `test_results.pkl`
structure used by the standard evaluation pipeline.

## Complete Experiment Matrix

Print the complete 15-configuration, single-seed matrix without launching jobs:

```bash
python scripts/run_experiments.py --data-root ./data
```

Run the complete five-fold trainable-model matrix for one cohort, including
evaluation after each configuration:

```bash
python scripts/run_experiments.py --data-root ./data --cancers PAAD --execute
```

Select individual configurations when running a focused experiment:

```bash
python scripts/run_experiments.py \
  --data-root ./data \
  --cancers PAAD \
  --experiments morsa_k8 morsa_k32 \
  --execute
```

All experiments use training seed 29. The default matrix is ordered so that
the primary MORSA configuration is trained before the matched DiagSPD
experiment that consumes its fold-specific RNA basis. Every run is created
from its declared configuration; existing non-empty output directories cause
an error instead of being overwritten or silently reused.

## Structure Analyses

Run structure analyses for a cohort:

```bash
DATA_ROOT=./data MODEL_ROOT=./data/COAD/output/TCGA RESULTS_ROOT=./data/COAD/output/TCGA/analysis CANCER=COAD bash scripts/run_analysis.sh
```

The available structural-analysis tasks are `rna_structure`,
`basis_stability`, `coordinate_recovery`, `morphology_structure`,
`component_attribution`, `learnedrank_alignment` and `efficiency`. The
`learnedrank_alignment` task compares the final LearnedRank gene-space basis
with the matched fold-specific PCA basis. Both matrices are orthonormalized
with QR before principal angles are computed from the singular values of their
cross-basis matrix. The outputs include learned-versus-PCA similarity and the
theoretical random-subspace expectation `K / G`.

Summarize the cross-cancer head, rank and morphology experiments with:

```bash
python -m analysis.experiment_summary \
  --data-root ./data \
  --output ./data/experiment_summary
```

`CovNull-SPEX` disrupts only the training-RNA covariance used to build the fixed
basis, while retaining the true paired WSI-RNA supervision. The `random` array
stored inside each `test_results.pkl` is an untrained reference prediction used
by the recovered-gene evaluation rule, not a trained experimental arm.

DiagSPD uses `--fixed_basis_dir` to share the exact fold-specific `mu` and `U_k`
generated by canonical MORSA earlier in the same matrix. This keeps the
DiagSPD comparison restricted to the morphology covariance representation.

## Downstream Molecular Reporting

The downstream reporting workflow compares transcriptome sources including `TrueRNA`, `Mean`, `HE2RNA`, `ViS`, `MORSA-Enc`, `MORSA-Mean`, `MORSA-HE2RNA`, `MORSA-ViS` and `MORSA`.

Implemented downstream task families:

- molecular program and ESTIMATE report recovery
- MSI-H triage
- molecular subtype report recovery

Expected downstream resources:

```text
downstream_data/
  prepared/hallmark.gmt
  prepared/estimate_si_genesets.gmt
  prepared/verhaak_gbm_subtypes.gmt
  prepared/pam50_centroids.csv
  prepared/cms_templates.csv
  clinical/cbioportal/{CANCER}/clinical_sample_long.tsv
```

Run all downstream tasks:

```bash
DATA_ROOT=./data \
RESOURCE_ROOT=./data/downstream_data \
RESULTS_ROOT=./data/downstream_data/results/digital_molecular_reporting \
bash scripts/run_downstream.sh
```

## Results Dataset

The processed observations supporting the structural, transcriptome-recovery,
gene-level, analytical, molecular-readout and virtual-cartography analyses are
provided in the Zenodo record listed in
[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). Manuscript working files and final
figure exports are not part of this code package.

Runtime values refer to model fitting after precomputed WSI-derived
`cluster_features` are available. Neural values sum the five optimizer-based
training loops; analytical MORSA-Mean sums the complete 13-candidate ridge
sweeps. The matched trained-versus-analytical MORSA-Mean comparison excludes
prototype averaging and fold-specific PCA from both measurements. All values
exclude raw WSI tiling, UNI feature extraction and k-means feature
summarization.

## Licence and Provenance

This code release is distributed under the **GNU General Public License,
version 3 or later**. See:

- [LICENSE](LICENSE)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

The HE2RNA-style baseline path is adapted from `gevaertlab/sequoia-pub/src/he2rna.py`,
which retains the Owkin HE2RNA GPL notice. SEQUOIA/ViS-style components are
adapted from `gevaertlab/sequoia-pub`.

## Citation

Citation information will be added upon publication.
