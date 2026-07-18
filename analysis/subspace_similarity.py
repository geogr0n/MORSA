from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def orthonormalize_basis(matrix: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    basis = np.asarray(matrix, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] < basis.shape[1] or basis.shape[1] < 1:
        raise ValueError(f"{label} must be a nonempty ambient-dimension x rank matrix, got {basis.shape}.")
    if not np.isfinite(basis).all():
        raise ValueError(f"{label} contains non-finite values.")
    q, r = np.linalg.qr(basis, mode="reduced")
    if np.linalg.matrix_rank(r) != basis.shape[1]:
        raise ValueError(f"{label} is rank deficient.")
    return q, r


def subspace_similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    if np.asarray(left).shape != np.asarray(right).shape:
        raise ValueError(f"Subspace bases must have the same shape, got {np.shape(left)} and {np.shape(right)}.")
    q_left, _ = orthonormalize_basis(left, label="left basis")
    q_right, _ = orthonormalize_basis(right, label="right basis")
    singular_values = np.clip(np.linalg.svd(q_left.T @ q_right, compute_uv=False), 0.0, 1.0)
    angles = np.degrees(np.arccos(singular_values))
    return {
        "projection_similarity": float(np.mean(singular_values**2)),
        "mean_cosine_principal_vectors": float(np.mean(singular_values)),
        "mean_principal_angle_deg": float(np.mean(angles)),
        "min_principal_angle_deg": float(np.min(angles)),
        "max_principal_angle_deg": float(np.max(angles)),
        "principal_angles_lt_30_deg": int(np.sum(angles < 30.0)),
        "principal_angles_lt_45_deg": int(np.sum(angles < 45.0)),
    }


def _checkpoint_path(experiment_dir: Path, fold: int) -> Path:
    suffix = "" if fold == 0 else f"_{fold}"
    return experiment_dir / f"model_best{suffix}.pt"


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing LearnedRank checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")
    return payload


def _find_learned_rank_experiment(model_root: Path) -> tuple[Path, dict]:
    if not model_root.is_dir():
        raise FileNotFoundError(f"Model root not found: {model_root}")
    experiments = []
    for experiment_dir in sorted(model_root.iterdir()):
        metadata_path = experiment_dir / "metadata.json"
        if not experiment_dir.is_dir() or not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("model_type", "")).lower() != "morsa_enc":
            continue
        if str(metadata.get("head_type", "")).lower() != "learned_rank":
            continue
        experiments.append((experiment_dir, metadata))
    if len(experiments) != 1:
        raise ValueError(
            "Expected exactly one morsa_enc + learned_rank experiment under "
            f"{model_root}, found {len(experiments)}."
        )
    return experiments[0]


def run_learned_vs_pca_subspace(
    *,
    cancer: str,
    model_root: Path,
    output_dir: Path,
    canonical_experiment: str = "morsa",
) -> tuple[Path, Path]:
    canonical_dir = model_root / canonical_experiment
    if not canonical_dir.is_dir():
        raise FileNotFoundError(f"Canonical PCA-basis experiment not found: {canonical_dir}")
    basis_paths = sorted(
        canonical_dir.glob("basis_fold_*.npz"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if not basis_paths:
        raise FileNotFoundError(f"No basis_fold_<fold>.npz files found in {canonical_dir}.")

    pca_by_fold: dict[int, np.ndarray] = {}
    for basis_path in basis_paths:
        fold = int(basis_path.stem.rsplit("_", 1)[1])
        with np.load(basis_path, allow_pickle=False) as payload:
            if "U_k" not in payload:
                raise KeyError(f"Canonical basis file lacks U_k: {basis_path}")
            pca_by_fold[fold] = np.asarray(payload["U_k"], dtype=np.float64)

    experiment_dir, metadata = _find_learned_rank_experiment(model_root)
    if metadata.get("training_seed") is None or metadata.get("rank_k") is None:
        raise ValueError(f"LearnedRank metadata lacks training_seed or rank_k: {experiment_dir}")
    seed = int(metadata["training_seed"])
    rank_k = int(metadata["rank_k"])
    rows: list[dict[str, object]] = []
    for fold, pca_basis in sorted(pca_by_fold.items()):
        checkpoint_path = _checkpoint_path(experiment_dir, fold)
        state = _load_state_dict(checkpoint_path)
        key = "linear_head.V"
        if key not in state:
            raise KeyError(f"Checkpoint lacks {key}: {checkpoint_path}")
        learned_basis = state[key].detach().cpu().numpy().astype(np.float64, copy=False)
        if learned_basis.shape != pca_basis.shape:
            raise ValueError(
                f"LearnedRank/PCA shape mismatch for {experiment_dir.name}, fold {fold}: "
                f"{learned_basis.shape} vs {pca_basis.shape}."
            )
        if learned_basis.shape[1] != rank_k:
            raise ValueError(
                f"Metadata rank_k={rank_k} disagrees with checkpoint basis {learned_basis.shape}."
            )
        q_learned, r_learned = orthonormalize_basis(
            learned_basis,
            label=f"{experiment_dir.name} fold {fold} learned basis",
        )
        q_pca, _ = orthonormalize_basis(
            pca_basis,
            label=f"{canonical_experiment} fold {fold} PCA basis",
        )
        metrics = subspace_similarity(q_learned, q_pca)
        random_expectation = float(rank_k / learned_basis.shape[0])
        rows.append(
            {
                "cancer": cancer,
                "experiment": experiment_dir.name,
                "training_seed": seed,
                "fold": fold,
                "num_genes": int(learned_basis.shape[0]),
                "rank_k": rank_k,
                **metrics,
                "gain_over_random_expectation": float(
                    metrics["projection_similarity"] - random_expectation
                ),
                "random_projection_similarity_expectation": random_expectation,
                "learned_basis_condition_number": float(np.linalg.cond(r_learned)),
                "learned_checkpoint": f"{experiment_dir.name}/{checkpoint_path.name}",
                "pca_basis": f"{canonical_experiment}/basis_fold_{fold}.npz",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "learned_vs_pca_subspace.csv"
    raw = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    raw.to_csv(raw_path, index=False)

    summary = {
        "cancer": cancer,
        "n_fold_comparisons": int(len(raw)),
        "n_folds": int(raw["fold"].nunique()),
        "rank_k": int(raw["rank_k"].iloc[0]),
        "mean_learned_vs_pca_projection_similarity": float(raw["projection_similarity"].mean()),
        "sd_learned_vs_pca_projection_similarity": float(raw["projection_similarity"].std(ddof=1)),
        "mean_learned_vs_pca_principal_angle_deg": float(raw["mean_principal_angle_deg"].mean()),
        "mean_max_learned_vs_pca_principal_angle_deg": float(raw["max_principal_angle_deg"].mean()),
        "random_projection_similarity_expectation": float(
            raw["random_projection_similarity_expectation"].mean()
        ),
        "mean_gain_over_random_expectation": float(raw["gain_over_random_expectation"].mean()),
        "mean_learned_basis_condition_number": float(raw["learned_basis_condition_number"].mean()),
    }
    summary_path = output_dir / "summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return raw_path, summary_path
