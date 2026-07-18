from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class BasisResult:
    mu: np.ndarray
    U_k: np.ndarray
    diagnostics: dict[str, float | int | str]


def _as_expression_matrix(y_train: np.ndarray) -> np.ndarray:
    y = np.asarray(y_train, dtype=np.float32)
    if y.ndim != 2:
        raise ValueError(f"Expected a 2D RNA matrix, got shape={y.shape}.")
    if y.shape[0] < 2 or y.shape[1] < 1:
        raise ValueError(f"RNA matrix is too small for PCA: shape={y.shape}.")
    if not np.isfinite(y).all():
        raise ValueError("RNA matrix contains NaN or infinite values.")
    return y


def max_pca_rank(y_train: np.ndarray) -> int:
    y = _as_expression_matrix(y_train)
    return min(int(y.shape[1]), int(y.shape[0]) - 1)


def _validate_rank(y: np.ndarray, rank_k: int) -> int:
    rank = int(rank_k)
    maximum = max_pca_rank(y)
    if rank < 1 or rank > maximum:
        raise ValueError(
            f"rank_k={rank} is invalid for train_samples={y.shape[0]} and genes={y.shape[1]}; "
            f"the centered PCA maximum is {maximum}."
        )
    return rank


def _orthonormality_error(U_k: np.ndarray) -> float:
    gram = np.asarray(U_k, dtype=np.float64).T @ np.asarray(U_k, dtype=np.float64)
    return float(np.max(np.abs(gram - np.eye(gram.shape[0]))))


def _offdiag_covariance_frobenius(y: np.ndarray) -> float:
    centered = np.asarray(y, dtype=np.float64) - np.mean(y, axis=0, keepdims=True)
    denom = float(max(centered.shape[0] - 1, 1))
    patient_gram = centered @ centered.T
    covariance_frobenius_sq = float(np.sum(patient_gram * patient_gram) / (denom * denom))
    variances = np.sum(centered * centered, axis=0) / denom
    diagonal_frobenius_sq = float(np.sum(variances * variances))
    return float(np.sqrt(max(covariance_frobenius_sq - diagonal_frobenius_sq, 0.0)))


def fit_pca_basis(y_train: np.ndarray, rank_k: int, seed: int) -> BasisResult:
    y = _as_expression_matrix(y_train)
    rank = _validate_rank(y, rank_k)
    pca = PCA(n_components=rank, random_state=int(seed))
    pca.fit(y)
    U_k = pca.components_.T.astype(np.float32)
    diagnostics = {
        "basis_type": "pca",
        "rank_k": rank,
        "basis_seed": int(seed),
        "train_samples": int(y.shape[0]),
        "num_genes": int(y.shape[1]),
        "orthonormality_max_abs_error": _orthonormality_error(U_k),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return BasisResult(pca.mean_.astype(np.float32), U_k, diagnostics)


def independently_permute_genes(y_train: np.ndarray, seed: int) -> np.ndarray:
    y = _as_expression_matrix(y_train)
    rng = np.random.default_rng(int(seed))
    permuted = np.empty_like(y)
    for gene_index in range(y.shape[1]):
        permuted[:, gene_index] = y[rng.permutation(y.shape[0]), gene_index]
    return permuted


def fit_covnull_basis(y_train: np.ndarray, rank_k: int, seed: int) -> BasisResult:
    y = _as_expression_matrix(y_train)
    rank = _validate_rank(y, rank_k)
    y_null = independently_permute_genes(y, seed)

    y64 = y.astype(np.float64)
    y_null64 = y_null.astype(np.float64)
    mean_error = float(np.max(np.abs(np.mean(y64, axis=0) - np.mean(y_null64, axis=0))))
    variance_error = float(np.max(np.abs(np.var(y64, axis=0) - np.var(y_null64, axis=0))))
    before = _offdiag_covariance_frobenius(y)
    after = _offdiag_covariance_frobenius(y_null)

    pca = PCA(n_components=rank, random_state=int(seed))
    pca.fit(y_null)
    U_k = pca.components_.T.astype(np.float32)
    diagnostics = {
        "basis_type": "covariance_null_pca",
        "rank_k": rank,
        "basis_seed": int(seed),
        "train_samples": int(y.shape[0]),
        "num_genes": int(y.shape[1]),
        "marginal_mean_max_abs_error": mean_error,
        "marginal_variance_max_abs_error": variance_error,
        "offdiag_covariance_frobenius_before": before,
        "offdiag_covariance_frobenius_after": after,
        "offdiag_covariance_ratio": float(after / before) if before > 0 else 0.0,
        "orthonormality_max_abs_error": _orthonormality_error(U_k),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return BasisResult(pca.mean_.astype(np.float32), U_k, diagnostics)


def random_reference_basis(mu: np.ndarray, rank_k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mu, dtype=np.float32)
    if mean.ndim != 1:
        raise ValueError(f"Expected a 1D mean vector, got shape={mean.shape}.")
    rng = np.random.default_rng(int(seed))
    random_mu = (
        rng.standard_normal(mean.shape[0]) * (float(np.std(mean)) + 1e-8) + float(np.mean(mean))
    ).astype(np.float32)
    q, _ = np.linalg.qr(rng.standard_normal((mean.shape[0], int(rank_k))))
    return random_mu, q.astype(np.float32)


def save_basis(path: str, result: BasisResult, **metadata) -> None:
    np.savez_compressed(
        path,
        mu=result.mu,
        U_k=result.U_k,
        **{**result.diagnostics, **metadata},
    )
