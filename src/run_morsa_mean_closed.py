from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data_utils import custom_collate_fn, filter_no_features, patient_kfold
from dataset import SuperTileRNADataset
from experiment_config import (
    DEFAULT_BASIS_SEED,
    DEFAULT_RANK_K,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAINING_SEED,
)
from morsa_mean_closed import ClosedMORSAMean
from spex_basis import fit_pca_basis, random_reference_basis, save_basis
from training import evaluate


RIDGE_LAMBDAS = tuple(float(10**power) for power in range(-6, 7))
EXPERIMENT_NAME = "morsa_mean_closed"


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))


def _resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    _sync_if_cuda(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _iter_features(dataset: SuperTileRNADataset):
    if getattr(dataset, "preloaded", False):
        yield from dataset._features
        return
    for index in range(len(dataset)):
        features, _, _, _ = dataset[index]
        yield features


def _precompute_means(dataset: SuperTileRNADataset) -> None:
    means = []
    for features in _iter_features(dataset):
        if features is None:
            raise ValueError("A sample has no cluster_features after input filtering.")
        means.append(features.mean(dim=0).detach().clone())
    dataset.set_precomputed_inputs(means)


def _stack_means(dataset: SuperTileRNADataset) -> np.ndarray:
    values = getattr(dataset, "_precomputed_inputs", None)
    if values is None:
        raise ValueError("Mean-pooled features have not been prepared.")
    return torch.stack(values).cpu().numpy().astype(np.float32, copy=False)


def _fit_ridge(
    features: np.ndarray,
    coordinates: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    z = np.asarray(coordinates, dtype=np.float64)
    x_mean = x.mean(axis=0, keepdims=True)
    z_mean = z.mean(axis=0, keepdims=True)
    x_centered = x - x_mean
    z_centered = z - z_mean

    gram = x_centered @ x_centered.T
    gram.flat[:: gram.shape[0] + 1] += float(ridge_lambda)
    dual = np.linalg.solve(gram, z_centered)
    weight = x_centered.T @ dual
    bias = (z_mean - x_mean @ weight).reshape(-1)
    return weight.astype(np.float32), bias.astype(np.float32)


def _predict(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    output_mu: np.ndarray,
    output_basis: np.ndarray,
) -> np.ndarray:
    coordinates = np.asarray(features, dtype=np.float64) @ weight + bias
    reconstruction = coordinates @ np.asarray(output_basis, dtype=np.float64).T
    return np.asarray(output_mu, dtype=np.float64) + reconstruction


def _mean_gene_pcc(real: np.ndarray, predicted: np.ndarray) -> float:
    real_centered = np.asarray(real, dtype=np.float64) - np.mean(real, axis=0, keepdims=True)
    pred_centered = np.asarray(predicted, dtype=np.float64) - np.mean(predicted, axis=0, keepdims=True)
    numerator = np.sum(real_centered * pred_centered, axis=0)
    denominator = np.sqrt(
        np.sum(real_centered**2, axis=0) * np.sum(pred_centered**2, axis=0)
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )
    return float(np.mean(np.nan_to_num(correlations)))


def _loader(dataset: SuperTileRNADataset, args, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=custom_collate_fn,
    )


def _random_reference(
    *,
    input_dim: int,
    output_mu: np.ndarray,
    rank_k: int,
    seed: int,
    device: torch.device,
) -> ClosedMORSAMean:
    random_mu, random_basis = random_reference_basis(output_mu, rank_k, seed)
    rng = np.random.default_rng(int(seed))
    limit = 1.0 / np.sqrt(float(input_dim))
    weight = rng.uniform(-limit, limit, size=(input_dim, rank_k)).astype(np.float32)
    bias = rng.uniform(-limit, limit, size=rank_k).astype(np.float32)
    return ClosedMORSAMean(
        input_dim=input_dim,
        num_outputs=len(output_mu),
        rank_k=rank_k,
        weight=weight,
        bias=bias,
        output_mu=random_mu,
        output_basis=random_basis,
        device=device,
    )


def _experiment_name(rank_k: int) -> str:
    return EXPERIMENT_NAME if int(rank_k) == DEFAULT_RANK_K else f"{EXPERIMENT_NAME}_k{int(rank_k)}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit analytical MORSA-Mean by ridge regression.")
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--cohort", default="TCGA")
    parser.add_argument("--cancer", default=None)
    parser.add_argument("--filter_no_features", type=int, choices=[0, 1], default=1)
    parser.add_argument("--rank_k", type=int, default=DEFAULT_RANK_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--basis_seed", type=int, default=DEFAULT_BASIS_SEED)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_folds", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def run(args) -> Path:
    rank_k = int(args.rank_k)
    experiment_name = _experiment_name(rank_k)
    experiment_dir = Path(args.save_dir) / str(args.cohort) / experiment_name
    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            f"Experiment directory already contains files: {experiment_dir}. "
            "Use a different --save_dir or remove the directory before starting a new run."
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "RUNNING").write_text("analytical MORSA-Mean\n", encoding="utf-8")

    device = _resolve_device(args.device)
    _seed_everything(args.seed)
    frame = pd.read_csv(args.ref_file)
    if int(args.filter_no_features) == 1:
        frame = filter_no_features(frame, args.feature_path, "cluster_features")
    if frame.empty:
        raise ValueError("No samples remain after filtering unavailable features.")

    rna_columns = [column for column in frame.columns if str(column).startswith("rna_")]
    if not rna_columns:
        raise ValueError("The reference file contains no RNA columns prefixed with 'rna_'.")
    genes = [column[4:] for column in rna_columns]
    train_indices, val_indices, test_indices = patient_kfold(
        frame,
        n_splits=int(args.folds),
        random_state=int(args.split_seed),
    )
    folds = list(zip(train_indices, val_indices, test_indices))
    if int(args.max_folds) > 0:
        folds = folds[: int(args.max_folds)]

    test_results: dict[str, object] = {}
    validation_results: dict[str, object] = {}
    sweep_rows = []
    fold_fit_seconds = []
    fold_total_closed_seconds = []
    fold_mean_preparation_seconds = []
    fold_pca_seconds = []
    fold_peak_memory_mb = []
    fold_inference_seconds = []
    fold_inference_per_wsi_seconds = []
    selected_lambdas = []
    feature_dim = 0
    num_outputs = len(genes)

    for fold, (train_index, val_index, test_index) in enumerate(folds):
        train_frame = frame.iloc[train_index]
        val_frame = frame.iloc[val_index]
        test_frame = frame.iloc[test_index]
        np.save(experiment_dir / f"train_{fold}.npy", np.unique(train_frame.patient_id))
        np.save(experiment_dir / f"val_{fold}.npy", np.unique(val_frame.patient_id))
        np.save(experiment_dir / f"test_{fold}.npy", np.unique(test_frame.patient_id))

        train_dataset = SuperTileRNADataset(train_frame, args.feature_path)
        val_dataset = SuperTileRNADataset(val_frame, args.feature_path)
        test_dataset = SuperTileRNADataset(test_frame, args.feature_path)
        feature_dim = int(train_dataset.feature_dim)

        mean_start = time.perf_counter()
        _precompute_means(train_dataset)
        _precompute_means(val_dataset)
        _precompute_means(test_dataset)
        mean_seconds = float(time.perf_counter() - mean_start)
        fold_mean_preparation_seconds.append(mean_seconds)

        y_train = train_frame[rna_columns].to_numpy(dtype=np.float32)
        pca_start = time.perf_counter()
        basis = fit_pca_basis(y_train, rank_k, int(args.basis_seed))
        coordinates_train = (y_train - basis.mu) @ basis.U_k
        pca_seconds = float(time.perf_counter() - pca_start)
        fold_pca_seconds.append(pca_seconds)
        save_basis(experiment_dir / f"basis_fold_{fold}.npz", basis, fold=fold)

        train_features = _stack_means(train_dataset)
        val_features = _stack_means(val_dataset)
        val_rna = val_frame[rna_columns].to_numpy(dtype=np.float32)

        sweep_start = time.perf_counter()
        best_score = -np.inf
        best_weight = None
        best_bias = None
        best_lambda = None
        for ridge_lambda in RIDGE_LAMBDAS:
            candidate_start = time.perf_counter()
            weight, bias = _fit_ridge(train_features, coordinates_train, ridge_lambda)
            prediction = _predict(val_features, weight, bias, basis.mu, basis.U_k)
            score = _mean_gene_pcc(val_rna, prediction)
            sweep_rows.append(
                {
                    "fold": fold,
                    "ridge_lambda": ridge_lambda,
                    "validation_all_gene_pcc": score,
                    "candidate_seconds": float(time.perf_counter() - candidate_start),
                }
            )
            if score > best_score:
                best_score = score
                best_weight = weight
                best_bias = bias
                best_lambda = ridge_lambda
        sweep_seconds = float(time.perf_counter() - sweep_start)
        if best_weight is None or best_bias is None or best_lambda is None:
            raise RuntimeError("The ridge sweep did not produce a model.")
        fold_fit_seconds.append(sweep_seconds)
        fold_total_closed_seconds.append(mean_seconds + pca_seconds + sweep_seconds)
        selected_lambdas.append(float(best_lambda))

        model = ClosedMORSAMean(
            input_dim=feature_dim,
            num_outputs=num_outputs,
            rank_k=rank_k,
            weight=best_weight,
            bias=best_bias,
            output_mu=basis.mu,
            output_basis=basis.U_k,
            device=device,
        )
        torch.save(model.state_dict(), experiment_dir / f"model_best{'_' + str(fold) if fold else ''}.pt")

        val_loader = _loader(val_dataset, args, device)
        val_pred, val_real, val_wsis, val_projects = evaluate(model, val_loader, suff=f"_{fold}_validation")
        validation_results[f"split_{fold}"] = {
            "real": val_real,
            "preds": val_pred,
            "wsi_file_name": val_wsis,
            "tcga_project": val_projects,
        }

        test_loader = _loader(test_dataset, args, device)
        _reset_peak_memory(device)
        _sync_if_cuda(device)
        inference_start = time.perf_counter()
        prediction, real, wsis, projects = evaluate(model, test_loader, suff=f"_{fold}")
        _sync_if_cuda(device)
        inference_seconds = float(time.perf_counter() - inference_start)
        fold_inference_seconds.append(inference_seconds)
        fold_inference_per_wsi_seconds.append(inference_seconds / float(max(len(wsis), 1)))
        fold_peak_memory_mb.append(_peak_memory_mb(device))
        random_model = _random_reference(
            input_dim=feature_dim,
            output_mu=basis.mu,
            rank_k=rank_k,
            seed=int(args.seed) + 1000 + fold,
            device=device,
        )
        random_prediction, _, _, _ = evaluate(random_model, test_loader, suff=f"_{fold}_random")
        test_results[f"split_{fold}"] = {
            "real": real,
            "preds": prediction,
            "random": random_prediction,
            "wsi_file_name": wsis,
            "tcga_project": projects,
        }

    test_results["genes"] = genes
    validation_results["genes"] = genes
    with (experiment_dir / "test_results.pkl").open("wb") as handle:
        pickle.dump(test_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (experiment_dir / "validation_results.pkl").open("wb") as handle:
        pickle.dump(validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(sweep_rows).to_csv(experiment_dir / "ridge_sweep.csv", index=False)

    fitted_coefficient_count = int(feature_dim * rank_k + rank_k)
    metadata = {
        "experiment": experiment_name,
        "model_type": "morsa_mean_closed",
        "encoder": "mean",
        "head_type": "closed_form",
        "basis_type": "pca",
        "cancer": str(args.cancer or Path(args.ref_file).resolve().parent.name),
        "cohort": str(args.cohort),
        "rank_k": rank_k,
        "training_seed": int(args.seed),
        "split_seed": int(args.split_seed),
        "basis_seed": int(args.basis_seed),
        "num_folds": len(folds),
        "ridge_lambdas": list(RIDGE_LAMBDAS),
        "selected_ridge_lambdas": selected_lambdas,
        "parameter_count": 0,
        "parameter_count_excluding_output_head": 0,
        "parameter_count_including_output_head": fitted_coefficient_count,
        "fitted_coefficient_count": fitted_coefficient_count,
        "fold_train_seconds": fold_fit_seconds,
        "sum_fold_train_seconds": float(np.sum(fold_fit_seconds)),
        "avg_fold_train_seconds": float(np.mean(fold_fit_seconds)),
        "fold_mean_preparation_seconds": fold_mean_preparation_seconds,
        "sum_fold_mean_preparation_seconds": float(np.sum(fold_mean_preparation_seconds)),
        "fold_pca_seconds": fold_pca_seconds,
        "sum_fold_pca_seconds": float(np.sum(fold_pca_seconds)),
        "fold_total_closed_seconds": fold_total_closed_seconds,
        "sum_fold_total_closed_seconds": float(np.sum(fold_total_closed_seconds)),
        "fold_peak_gpu_memory_mb": fold_peak_memory_mb,
        "peak_gpu_memory_mb": float(np.max(fold_peak_memory_mb)),
        "fold_inference_seconds": fold_inference_seconds,
        "fold_inference_time_per_wsi_seconds": fold_inference_per_wsi_seconds,
        "avg_inference_time_per_wsi_seconds": float(np.mean(fold_inference_per_wsi_seconds)),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (experiment_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (experiment_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (experiment_dir / "RUNNING").unlink()
    (experiment_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return experiment_dir


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except Exception:
        experiment_dir = Path(args.save_dir) / str(args.cohort) / _experiment_name(args.rank_k)
        (experiment_dir / "RUNNING").unlink(missing_ok=True)
        raise
    print(f"[complete] {output}")


if __name__ == "__main__":
    main()
