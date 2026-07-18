from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiment_config import (
    DEFAULT_BASIS_SEED,
    DEFAULT_RANK_K,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAINING_SEED,
    HEAD_TYPES,
    MODEL_TYPES,
    ExperimentSpec,
)
from he2rna_backbone import HE2RNABackbone
from mean_backbone import MeanBackbone
from morsa_enc_backbone import MORSAEncoderBackbone
from dataset import SuperTileRNADataset
from spex_basis import (
    BasisResult,
    fit_covnull_basis,
    fit_pca_basis,
    random_reference_basis,
    save_basis,
)
from vis_backbone import ViS
from training import evaluate, train
from data_utils import custom_collate_fn, filter_no_features, patient_kfold


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))


def _resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_cuda_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    _sync_if_cuda(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _param_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def _decoder_param_prefixes(model: nn.Module) -> tuple[str, ...]:
    prefixes = []
    if hasattr(model, "linear_head"):
        prefixes.append("linear_head.")
    if getattr(model, "classifier", None) is not None:
        prefixes.append("classifier.")
    return tuple(prefixes)


def _backbone_param_counts(model: nn.Module) -> tuple[int, int]:
    excluded = _decoder_param_prefixes(model)
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        if excluded and any(name.startswith(prefix) for prefix in excluded):
            continue
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return int(total), int(trainable)


def _backbone_param_count(model: nn.Module) -> int:
    return _backbone_param_counts(model)[1]


def _print_model_summary(model: nn.Module, *, model_type: str, decoder: str) -> None:
    backbone_params = _backbone_param_count(model)
    total_params, trainable_params = _param_counts(model)
    print(
        f"[model] type={model_type} class={model.__class__.__name__} head={decoder} "
        f"backbone_trainable={backbone_params:,} total={total_params:,} trainable={trainable_params:,}",
        flush=True,
    )


def _iter_dataset_features(dataset: SuperTileRNADataset):
    if getattr(dataset, "preloaded", False):
        yield from dataset._features
        return
    for index in range(len(dataset)):
        feature, _, _, _ = dataset[index]
        yield feature


def _precompute_mean_inputs(dataset: SuperTileRNADataset) -> None:
    cached = []
    for feature in _iter_dataset_features(dataset):
        cached.append(None if feature is None else feature.mean(dim=0).detach().clone())
    dataset.set_precomputed_inputs(cached)


def _build_model(
    args,
    *,
    model_type: str,
    num_outputs: int,
    feature_dim: int,
    device: torch.device,
    output_mu=None,
    output_basis=None,
) -> nn.Module:
    head_type = str(args.head_type).lower()
    rank_k = int(args.rank_k)
    morsa_r = int(args.morsa_r)

    if model_type == "vis":
        return ViS(
            num_outputs=num_outputs,
            input_dim=feature_dim,
            depth=int(args.depth),
            nheads=int(args.num_heads),
            dimensions_f=64,
            dimensions_c=64,
            dimensions_s=64,
            device=device,
            head_type=head_type,
            rank_k=rank_k,
            output_mu=output_mu,
            output_basis=output_basis,
        )
    if model_type in {"morsa_enc", "diag_spd"}:
        return MORSAEncoderBackbone(
            input_dim=feature_dim,
            num_outputs=num_outputs,
            device=device,
            head_type=head_type,
            rank_k=rank_k,
            output_mu=output_mu,
            output_basis=output_basis,
            diagonal_covariance=(model_type == "diag_spd"),
            morsa_r=morsa_r,
            morsa_eps=float(args.morsa_eps),
            morsa_latent_dim=int(args.morsa_latent_dim),
            morsa_mlp_depth=int(args.morsa_mlp_depth),
            morsa_dropout=float(args.morsa_dropout),
        )
    if model_type == "mean":
        return MeanBackbone(
            input_dim=feature_dim,
            num_outputs=num_outputs,
            device=device,
            head_type=head_type,
            rank_k=rank_k,
            output_mu=output_mu,
            output_basis=output_basis,
        )
    if model_type == "he2rna":
        top_k = tuple(
            int(value)
            for value in str(args.he2rna_ks).replace(" ", "").split(",")
            if value.strip()
        )
        return HE2RNABackbone(
            input_dim=feature_dim,
            num_outputs=num_outputs,
            ks=top_k,
            dropout=float(args.he2rna_dropout),
            device=device,
            head_type=head_type,
            rank_k=rank_k,
            output_mu=output_mu,
            output_basis=output_basis,
        )
    raise ValueError(f"Unsupported model_type={model_type!r}.")


def _make_dataloader(
    dataset: SuperTileRNADataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "collate_fn": custom_collate_fn,
        "generator": generator,
        "worker_init_fn": seed_worker if int(num_workers) > 0 and shuffle else None,
    }
    if int(num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def _rna_matrix(frame: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    columns = [column for column in frame.columns if column.startswith("rna_")]
    if not columns:
        raise ValueError("No RNA columns with prefix 'rna_' were found.")
    return columns, frame[columns].to_numpy(dtype=np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixed_basis_digest(directory: str | None, num_folds: int) -> str | None:
    if directory is None:
        return None
    root = Path(directory)
    digest = hashlib.sha256()
    for fold in range(num_folds):
        path = root / f"basis_fold_{fold}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing canonical basis cache: {path}")
        digest.update(f"fold={fold}:".encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()[:16]


def _prepare_output_structure(
    *,
    head_type: str,
    y_train: np.ndarray,
    rank_k: int,
    basis_seed: int,
    fold: int,
    experiment_dir: Path,
    genes: list[str],
    train_patient_hash: str,
    fixed_basis_dir: Path | None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    effective_basis_seed = (
        int(basis_seed) + int(fold) if head_type in {"spex", "covnull_spex"} else -1
    )
    cache_metadata = {
        "fold": int(fold),
        "basis_seed": effective_basis_seed,
        "genes": np.asarray(genes, dtype=str),
        "gene_order_sha256": hashlib.sha256("\n".join(genes).encode("utf-8")).hexdigest(),
        "train_patient_hash": str(train_patient_hash),
    }
    if head_type == "linear":
        return None, None, {"basis_type": None, "rank_k": None}
    if head_type == "learned_rank":
        mu = np.mean(y_train, axis=0).astype(np.float32)
        diagnostics = {
            "basis_type": "learned",
            "rank_k": int(rank_k),
            "train_samples": int(y_train.shape[0]),
            "num_genes": int(y_train.shape[1]),
        }
        np.savez_compressed(
            experiment_dir / f"basis_fold_{fold}.npz",
            mu=mu,
            **diagnostics,
            **cache_metadata,
        )
        return mu, None, diagnostics

    if fixed_basis_dir is not None:
        if head_type != "spex":
            raise ValueError("--fixed_basis_dir is only valid for the spex head.")
        source_path = fixed_basis_dir / f"basis_fold_{fold}.npz"
        with np.load(source_path, allow_pickle=False) as payload:
            required = {"mu", "U_k", "genes", "gene_order_sha256", "train_patient_hash"}
            missing = sorted(required - set(payload.files))
            if missing:
                raise KeyError(f"Canonical basis cache {source_path} lacks fields: {missing}")
            mu = np.asarray(payload["mu"], dtype=np.float32)
            U_k = np.asarray(payload["U_k"], dtype=np.float32)
            source_genes = [str(value) for value in payload["genes"].tolist()]
            source_gene_hash = str(payload["gene_order_sha256"].item())
            source_train_hash = str(payload["train_patient_hash"].item())
            source_basis_seed = (
                int(payload["basis_seed"].item()) if "basis_seed" in payload else effective_basis_seed
            )
        expected_gene_hash = cache_metadata["gene_order_sha256"]
        if source_genes != genes or source_gene_hash != expected_gene_hash:
            raise ValueError(f"Canonical basis gene order does not match the current fold: {source_path}")
        if source_train_hash != train_patient_hash:
            raise ValueError(
                f"Canonical basis training-patient hash does not match the current fold: {source_path}"
            )
        if mu.shape != (y_train.shape[1],) or U_k.shape != (y_train.shape[1], int(rank_k)):
            raise ValueError(
                f"Canonical basis shape mismatch in {source_path}: mu={mu.shape}, U_k={U_k.shape}."
            )
        if not np.isfinite(mu).all() or not np.isfinite(U_k).all():
            raise ValueError(f"Canonical basis contains non-finite values: {source_path}")
        orthogonality_error = float(
            np.max(np.abs(U_k.astype(np.float64).T @ U_k.astype(np.float64) - np.eye(rank_k)))
        )
        diagnostics = {
            "basis_type": "pca",
            "basis_source": "canonical_cache",
            "rank_k": int(rank_k),
            "basis_seed": source_basis_seed,
            "source_basis_file": f"{fixed_basis_dir.name}/{source_path.name}",
            "source_basis_sha256": _sha256_file(source_path),
            "orthonormality_max_abs_error": orthogonality_error,
        }
        cache_metadata["basis_seed"] = source_basis_seed
        save_basis(
            str(experiment_dir / f"basis_fold_{fold}.npz"),
            BasisResult(mu=mu, U_k=U_k, diagnostics=diagnostics),
            **cache_metadata,
        )
        return mu, U_k, diagnostics

    result: BasisResult
    if head_type == "spex":
        result = fit_pca_basis(y_train, rank_k, seed=effective_basis_seed)
    elif head_type == "covnull_spex":
        result = fit_covnull_basis(y_train, rank_k, seed=effective_basis_seed)
    else:
        raise ValueError(f"Unsupported head_type={head_type!r}.")
    save_basis(str(experiment_dir / f"basis_fold_{fold}.npz"), result, **cache_metadata)
    return result.mu, result.U_k, dict(result.diagnostics)


def _mean_gene_pcc(real: np.ndarray, preds: np.ndarray) -> float:
    y = np.asarray(real, dtype=np.float64)
    p = np.asarray(preds, dtype=np.float64)
    y_centered = y - np.mean(y, axis=0, keepdims=True)
    p_centered = p - np.mean(p, axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(y_centered**2, axis=0) * np.sum(p_centered**2, axis=0))
    correlations = np.divide(
        np.sum(y_centered * p_centered, axis=0),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(np.mean(np.nan_to_num(correlations, nan=0.0)))


def _patient_hash(values) -> str:
    normalized = "\n".join(sorted(str(value) for value in np.asarray(values).tolist()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _run_payload(
    args,
    spec: ExperimentSpec,
    cancer: str,
    fixed_basis_digest: str | None,
) -> dict:
    return {
        "experiment": spec.name,
        "cancer": cancer,
        "cohort": args.cohort,
        "model_type": spec.model_type,
        "encoder": spec.encoder,
        "head_type": spec.head_type,
        "basis_type": spec.basis_type,
        "rank_k": spec.rank_k,
        "training_seed": spec.training_seed,
        "split_seed": spec.split_seed,
        "basis_seed": spec.basis_seed if fixed_basis_digest is None else None,
        "requested_basis_seed": spec.basis_seed,
        "untrained_reference_base_seed": DEFAULT_TRAINING_SEED + 1000,
        "fixed_basis_source_experiment": (
            Path(args.fixed_basis_dir).name if args.fixed_basis_dir is not None else None
        ),
        "fixed_basis_digest": fixed_basis_digest,
        "folds": int(args.folds),
        "max_folds": int(args.max_folds),
        "morsa_r": int(args.morsa_r),
        "morsa_eps": float(args.morsa_eps),
        "morsa_latent_dim": int(args.morsa_latent_dim),
        "morsa_mlp_depth": int(args.morsa_mlp_depth),
        "morsa_dropout": float(args.morsa_dropout),
        "depth": int(args.depth),
        "num_heads": int(args.num_heads),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "num_epochs": int(args.num_epochs),
        "patience": int(args.patience),
        "save_on": args.save_on,
        "stop_on": args.stop_on,
        "he2rna_ks": str(args.he2rna_ks),
        "he2rna_dropout": float(args.he2rna_dropout),
    }


def _config_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _optimizer_for(model: nn.Module, args) -> torch.optim.Optimizer:
    if args.model_type in {"vis", "he2rna"}:
        return torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    return torch.optim.AdamW(model.parameters(), lr=float(args.lr), amsgrad=False, weight_decay=0.0)


def _build_untrained_reference(
    args,
    *,
    num_outputs: int,
    feature_dim: int,
    device: torch.device,
    output_mu,
    output_basis,
    seed: int,
) -> nn.Module:
    random_mu = output_mu
    random_basis = output_basis
    if args.head_type in {"spex", "covnull_spex"}:
        random_mu, random_basis = random_reference_basis(output_mu, args.rank_k, seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = _build_model(
            args,
            model_type=args.model_type,
            num_outputs=num_outputs,
            feature_dim=feature_dim,
            device=device,
            output_mu=random_mu,
            output_basis=random_basis,
        )
    return model.to(device)


def run_experiment(args) -> Path:
    spec = ExperimentSpec(
        model_type=str(args.model_type),
        head_type=str(args.head_type),
        rank_k=int(args.rank_k),
        training_seed=int(args.seed),
        split_seed=int(args.split_seed),
        basis_seed=int(args.basis_seed),
    )
    expected_folds = int(args.max_folds) if int(args.max_folds) > 0 else int(args.folds)
    if args.fixed_basis_dir is not None and args.head_type != "spex":
        raise ValueError("--fixed_basis_dir is only valid with --head_type spex.")
    fixed_basis_digest = _fixed_basis_digest(args.fixed_basis_dir, expected_folds)
    cancer = str(args.cancer or Path(args.ref_file).resolve().parent.name)
    payload = _run_payload(args, spec, cancer, fixed_basis_digest)
    payload["config_hash"] = _config_hash(payload)
    experiment_dir = Path(args.save_dir) / str(args.cohort) / spec.name

    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        raise FileExistsError(
            f"Experiment directory already contains files: {experiment_dir}. "
            "Use a different --save_dir or remove the directory before starting a new run."
        )

    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_token = str(getattr(args, "_run_token", uuid.uuid4().hex))
    try:
        with (experiment_dir / "RUNNING").open("x", encoding="utf-8") as handle:
            handle.write(run_token + "\n")
    except FileExistsError as error:
        raise FileExistsError(f"Experiment is already running: {experiment_dir}") from error
    (experiment_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    device = _resolve_device(args.device)
    print(f"[run] experiment={spec.name} cancer={cancer} device={device}", flush=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    frame = pd.read_csv(args.ref_file)
    if int(args.filter_no_features) == 1:
        frame = filter_no_features(frame, args.feature_path, "cluster_features")
    if frame.empty:
        raise ValueError("No samples remain after filtering unavailable features.")

    train_indices, val_indices, test_indices = patient_kfold(
        frame,
        n_splits=int(args.folds),
        random_state=int(args.split_seed),
    )
    fold_indices = list(zip(train_indices, val_indices, test_indices))
    if int(args.max_folds) > 0:
        fold_indices = fold_indices[: int(args.max_folds)]

    test_results: dict[str, object] = {}
    validation_results: dict[str, object] = {}
    validation_metric_rows = []
    basis_diagnostics = []
    split_hashes = []
    fold_train_seconds = []
    fold_peak_gpu_memory_mb = []
    fold_inference_seconds = []
    fold_inference_time_per_wsi = []
    model_param_count_excluding_head = None
    model_param_count_total = None
    model_param_count_trainable = None

    for fold, (train_index, val_index, test_index) in enumerate(fold_indices):
        fold_seed = int(args.seed) + fold
        _seed_everything(fold_seed)
        train_frame = frame.iloc[train_index]
        val_frame = frame.iloc[val_index]
        test_frame = frame.iloc[test_index]

        train_patients = np.unique(train_frame.patient_id)
        val_patients = np.unique(val_frame.patient_id)
        test_patients = np.unique(test_frame.patient_id)
        np.save(experiment_dir / f"train_{fold}.npy", train_patients)
        np.save(experiment_dir / f"val_{fold}.npy", val_patients)
        np.save(experiment_dir / f"test_{fold}.npy", test_patients)
        split_hashes.append(
            {
                "fold": fold,
                "train": _patient_hash(train_patients),
                "validation": _patient_hash(val_patients),
                "test": _patient_hash(test_patients),
            }
        )

        train_dataset = SuperTileRNADataset(train_frame, args.feature_path)
        val_dataset = SuperTileRNADataset(val_frame, args.feature_path)
        test_dataset = SuperTileRNADataset(test_frame, args.feature_path)
        num_outputs = int(train_dataset.num_genes)
        feature_dim = int(train_dataset.feature_dim)
        _, y_train = _rna_matrix(train_frame)
        genes = [column[4:] for column in train_frame.columns if column.startswith("rna_")]
        output_mu, output_basis, diagnostics = _prepare_output_structure(
            head_type=args.head_type,
            y_train=y_train,
            rank_k=int(args.rank_k),
            basis_seed=int(args.basis_seed),
            fold=fold,
            experiment_dir=experiment_dir,
            genes=genes,
            train_patient_hash=_patient_hash(train_patients),
            fixed_basis_dir=Path(args.fixed_basis_dir) if args.fixed_basis_dir is not None else None,
        )
        diagnostics = {"fold": fold, **diagnostics}
        basis_diagnostics.append(diagnostics)

        if args.model_type == "mean":
            _precompute_mean_inputs(train_dataset)
            _precompute_mean_inputs(val_dataset)
            _precompute_mean_inputs(test_dataset)

        train_loader = _make_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers_train,
            shuffle=True,
            seed=fold_seed,
        )
        val_loader = _make_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers_val,
            shuffle=False,
            seed=fold_seed,
        )
        test_loader = _make_dataloader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers_test,
            shuffle=False,
            seed=fold_seed,
        )

        model = _build_model(
            args,
            model_type=args.model_type,
            num_outputs=num_outputs,
            feature_dim=feature_dim,
            device=device,
            output_mu=output_mu,
            output_basis=output_basis,
        ).to(device)
        _reset_cuda_peak_memory(device)
        _print_model_summary(model, model_type=args.model_type, decoder=args.head_type)
        if model_param_count_excluding_head is None:
            model_param_count_excluding_head = _backbone_param_count(model)
            model_param_count_total, model_param_count_trainable = _param_counts(model)

        optimizer = _optimizer_for(model, args)
        train_start = time.perf_counter()
        model = train(
            model,
            {"train": train_loader, "val": val_loader},
            optimizer,
            num_epochs=int(args.num_epochs),
            run=None,
            split=fold,
            save_on=args.save_on,
            stop_on=args.stop_on,
            delta=0.5,
            save_dir=str(experiment_dir),
            patience=int(args.patience),
        )
        fold_train_seconds.append(float(time.perf_counter() - train_start))

        val_preds, val_real, val_wsis, val_projects = evaluate(
            model, val_loader, run=None, suff=f"_{fold}_validation"
        )
        validation_results[f"split_{fold}"] = {
            "real": val_real,
            "preds": val_preds,
            "wsi_file_name": val_wsis,
            "tcga_project": val_projects,
        }
        validation_metric_rows.append(
            {
                "fold": fold,
                "all_gene_pcc": _mean_gene_pcc(val_real, val_preds),
                "mse": float(np.mean((val_real - val_preds) ** 2)),
                "n_samples": int(val_real.shape[0]),
            }
        )

        _sync_if_cuda(device)
        inference_start = time.perf_counter()
        preds, real, wsis, projects = evaluate(model, test_loader, run=None, suff=f"_{fold}")
        _sync_if_cuda(device)
        inference_seconds = float(time.perf_counter() - inference_start)
        fold_inference_seconds.append(inference_seconds)
        fold_inference_time_per_wsi.append(inference_seconds / float(max(len(wsis), 1)))
        peak_memory = _cuda_peak_memory_mb(device)
        if peak_memory is not None:
            fold_peak_gpu_memory_mb.append(peak_memory)

        random_model = _build_untrained_reference(
            args,
            num_outputs=num_outputs,
            feature_dim=feature_dim,
            device=device,
            output_mu=output_mu,
            output_basis=output_basis,
            seed=DEFAULT_TRAINING_SEED + 1000 + fold,
        )
        random_preds, _, _, _ = evaluate(random_model, test_loader, run=None, suff=f"_{fold}_random")
        test_results[f"split_{fold}"] = {
            "real": real,
            "preds": preds,
            "random": random_preds,
            "wsi_file_name": wsis,
            "tcga_project": projects,
        }

    genes = [column[4:] for column in frame.columns if column.startswith("rna_")]
    test_results["genes"] = genes
    validation_results["genes"] = genes
    with (experiment_dir / "test_results.pkl").open("wb") as handle:
        pickle.dump(test_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (experiment_dir / "validation_results.pkl").open("wb") as handle:
        pickle.dump(validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(validation_metric_rows).to_csv(experiment_dir / "validation_metrics.csv", index=False)

    metadata = {
        **payload,
        "num_folds": len(fold_indices),
        "split_hashes": split_hashes,
        "basis_diagnostics": basis_diagnostics,
        "parameter_count": int(model_param_count_excluding_head or 0),
        "parameter_count_excluding_output_head": int(model_param_count_excluding_head or 0),
        "parameter_count_total": int(model_param_count_total or 0),
        "parameter_count_including_output_head": int(model_param_count_trainable or 0),
        "fold_train_seconds": fold_train_seconds,
        "avg_fold_train_seconds": float(np.mean(fold_train_seconds)),
        "sum_fold_train_seconds": float(np.sum(fold_train_seconds)),
        "fold_peak_gpu_memory_mb": fold_peak_gpu_memory_mb,
        "peak_gpu_memory_mb": float(np.max(fold_peak_gpu_memory_mb)) if fold_peak_gpu_memory_mb else None,
        "fold_inference_seconds": fold_inference_seconds,
        "fold_inference_time_per_wsi_seconds": fold_inference_time_per_wsi,
        "avg_inference_time_per_wsi_seconds": float(np.mean(fold_inference_time_per_wsi)),
        "validation_all_gene_pcc_mean": float(
            np.mean([row["all_gene_pcc"] for row in validation_metric_rows])
        ),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    (experiment_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (experiment_dir / "RUNNING").unlink(missing_ok=True)
    (experiment_dir / "COMPLETE").write_text(payload["config_hash"] + "\n", encoding="utf-8")
    print(f"[complete] {experiment_dir}", flush=True)
    return experiment_dir


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one MORSA experiment with patient-level cross-validation.")
    parser.add_argument("--ref_file", required=True)
    parser.add_argument("--feature_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--cohort", default="TCGA")
    parser.add_argument("--cancer", default=None)
    parser.add_argument("--filter_no_features", type=int, choices=[0, 1], default=1)
    parser.add_argument("--model_type", choices=MODEL_TYPES, required=True)
    parser.add_argument("--head_type", choices=HEAD_TYPES, required=True)
    parser.add_argument("--rank_k", type=int, default=DEFAULT_RANK_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--basis_seed", type=int, default=DEFAULT_BASIS_SEED)
    parser.add_argument(
        "--fixed_basis_dir",
        default=None,
        help="Directory containing canonical basis_fold_<fold>.npz files shared across matched SPEX runs.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_folds", type=int, default=0, help="0 runs all folds; use only for development tests.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--save_on", choices=["loss", "corr", "loss+corr"], default="loss+corr")
    parser.add_argument("--stop_on", choices=["loss", "corr", "loss+corr"], default="loss+corr")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_workers_train", type=int, default=0)
    parser.add_argument("--num_workers_val", type=int, default=0)
    parser.add_argument("--num_workers_test", type=int, default=0)
    parser.add_argument("--morsa_r", type=int, default=96)
    parser.add_argument("--morsa_eps", type=float, default=1e-3)
    parser.add_argument("--morsa_latent_dim", type=int, default=1024)
    parser.add_argument("--morsa_mlp_depth", type=int, default=1)
    parser.add_argument("--morsa_dropout", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--he2rna_ks", default="1,2,5,10,20,50,100")
    parser.add_argument("--he2rna_dropout", type=float, default=0.5)
    return parser


def parse_train_args(argv=None):
    return build_train_arg_parser().parse_args(argv)


def main(argv=None) -> None:
    args = parse_train_args(argv)
    run_token = uuid.uuid4().hex
    args._run_token = run_token
    experiment_dir = Path(args.save_dir) / str(args.cohort) / ExperimentSpec(
        args.model_type,
        args.head_type,
        args.rank_k,
        args.seed,
        args.split_seed,
        args.basis_seed,
    ).name
    running_marker = experiment_dir / "RUNNING"
    try:
        run_experiment(args)
    except Exception as error:
        marker_owner = running_marker.read_text(encoding="utf-8").strip() if running_marker.is_file() else None
        if marker_owner == run_token:
            running_marker.unlink()
            (experiment_dir / "FAILED").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
