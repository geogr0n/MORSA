import argparse
import json
import os
import pickle
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(EVAL_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from evaluate_model import evaluate_experiment
from experiment_config import DEFAULT_RANK_K
from main import _backbone_param_count, _build_model, _param_counts
from dataset import SuperTileRNADataset
from data_utils import custom_collate_fn, filter_no_features


def _require_file(path: str, label: str) -> str:
    if not path:
        raise ValueError(f"{label} is required")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return os.path.abspath(path)


def _require_dir(path: str, label: str) -> str:
    if not path:
        raise ValueError(f"{label} is required")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return os.path.abspath(path)


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _discover_experiments(tcga_model_dir: str) -> list[str]:
    experiments = []
    for name in sorted(os.listdir(tcga_model_dir)):
        exp_dir = os.path.join(tcga_model_dir, name)
        if name == "results" or not os.path.isdir(exp_dir):
            continue
        if os.path.exists(_checkpoint_path(exp_dir, 0)):
            experiments.append(name)
    return experiments


def _checkpoint_path(exp_dir: str, fold: int) -> str:
    if int(fold) == 0:
        return os.path.join(exp_dir, "model_best.pt")
    return os.path.join(exp_dir, f"model_best_{int(fold)}.pt")


def _load_metadata(exp_dir: str, experiment: str) -> dict:
    path = os.path.join(exp_dir, "metadata.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metadata.json for experiment: {experiment} ({path})")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_gene_names_from_columns(columns: list[str]) -> list[str]:
    return [str(c)[4:] if str(c).startswith("rna_") else str(c) for c in columns]


def _load_training_genes(exp_dir: str) -> list[str]:
    pkl_path = os.path.join(exp_dir, "test_results.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Training gene order is required; missing {pkl_path}")
    with open(pkl_path, "rb") as f:
        test_res = pickle.load(f)
    genes = [str(g) for g in test_res.get("genes", [])]
    if not genes:
        raise ValueError(f"No genes found in training test_results.pkl: {pkl_path}")
    return _clean_gene_names_from_columns(genes)


def _align_external_ref(ref_file: str, training_genes: list[str]) -> tuple[pd.DataFrame, list[int], list[str], list[str]]:
    df = pd.read_csv(ref_file)
    required_cols = {"wsi_file_name", "tcga_project"}
    missing_required = sorted(required_cols - set(df.columns))
    if missing_required:
        raise ValueError(f"External ref_file is missing required columns: {missing_required}")

    rna_cols = [f"rna_{g}" for g in training_genes]
    available_indices = [i for i, c in enumerate(rna_cols) if c in df.columns]
    available_genes = [training_genes[i] for i in available_indices]
    missing_genes = [c for c in rna_cols if c not in df.columns]
    if not available_indices:
        raise ValueError("External ref_file has no RNA columns overlapping the TCGA training genes.")
    if missing_genes:
        preview = ", ".join(missing_genes[:10])
        more = "" if len(missing_genes) <= 10 else f" ... (+{len(missing_genes) - 10} more)"
        print(
            f"[cross-cohort] external ref_file is missing {len(missing_genes)} training genes; "
            f"metrics will use {len(available_genes)} overlapping genes. Missing preview: {preview}{more}",
            flush=True,
        )

    metadata_cols = [c for c in ("patient_id", "wsi_file_name", "tcga_project") if c in df.columns]
    rna_data = {
        c: df[c].to_numpy(dtype=np.float32) if c in df.columns else np.full(len(df), np.nan, dtype=np.float32)
        for c in rna_cols
    }
    rna_df = pd.DataFrame(rna_data, index=df.index)
    aligned = pd.concat([df.loc[:, metadata_cols].copy(), rna_df], axis=1)
    return aligned, available_indices, available_genes, missing_genes


def _make_model_args(args: argparse.Namespace, metadata: dict) -> SimpleNamespace:
    model_type = str(metadata.get("model_type") or "").lower()
    head_type = str(metadata.get("head_type") or "").lower()
    if model_type not in {"vis", "he2rna", "morsa_enc", "mean"}:
        raise ValueError(f"Unsupported or missing model_type in metadata: {model_type!r}")
    if head_type not in {"linear", "spex", "learned_rank", "covnull_spex"}:
        raise ValueError(f"Unsupported or missing head_type in metadata: {head_type!r}")
    return SimpleNamespace(
        model_type=model_type,
        head_type=head_type,
        depth=int(args.depth),
        num_heads=int(args.num_heads),
        morsa_r=int(args.morsa_r),
        morsa_eps=float(args.morsa_eps),
        morsa_latent_dim=int(args.morsa_latent_dim),
        morsa_mlp_depth=int(args.morsa_mlp_depth),
        morsa_dropout=float(args.morsa_dropout),
        he2rna_ks=args.he2rna_ks,
        he2rna_dropout=float(args.he2rna_dropout),
        rank_k=int(metadata.get("rank_k", DEFAULT_RANK_K)),
    )


def _dataloader(dataset: SuperTileRNADataset, args: argparse.Namespace, device: torch.device) -> DataLoader:
    kwargs = dict(
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=custom_collate_fn,
    )
    if int(args.num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


def _load_model_from_checkpoint(
    *,
    model_args: SimpleNamespace,
    checkpoint_path: str,
    num_outputs: int,
    feature_dim: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    model = _build_model(
        model_args,
        model_type=model_args.model_type,
        num_outputs=num_outputs,
        feature_dim=feature_dim,
        device=device,
        output_mu=None,
        output_basis=None,
    )
    model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, state


def _init_untrained_spex_reference(random_model: torch.nn.Module, trained_state: dict, seed: int) -> None:
    head = getattr(random_model, "linear_head", None)
    if head is None or not hasattr(head, "mu") or not hasattr(head, "U_k"):
        return
    trained_mu = trained_state.get("linear_head.mu")
    trained_basis = trained_state.get("linear_head.U_k")
    if trained_mu is None or trained_basis is None:
        return

    mu_np = trained_mu.detach().cpu().numpy()
    num_outputs, k = int(trained_basis.shape[0]), int(trained_basis.shape[1])
    rng = np.random.default_rng(int(seed))
    random_mu = (rng.standard_normal(num_outputs) * (np.std(mu_np) + 1e-8) + np.mean(mu_np)).astype(np.float32)
    q, _ = np.linalg.qr(rng.standard_normal((num_outputs, k)))
    random_basis = q.astype(np.float32)

    with torch.no_grad():
        head.mu.copy_(torch.from_numpy(random_mu).to(device=head.mu.device, dtype=head.mu.dtype))
        head.U_k.copy_(torch.from_numpy(random_basis).to(device=head.U_k.device, dtype=head.U_k.dtype))


def _make_random_model(
    *,
    model_args: SimpleNamespace,
    trained_state: dict,
    num_outputs: int,
    feature_dim: int,
    device: torch.device,
    seed: int,
) -> torch.nn.Module:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    random.seed(int(seed))
    model = _build_model(
        model_args,
        model_type=model_args.model_type,
        num_outputs=num_outputs,
        feature_dim=feature_dim,
        device=device,
        output_mu=None,
        output_basis=None,
    )
    model.to(device)
    if model_args.head_type in {"spex", "covnull_spex"}:
        _init_untrained_spex_reference(model, trained_state, seed)
    model.eval()
    return model


def _predict_external(
    model: torch.nn.Module,
    loader: DataLoader,
    available_indices: list[int],
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device = model.device
    model.eval()
    all_preds = []
    all_labels = []
    all_wsi_names = []
    all_projects = []
    indexer = np.asarray(available_indices, dtype=np.int64)

    with torch.no_grad():
        for x, y, wsi_names, projects in loader:
            x = x.float().to(device)
            outputs = model(x)
            all_preds.append(outputs.detach().cpu().numpy()[:, indexer])
            all_labels.append(y.numpy()[:, indexer])
            all_wsi_names.extend(wsi_names)
            all_projects.extend(projects)

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    print(f"[cross-cohort] {label}: predicted {preds.shape[0]} samples x {preds.shape[1]} genes", flush=True)
    return preds, labels, np.asarray(all_wsi_names), np.asarray(all_projects)


def _evaluate_one_fold(
    *,
    model: torch.nn.Module,
    random_model: torch.nn.Module,
    loader: DataLoader,
    fold: int,
    available_indices: list[int],
) -> dict[str, object]:
    preds, real, wsis, projs = _predict_external(model, loader, available_indices, label=f"fold {fold}")
    random_preds, _, _, _ = _predict_external(random_model, loader, available_indices, label=f"fold {fold} random")
    return {
        "real": real,
        "preds": preds,
        "random": random_preds,
        "wsi_file_name": wsis,
        "tcga_project": projs,
    }


def _write_metric_outputs(output_dir: str, experiment: str, result, extra_summary: dict) -> dict:
    combine_res, sig_res, summary = result
    summary.update(extra_summary)

    save_path = os.path.join(output_dir, "results", experiment)
    os.makedirs(save_path, exist_ok=True)
    combine_res.to_csv(os.path.join(save_path, "all_genes.csv"))
    sig_res.to_csv(os.path.join(save_path, "sig_genes.csv"))
    pd.DataFrame([summary]).to_csv(os.path.join(save_path, "summary_row.csv"), index=False)
    return summary


def evaluate_cross_cohort_experiment(
    *,
    experiment: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    source_exp_dir = os.path.join(args.tcga_model_dir, experiment)
    if not os.path.isdir(source_exp_dir):
        raise FileNotFoundError(f"Experiment directory not found: {source_exp_dir}")

    metadata = _load_metadata(source_exp_dir, experiment)
    model_args = _make_model_args(args, metadata)
    training_genes = _load_training_genes(source_exp_dir)

    external_df, available_indices, available_genes, missing_genes = _align_external_ref(args.external_ref_file, training_genes)
    if int(args.filter_no_features):
        external_df = filter_no_features(external_df, args.external_feature_path, "cluster_features")
    if external_df.empty:
        raise ValueError(f"No external samples left after filtering for experiment={experiment}")

    dataset = SuperTileRNADataset(external_df, args.external_feature_path, preload=bool(int(args.preload)))
    loader = _dataloader(dataset, args, device)

    fold_results = []
    param_count_excluding_head = None
    param_count_including_head = None
    for fold in range(int(args.folds)):
        checkpoint_path = _checkpoint_path(source_exp_dir, fold)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint for experiment={experiment}, fold={fold}: {checkpoint_path}")
        print(f"[{experiment}] external fold {fold}: {checkpoint_path}", flush=True)
        model, state = _load_model_from_checkpoint(
            model_args=model_args,
            checkpoint_path=checkpoint_path,
            num_outputs=dataset.num_genes,
            feature_dim=dataset.feature_dim,
            device=device,
        )
        if param_count_excluding_head is None:
            param_count_excluding_head = int(_backbone_param_count(model))
            _, total_trainable = _param_counts(model)
            param_count_including_head = int(total_trainable)
        random_model = _make_random_model(
            model_args=model_args,
            trained_state=state,
            num_outputs=dataset.num_genes,
            feature_dim=dataset.feature_dim,
            device=device,
            seed=int(args.seed) + 1000 + fold,
        )
        fold_results.append(
            _evaluate_one_fold(
                model=model,
                random_model=random_model,
                loader=loader,
                fold=fold,
                available_indices=available_indices,
            )
        )

    test_results = {"genes": available_genes}
    first = fold_results[0]
    test_results["split_0"] = {
        "real": first["real"],
        "preds": np.mean([x["preds"] for x in fold_results], axis=0),
        "random": np.mean([x["random"] for x in fold_results], axis=0),
        "wsi_file_name": first["wsi_file_name"],
        "tcga_project": first["tcga_project"],
    }
    metric_folds = 1

    out_exp_dir = os.path.join(args.output_dir, experiment)
    os.makedirs(out_exp_dir, exist_ok=True)
    with open(os.path.join(out_exp_dir, "test_results.pkl"), "wb") as f:
        pickle.dump(test_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    out_metadata = {
        "experiment": experiment,
        "source_experiment_dir": source_exp_dir,
        "external_ref_file": args.external_ref_file,
        "external_feature_path": args.external_feature_path,
        "external_cohort": args.external_cohort,
        "prediction_mode": "ensemble",
        "source_folds": int(args.folds),
        "metric_folds": int(metric_folds),
        "num_external_samples": int(len(dataset)),
        "num_training_genes": int(len(training_genes)),
        "num_evaluated_genes": int(len(available_genes)),
        "num_missing_external_genes": int(len(missing_genes)),
        "missing_external_genes_preview": missing_genes[:20],
        "model_type": model_args.model_type,
        "head_type": model_args.head_type,
        "rank_k": int(model_args.rank_k),
        "parameter_count": int(param_count_excluding_head or 0),
        "parameter_count_excluding_output_head": int(param_count_excluding_head or 0),
        "parameter_count_including_output_head": int(param_count_including_head or 0),
    }
    with open(os.path.join(out_exp_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(out_metadata, f, indent=2, ensure_ascii=False)

    result = evaluate_experiment(experiment, args.output_dir, folds=metric_folds)
    if result is None:
        raise RuntimeError(f"Metric evaluation failed for experiment={experiment}")
    return _write_metric_outputs(
        args.output_dir,
        experiment,
        result,
        {
            "external_cohort": args.external_cohort,
            "prediction_mode": "ensemble",
            "source_folds": int(args.folds),
            "metric_folds": int(metric_folds),
            "num_external_samples": int(len(dataset)),
            "num_training_genes": int(len(training_genes)),
            "num_evaluated_genes": int(len(available_genes)),
            "num_missing_external_genes": int(len(missing_genes)),
            "source_experiment_dir": source_exp_dir,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TCGA-trained MORSA checkpoints on an external cohort and evaluate cross-cohort performance."
    )
    parser.add_argument("--tcga_model_dir", required=True, help="Directory containing TCGA-trained experiment subdirectories.")
    parser.add_argument("--external_ref_file", required=True, help="External cohort ref_file.csv.")
    parser.add_argument("--external_feature_path", required=True, help="External cohort feature directory.")
    parser.add_argument("--output_dir", required=True, help="Destination directory for external predictions and metrics.")
    parser.add_argument("--external_cohort", required=True, help="Name written to metadata and summary rows.")
    parser.add_argument("--experiments", default="", help="Comma-separated experiments; if empty, scan tcga_model_dir.")
    parser.add_argument("--folds", type=int, default=5, help="Number of TCGA fold checkpoints to load.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preload", type=int, default=1)
    parser.add_argument("--filter_no_features", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0 style device.")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=16)
    parser.add_argument("--morsa_r", type=int, default=96)
    parser.add_argument("--morsa_eps", type=float, default=1e-3)
    parser.add_argument("--morsa_latent_dim", type=int, default=1024)
    parser.add_argument("--morsa_mlp_depth", type=int, default=1)
    parser.add_argument("--morsa_dropout", type=float, default=0.1)
    parser.add_argument("--he2rna_ks", default="1,2,5,10,20,50,100")
    parser.add_argument("--he2rna_dropout", type=float, default=0.5)
    return parser


def _resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.tcga_model_dir = _require_dir(args.tcga_model_dir, "--tcga_model_dir")
    args.external_ref_file = _require_file(args.external_ref_file, "--external_ref_file")
    args.external_feature_path = _require_dir(args.external_feature_path, "--external_feature_path")
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"[cross-cohort] device={device}", flush=True)
    print(f"[cross-cohort] output_dir={args.output_dir}", flush=True)

    experiments = _parse_csv(args.experiments) or _discover_experiments(args.tcga_model_dir)
    if not experiments:
        raise ValueError(f"No experiments found under --tcga_model_dir: {args.tcga_model_dir}")
    print(f"[cross-cohort] experiments={','.join(experiments)}", flush=True)

    summary_rows = []
    for experiment in experiments:
        summary_rows.append(evaluate_cross_cohort_experiment(experiment=experiment, args=args, device=device))

    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, "summary_metrics.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[cross-cohort] summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
