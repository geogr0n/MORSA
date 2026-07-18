from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


DEFAULT_RANK_K = 16
DEFAULT_TRAINING_SEED = 29
DEFAULT_SPLIT_SEED = 0
DEFAULT_BASIS_SEED = 29

HEAD_TYPES = ("linear", "spex", "learned_rank", "covnull_spex")
MODEL_TYPES = ("mean", "he2rna", "vis", "morsa_enc", "diag_spd")


_BASE_NAMES = {
    ("mean", "linear"): "mean",
    ("mean", "spex"): "morsa_mean",
    ("mean", "learned_rank"): "learned_rank_mean",
    ("mean", "covnull_spex"): "covnull_spex_mean",
    ("he2rna", "linear"): "he2rna",
    ("he2rna", "spex"): "morsa_he2rna",
    ("he2rna", "learned_rank"): "learned_rank_he2rna",
    ("he2rna", "covnull_spex"): "covnull_spex_he2rna",
    ("vis", "linear"): "vis",
    ("vis", "spex"): "morsa_vis",
    ("vis", "learned_rank"): "learned_rank_vis",
    ("vis", "covnull_spex"): "covnull_spex_vis",
    ("morsa_enc", "linear"): "morsa_enc",
    ("morsa_enc", "spex"): "morsa",
    ("morsa_enc", "learned_rank"): "learned_rank_spd",
    ("morsa_enc", "covnull_spex"): "covnull_spex_spd",
    ("diag_spd", "linear"): "diag_morsa_enc",
    ("diag_spd", "spex"): "diag_morsa",
    ("diag_spd", "learned_rank"): "learned_rank_diag_spd",
    ("diag_spd", "covnull_spex"): "covnull_spex_diag_spd",
}


@dataclass(frozen=True)
class ExperimentSpec:
    model_type: str
    head_type: str
    rank_k: int = DEFAULT_RANK_K
    training_seed: int = DEFAULT_TRAINING_SEED
    split_seed: int = DEFAULT_SPLIT_SEED
    basis_seed: int = DEFAULT_BASIS_SEED

    def __post_init__(self) -> None:
        if self.model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model_type={self.model_type!r}; choose from {MODEL_TYPES}.")
        if self.head_type not in HEAD_TYPES:
            raise ValueError(f"Unsupported head_type={self.head_type!r}; choose from {HEAD_TYPES}.")
        if int(self.rank_k) < 1:
            raise ValueError(f"rank_k must be positive, got {self.rank_k}.")
        if self.head_type == "linear" and int(self.rank_k) != DEFAULT_RANK_K:
            raise ValueError("rank_k applies only to rank-constrained heads; Linear uses the default value.")

    @property
    def name(self) -> str:
        name = _BASE_NAMES[(self.model_type, self.head_type)]
        if self.head_type != "linear" and int(self.rank_k) != DEFAULT_RANK_K:
            name += f"_k{int(self.rank_k)}"
        if int(self.training_seed) != DEFAULT_TRAINING_SEED:
            name += f"_seed{int(self.training_seed)}"
        return name

    @property
    def basis_type(self) -> str | None:
        return {
            "linear": None,
            "spex": "pca",
            "learned_rank": "learned",
            "covnull_spex": "covariance_null_pca",
        }[self.head_type]

    @property
    def encoder(self) -> str:
        return {
            "mean": "mean",
            "he2rna": "he2rna",
            "vis": "vis",
            "morsa_enc": "spd",
            "diag_spd": "diag_spd",
        }[self.model_type]

    @property
    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
