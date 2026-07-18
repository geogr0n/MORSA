from __future__ import annotations

import torch
import torch.nn as nn

from experiment_config import HEAD_TYPES


def _tensor_or_default(value, shape: tuple[int, ...]) -> torch.Tensor:
    if value is None:
        return torch.zeros(shape, dtype=torch.float32)
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float()
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"Expected tensor shape={shape}, got {tuple(tensor.shape)}.")
    return tensor


class SPEXHead(nn.Module):
    """Predict fixed-basis coordinates and reconstruct the full gene vector."""

    def __init__(
        self,
        hidden_dim: int,
        num_outputs: int,
        rank_k: int,
        mu=None,
        U_k=None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        self.rank_k = int(rank_k)
        self.z_proj = nn.Linear(self.hidden_dim, self.rank_k)
        self.register_buffer("mu", _tensor_or_default(mu, (self.num_outputs,)))
        self.register_buffer("U_k", _tensor_or_default(U_k, (self.num_outputs, self.rank_k)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mu + self.z_proj(h) @ self.U_k.T


class LearnedRankHead(nn.Module):
    """Learn both rank-constrained coordinates and their gene-space basis."""

    def __init__(self, hidden_dim: int, num_outputs: int, rank_k: int, mu=None) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_outputs = int(num_outputs)
        self.rank_k = int(rank_k)
        self.z_proj = nn.Linear(self.hidden_dim, self.rank_k)
        self.register_buffer("mu", _tensor_or_default(mu, (self.num_outputs,)))
        q, _ = torch.linalg.qr(torch.randn(self.num_outputs, self.rank_k), mode="reduced")
        self.V = nn.Parameter(q.contiguous())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mu + self.z_proj(h) @ self.V.T


def make_layernorm_linear_head(hidden_dim: int, num_outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(int(hidden_dim)),
        nn.Linear(int(hidden_dim), int(num_outputs)),
    )


def make_output_head(
    *,
    hidden_dim: int,
    num_outputs: int,
    head_type: str,
    rank_k: int,
    mu=None,
    U_k=None,
) -> nn.Module:
    head = str(head_type).lower()
    if head not in HEAD_TYPES:
        raise ValueError(f"Unsupported head_type={head!r}; choose from {HEAD_TYPES}.")
    if head == "linear":
        return make_layernorm_linear_head(hidden_dim, num_outputs)
    if head == "learned_rank":
        return LearnedRankHead(hidden_dim, num_outputs, rank_k, mu=mu)
    return SPEXHead(hidden_dim, num_outputs, rank_k, mu=mu, U_k=U_k)
