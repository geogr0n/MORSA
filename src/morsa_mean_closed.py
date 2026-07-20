from __future__ import annotations

import torch
import torch.nn as nn


class ClosedMORSAMean(nn.Module):
    """Closed-form affine prediction in a fixed RNA basis."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_outputs: int,
        rank_k: int,
        weight,
        bias,
        output_mu,
        output_basis,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.input_dim = int(input_dim)
        self.num_outputs = int(num_outputs)
        self.rank_k = int(rank_k)

        self.register_buffer("weight", self._tensor(weight, (self.input_dim, self.rank_k)))
        self.register_buffer("bias", self._tensor(bias, (self.rank_k,)))
        self.register_buffer("output_mu", self._tensor(output_mu, (self.num_outputs,)))
        self.register_buffer(
            "output_basis",
            self._tensor(output_basis, (self.num_outputs, self.rank_k)),
        )
        self.to(self.device)

    @staticmethod
    def _tensor(value, expected_shape: tuple[int, ...]) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {tuple(tensor.shape)}.")
        return tensor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 3:
            features = inputs.mean(dim=1)
        elif inputs.ndim == 2:
            features = inputs
        else:
            raise ValueError(f"Expected inputs with shape (B,M,D) or (B,D), got {tuple(inputs.shape)}.")
        if features.shape[1] != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, got {features.shape[1]}.")
        coordinates = features @ self.weight + self.bias
        return self.output_mu + coordinates @ self.output_basis.T
