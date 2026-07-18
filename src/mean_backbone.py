"""
Shallow mean-pool control:
x(B,T,D) -> mean over T -> prediction head.

This intentionally removes the intermediate stem MLP so the backbone is a
strict pooled-feature baseline.
"""
import torch
import torch.nn as nn

from heads import make_output_head


class MeanBackbone(nn.Module):
    """x(B,T,D) -> mean over T -> head."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_outputs: int,
        device: str | torch.device = "cuda:0",
        head_type: str = "linear",
        rank_k: int = 16,
        output_mu=None,
        output_basis=None,
    ):
        super().__init__()
        self.device = device
        self.input_dim = int(input_dim)
        self.num_outputs = int(num_outputs)
        self.head_type = str(head_type).lower()

        latent = self.input_dim
        self.linear_head = make_output_head(
            hidden_dim=latent,
            num_outputs=self.num_outputs,
            head_type=self.head_type,
            rank_k=rank_k,
            mu=output_mu,
            U_k=output_basis,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            z = x.mean(dim=1)
        elif x.dim() == 2:
            z = x
        else:
            raise ValueError(f"Expected x shape (B,T,D) or (B,D), got {tuple(x.shape)}")
        return self.linear_head(z)
