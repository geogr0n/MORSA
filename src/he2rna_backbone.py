# SPDX-License-Identifier: GPL-3.0-or-later
# This HE2RNA-style baseline path is adapted from gevaertlab/sequoia-pub/src/he2rna.py,
# which retains the Owkin HE2RNA GNU GPL v3-or-later notice.

import numpy as np
import torch
import torch.nn as nn
from heads import make_output_head


class HE2RNABackbone(nn.Module):
    """
    HE2RNA-style MLP/top-k aggregator (sequoia-aligned).
    conv stack input_dim -> 256 -> 256 -> num_outputs, top-k on (B, num_outputs, T), then LayerNorm.
    Input: x (B, T, D), where T is number of tiles/clusters and D is feature dim.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_outputs: int,
        ks=(10,),
        dropout: float = 0.5,
        device: str | torch.device = "cuda:0",
        head_type: str = "linear",
        rank_k: int = 16,
        output_mu=None,
        output_basis=None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_outputs = int(num_outputs)
        self.ks = tuple(max(1, int(k)) for k in ks)
        self.device = device
        self.head_type = str(head_type).lower()

        # sequoia HE2RNA: input_dim -> 256 -> 256 -> num_outputs
        self.tile_mlp = nn.Sequential(
            nn.Conv1d(self.input_dim, 256, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(256, 256, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(256, self.num_outputs, kernel_size=1, stride=1, bias=True),
        )
        backbone_out_dim = self.num_outputs
        self.hidden_dim = backbone_out_dim
        self.linear_head = nn.Identity()
        if self.head_type != "linear":
            self.linear_head = make_output_head(
                hidden_dim=backbone_out_dim,
                num_outputs=int(num_outputs),
                head_type=self.head_type,
                rank_k=rank_k,
                mu=output_mu,
                U_k=output_basis,
            )

    @staticmethod
    def _pool_topk(h: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
        # HE2RNA paper path: average the k largest tile predictions per gene.
        t = h.shape[2]
        k = max(1, min(int(k), int(t)))
        topk_vals, _ = torch.topk(h, k, dim=2, largest=True, sorted=True)
        denom = torch.sum(mask[:, :, :k], dim=2).clamp_min(1.0)
        return torch.sum(topk_vals * mask[:, :, :k], dim=2) / denom

    def _extract_he2rna_features(self, x_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask, _ = torch.max(x_feat, dim=1, keepdim=True)
        mask = (mask > 0).float()
        h = self.tile_mlp(x_feat) * mask
        return h, mask

    def _forward_with_k(self, tokens: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
        h = tokens.transpose(1, 2).contiguous()
        pooled = self._pool_topk(h, mask, k)
        return self.linear_head(pooled)

    def forward_features(self, x: torch.Tensor, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B,T,D), got {tuple(x.shape)}")
        x_feat = x.transpose(1, 2).contiguous()
        h, mask = self._extract_he2rna_features(x_feat)
        tokens = h.transpose(1, 2).contiguous()
        k_use = self.ks[0] if k is None else int(k)
        h_base = self._pool_topk(h, mask, k_use)
        return tokens, h_base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B,T,D), got {tuple(x.shape)}")

        x_feat = x.transpose(1, 2).contiguous()
        h, mask = self._extract_he2rna_features(x_feat)

        if self.head_type == "linear":
            if self.training:
                k = int(np.random.choice(self.ks))
                return self._pool_topk(h, mask, k)
            out = None
            for k in self.ks:
                y = self._pool_topk(h, mask, int(k))
                out = y if out is None else (out + y)
            return out / float(max(len(self.ks), 1))

        tokens = h.transpose(1, 2).contiguous()
        if self.training:
            k = int(np.random.choice(self.ks))
            return self._forward_with_k(tokens, mask, k)

        out = None
        for k in self.ks:
            y = self._forward_with_k(tokens, mask, int(k))
            out = y if out is None else (out + y)
        return out / float(max(len(self.ks), 1))
