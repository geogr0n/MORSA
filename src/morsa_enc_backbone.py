import contextlib

import torch
import torch.nn as nn

from heads import make_output_head


class MORSAEncoderBackbone(nn.Module):
    """
    x(B,T,D) -> LayerNorm+Linear(D->r) -> covariance -> logm -> upper-tri vec
    -> bottleneck -> head
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_outputs: int,
        morsa_r: int = 256,
        morsa_eps: float = 1e-3,
        morsa_latent_dim: int = 1024,
        morsa_dropout: float = 0.1,
        morsa_mlp_depth: int = 1,
        device: str | torch.device = "cuda:0",
        head_type: str = "linear",
        rank_k: int = 16,
        output_mu=None,
        output_basis=None,
        diagonal_covariance: bool = False,
    ):
        super().__init__()
        self.device = device
        self.input_dim = int(input_dim)
        self.num_outputs = int(num_outputs)
        self.head_type = str(head_type).lower()
        self.r = int(morsa_r)
        self.eps = float(morsa_eps)
        self.diagonal_covariance = bool(diagonal_covariance)

        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.r),
        )
        tri_idx = torch.triu_indices(self.r, self.r)
        self.register_buffer("tri_i", tri_idx[0], persistent=False)
        self.register_buffer("tri_j", tri_idx[1], persistent=False)
        self.register_buffer("eye_r", torch.eye(self.r), persistent=False)

        vec_dim = self.r * (self.r + 1) // 2
        depth = max(int(morsa_mlp_depth), 1)
        layers: list[nn.Module] = [
            nn.Linear(vec_dim, int(morsa_latent_dim)),
            nn.GELU(),
            nn.Dropout(float(morsa_dropout)),
            nn.LayerNorm(int(morsa_latent_dim)),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(int(morsa_latent_dim), int(morsa_latent_dim)),
                    nn.GELU(),
                    nn.Dropout(float(morsa_dropout)),
                    nn.LayerNorm(int(morsa_latent_dim)),
                ]
            )
        layers.append(nn.Linear(int(morsa_latent_dim), int(morsa_latent_dim)))
        self.bottleneck = nn.Sequential(*layers)

        latent = int(morsa_latent_dim)
        self.linear_head = make_output_head(
            hidden_dim=latent,
            num_outputs=self.num_outputs,
            head_type=self.head_type,
            rank_k=rank_k,
            mu=output_mu,
            U_k=output_basis,
        )

    def _cov(self, H: torch.Tensor) -> torch.Tensor:
        B, T, d = H.shape
        Hc = H - H.mean(dim=1, keepdim=True)
        denom = float(max(T - 1, 1))
        C = torch.bmm(Hc.transpose(1, 2), Hc) / denom
        I = self.eye_r.to(device=H.device, dtype=H.dtype).unsqueeze(0).expand(B, -1, -1)
        tr = C.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        scale = (tr / float(d)).clamp_min(1e-6).view(B, 1, 1)
        C = C / scale
        alpha = 0.05
        C = (1.0 - alpha) * C + alpha * I
        C = C + self.eps * I
        if self.diagonal_covariance:
            C = torch.diag_embed(torch.diagonal(C, dim1=-2, dim2=-1))
        return C

    def _logm_morsa_cov(self, C: torch.Tensor) -> torch.Tensor:
        C = C.float()
        evals, evecs = torch.linalg.eigh(C)
        evals = torch.clamp(evals, min=self.eps)
        log_e = torch.log(evals)
        return evecs @ torch.diag_embed(log_e) @ evecs.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape (B,T,D), got {tuple(x.shape)}")
        ctx = torch.amp.autocast("cuda", enabled=False) if x.is_cuda else contextlib.nullcontext()
        with ctx:
            z = self.proj(x.float())
            C = self._cov(z)
            S = self._logm_morsa_cov(C)
            h = S[:, self.tri_i, self.tri_j]
            h = self.bottleneck(h)
            y = self.linear_head(h)
        return y
