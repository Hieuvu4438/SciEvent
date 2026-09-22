"""Role representations: learned (H0), definition prototypes (H2), and
frequency-adaptive blending (H3).

``p_role[r] = lambda_r * W_def p_def[r] + (1 - lambda_r) * p_emp[r]``

with ``lambda_r = kappa / (kappa + n_r)`` in the adaptive mode, where ``n_r`` is
the **TRAIN-only** frequency of role ``r``. Rare roles stay close to their
annotation-codebook definition; frequent roles are free to learn from data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.schema import SEMANTIC_ROLES


class RolePrototypes(nn.Module):
    """Modes: ``learned`` (H0), ``fixed`` (H2), ``adaptive`` (H3)."""

    VALID_MODES = ("learned", "fixed", "adaptive")

    def __init__(
        self,
        hidden_size: int,
        mode: str = "learned",
        definition_vectors: torch.Tensor | None = None,
        role_frequencies: dict[str, int] | None = None,
        kappa: float = 20.0,
        fixed_lambda: float = 0.5,
        anchor_weight: float = 0.0,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown prototype mode: {mode!r}")
        self.mode = mode
        self.anchor_weight = anchor_weight
        self.num_roles = len(SEMANTIC_ROLES)

        self.empirical = nn.Embedding(self.num_roles, hidden_size)
        nn.init.normal_(self.empirical.weight, std=0.02)

        if mode == "learned":
            self.register_buffer("lambdas", torch.zeros(self.num_roles), persistent=False)
            self.definition_proj = None
            return

        if definition_vectors is None:
            raise ValueError(f"mode={mode!r} requires definition vectors")
        if definition_vectors.shape[0] != self.num_roles:
            raise ValueError("definition vectors must cover all nine semantic roles")

        # Frozen: encoded once with the initial backbone, never fine-tuned.
        self.register_buffer("definitions", definition_vectors.float(), persistent=True)
        self.definition_proj = nn.Linear(definition_vectors.shape[1], hidden_size)

        if mode == "fixed":
            lambdas = torch.full((self.num_roles,), float(fixed_lambda))
        else:
            freqs = role_frequencies or {}
            lambdas = torch.tensor(
                [kappa / (kappa + float(freqs.get(r, 0))) for r in SEMANTIC_ROLES],
                dtype=torch.float32,
            )
        self.register_buffer("lambdas", lambdas, persistent=True)

    def forward(self) -> torch.Tensor:
        """Returns ``[num_roles, hidden]``."""
        empirical = self.empirical.weight
        if self.definition_proj is None:
            return empirical
        projected = self.definition_proj(self.definitions.to(empirical.dtype))
        lam = self.lambdas.to(empirical.dtype).unsqueeze(-1)
        return lam * projected + (1.0 - lam) * empirical

    def anchor_loss(self) -> torch.Tensor:
        """``L_proto = sum_r lambda_r * ||norm(p_emp) - norm(W_def p_def)||^2``."""
        if self.definition_proj is None or self.anchor_weight <= 0:
            return self.empirical.weight.new_zeros(())
        projected = self.definition_proj(self.definitions.to(self.empirical.weight.dtype))
        diff = F.normalize(self.empirical.weight, dim=-1) - F.normalize(projected, dim=-1)
        return (self.lambdas.to(diff.dtype) * diff.pow(2).sum(-1)).sum()


@torch.no_grad()
def encode_role_definitions(
    definitions: dict[str, str], tokenizer, backbone, device: torch.device
) -> torch.Tensor:
    """Encode each codebook definition once with the *initial* frozen backbone."""
    texts = [definitions[r] for r in SEMANTIC_ROLES]
    batch = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(device)
    out = backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    states = out.last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(states.dtype)
    pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return pooled.float().cpu()
