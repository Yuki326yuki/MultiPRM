"""Lightweight LoRA implementation (PEFT-free).

The upstream PRIME/MultiRM codebase does not depend on `peft`. For the user's
7B+LoRA experiments (e.g., Qwen2.5-7B), we provide a small LoRA injector.

Design goals:
  - Minimal dependencies: pure PyTorch.
  - Safe defaults: freeze base weights; only LoRA params require grad.
  - Works with FSDP wrapping (LoRA params are normal nn.Parameters).

Limitations:
  - This is *not* a full PEFT implementation (no merging, no adapters stacking).
  - It targets nn.Linear layers only, which is sufficient for Qwen/Llama/Mistral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn


@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        # Common transformer projections
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # MLP projections
        "gate_proj",
        "up_proj",
        "down_proj",
    )


class LoRALinear(nn.Module):
    """A drop-in replacement for nn.Linear with a LoRA low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if r <= 0:
            raise ValueError("LoRA r must be > 0")

        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # LoRA parameters: A: (r, in_features), B: (out_features, r)
        # Initialize A with small random, B with zeros (common practice).
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        # Freeze base weights by default (caller can override).
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base projection
        out = self.base(x)
        # Low-rank update
        # (batch, *, in) -> (batch, *, r) -> (batch, *, out)
        x_d = self.dropout(x)
        update = torch.matmul(x_d, self.lora_A.t())
        update = torch.matmul(update, self.lora_B.t()) * self.scaling
        return out + update


def _parse_cfg(cfg: dict | None) -> LoRAConfig:
    if not cfg:
        return LoRAConfig(enabled=False)
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return LoRAConfig(enabled=False)
    r = int(cfg.get("r", 8))
    alpha = int(cfg.get("alpha", 16))
    dropout = float(cfg.get("dropout", 0.0))
    target = cfg.get("target_modules", None)
    if target is None:
        target_modules = LoRAConfig().target_modules
    else:
        if isinstance(target, str):
            # Allow comma-separated
            target_modules = tuple([t.strip() for t in target.split(",") if t.strip()])
        else:
            target_modules = tuple(list(target))
    return LoRAConfig(enabled=True, r=r, alpha=alpha, dropout=dropout, target_modules=target_modules)


def apply_lora(model: nn.Module, cfg: dict | None, *, verbose: bool = True) -> nn.Module:
    """Inject LoRA adapters into `model` per config.

    Config example (yaml/dict):
      lora:
        enabled: True
        r: 8
        alpha: 16
        dropout: 0.05
        target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    """
    lcfg = _parse_cfg(cfg)
    if not lcfg.enabled:
        return model

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    # Replace matching Linear modules.
    replaced: List[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf not in lcfg.target_modules:
            continue
        parent, attr = _get_parent_module(model, name)
        if parent is None:
            continue
        # Wrap
        wrapped = LoRALinear(module, r=lcfg.r, alpha=lcfg.alpha, dropout=lcfg.dropout)
        setattr(parent, attr, wrapped)
        replaced.append(name)

    # Ensure LoRA params are trainable.
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True

    if verbose:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[LoRA] enabled=True r={lcfg.r} alpha={lcfg.alpha} dropout={lcfg.dropout} targets={list(lcfg.target_modules)}")
        print(f"[LoRA] replaced {len(replaced)} Linear layers")
        if len(replaced) > 0:
            print(f"[LoRA] example replaced: {replaced[:5]}")
        print(f"[LoRA] trainable params: {trainable}/{total} ({trainable/total:.4%})")

    return model


def _get_parent_module(root: nn.Module, qualified_name: str) -> Tuple[Optional[nn.Module], str]:
    """Return (parent_module, attr_name) for a qualified module name."""
    parts = qualified_name.split(".")
    if len(parts) == 1:
        return root, parts[0]
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, parts[-1]
        parent = getattr(parent, p)
    return parent, parts[-1]
