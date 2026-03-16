"""Optimizer utilities for MARL training.

Includes an Adam-based optimizer with a lightweight meta-gradient learning-rate
adaptation step, inspired by meta-gradient training schedules.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Adam


class MetaAdam(Adam):
    """Adam optimizer with scalar hypergradient LR adaptation.

    The base optimizer is Adam. Before each step, if enabled and a previous
    parameter update exists, we adjust the group's LR using a hypergradient
    signal computed as:

        h_t = <g_t, u_{t-1}>

    where g_t is current gradient and u_{t-1} is previous parameter update.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        hyper_lr: float = 0.0,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        self.hyper_lr = float(hyper_lr)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self._prev_updates: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def step(self, closure=None):
        # Hypergradient LR adaptation from previous updates.
        if self.hyper_lr > 0.0 and self._prev_updates:
            hypergrad = 0.0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    prev = self._prev_updates.get(id(p))
                    if prev is None:
                        continue
                    hypergrad += float((p.grad.detach() * prev).sum().item())

            for group in self.param_groups:
                lr_old = float(group["lr"])
                lr_new = lr_old - self.hyper_lr * hypergrad
                group["lr"] = max(self.min_lr, min(self.max_lr, lr_new))

        # Snapshot params to recover applied update vector.
        old_params: dict[int, torch.Tensor] = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    old_params[id(p)] = p.detach().clone()

        loss = super().step(closure=closure)

        # Store update direction u_t = p_t(before) - p_t(after)
        self._prev_updates = {}
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                old = old_params.get(id(p))
                if old is None:
                    continue
                self._prev_updates[id(p)] = (old - p.detach()).clone()

        return loss


def make_optimizer(params: Iterable[torch.nn.Parameter], cfg: dict[str, Any]):
    """Create optimizer from config.

    Supported:
    - name=adam
    - name=meta_adam
    """
    name = str(cfg.get("name", "adam")).lower()
    lr = float(cfg.get("lr", 3e-4))
    betas_cfg = cfg.get("betas", [0.9, 0.999])
    betas = (float(betas_cfg[0]), float(betas_cfg[1]))
    eps = float(cfg.get("eps", 1e-8))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    amsgrad = bool(cfg.get("amsgrad", False))

    if name == "adam":
        return Adam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

    if name == "meta_adam":
        meta_cfg = cfg.get("meta", {})
        return MetaAdam(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            hyper_lr=float(meta_cfg.get("hyper_lr", 1e-7)),
            min_lr=float(meta_cfg.get("min_lr", 1e-6)),
            max_lr=float(meta_cfg.get("max_lr", 1e-2)),
        )

    raise ValueError(f"Unsupported optimizer name: {name}")
