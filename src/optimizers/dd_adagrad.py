from enum import StrEnum, auto
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import StateDict


def flatten(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.flatten() for t in tensors])


class MOMENTUM(StrEnum):
    HEAVY_BALL = auto()
    EMA = auto()  # Exponential Moving Average


class DD_Adagrad(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        higher_state: StateDict | None = None,
        lr: float | None = None,
        k1=0.001,
        k2=2.0,
        beta=0.9,  # momentum
        momentum_type=MOMENTUM.EMA,
        clamp_momentum=False,
        use_norms=False,
        device=None,
    ) -> None:
        self.lr = lr
        self.use_norms = use_norms
        self.momentum_type = momentum_type
        self.clamp_momentum = clamp_momentum
        self.k1 = k1
        self.k2 = k2

        w_lk = None
        moment = None
        theta1 = 0.0
        theta2 = torch.inf

        if higher_state is not None:
            gr = higher_state["param_groups"][0]
            gradient = gr["gradient"]
            assert gradient is not None, (
                "Higher optimizer state should have a valid gradient"
            )
            w_lk = torch.clone(gr["w_lk"])
            moment = torch.clone(gr["moment"])
            theta1, theta2 = self.compute_thetas(gradient, w_lk)

        defaults = {
            "theta1": theta1,
            "theta2": theta2,
            "gradient": None,
            "beta": beta,
        }
        super().__init__(params, defaults)

        # TODO: look into parameter groups
        if len(self.param_groups) != 1:
            raise ValueError(
                "DD Adagrad doesn't support per-parameter options (parameter groups)"
            )

        group = self.param_groups[0]
        self._params = group["params"]
        group["first_iter"] = True
        # NOTE: this requires the model to be moved to device before calling this constructor
        self.device = device if device is not None else self._params[0].device

        flat_params = flatten(self._params)
        group["w_lk"] = (
            w_lk
            if w_lk is not None
            else torch.full_like(flat_params, 0.01, device=self.device)
        )

        group["moment"] = (
            moment
            if moment is not None
            else torch.zeros_like(flat_params, device=self.device)
        )

    @torch.no_grad()
    def compute_thetas(self, gradient: Tensor, w_lk: Tensor):

        delta = gradient.abs() / w_lk

        # Prolongation
        s = torch.clamp(-gradient, -delta, delta)

        # NOTE: not used
        # theta1 = self.k1 * (gradient @ delta).abs().item()
        theta1 = 0.0

        theta2 = self.k2 * s.norm().item()

        return theta1, theta2

    @torch.no_grad()
    def step(self, closure: Callable):  # type: ignore[override]
        """
        Perform a single optimization step to update parameter.
        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Can not call `loss.backward()`
        """
        group = self.param_groups[0]
        w_lk = group["w_lk"].to(self.device)
        theta2 = group["theta2"]
        beta = group["beta"]

        with torch.enable_grad():
            loss = closure()
            gradient = torch.autograd.grad(
                loss,
                self._params,
                create_graph=True,
            )
            grad_flat = flatten(gradient).to(self.device)

        group["gradient"] = grad_flat

        w_new = (grad_flat**2 + w_lk**2).sqrt()
        delta = grad_flat.abs() / w_new

        if theta2 < torch.inf and group["first_iter"]:
            if self.use_norms:
                norm_delta = delta.norm()
                w_new *= max(1, norm_delta / theta2)
                delta *= min(1, theta2 / norm_delta)
            else:
                w_new = torch.max(w_new, w_lk)
                delta = grad_flat.abs() / w_new

        group["w_lk"] = w_new

        # Prolongation
        s_lk = torch.clamp(-grad_flat, -delta, delta)

        # Taylor step
        # FIXME: check for tolerances
        with torch.enable_grad():
            grad_dot_s = grad_flat @ s_lk
            hvp = torch.autograd.grad(grad_dot_s, self._params, retain_graph=True)

        hvp_flat = flatten(hvp).to(self.device)
        # s^T @ B @ s
        curvature = s_lk @ hvp_flat

        # gamma
        if self.lr is None:
            lr = (
                min(1.0, (-grad_flat @ s_lk / curvature).item())
                if curvature > 0
                else 1.0
            )
        else:
            lr = self.lr

        moment: Tensor = group["moment"]  # get reference
        match self.momentum_type:
            case MOMENTUM.HEAVY_BALL:
                moment.mul_(beta).add_(s_lk, alpha=lr)
            case MOMENTUM.EMA:
                moment.mul_(beta).add_(s_lk, alpha=lr * (1 - beta))

        if self.clamp_momentum:
            moment.clamp_(-delta, delta)

        # apply step
        splits = torch.split(moment, [p.numel() for p in self._params])
        shaped_steps = [s.view_as(p) for s, p in zip(splits, self._params)]

        for p, step in zip(self._params, shaped_steps):
            p.add_(step.to(p.device))
        group["first_iter"] = False

        return loss
