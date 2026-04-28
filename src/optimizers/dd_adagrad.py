from copy import deepcopy
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import StateDict


def flatten(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.flatten() for t in tensors])


class DD_Adagrad(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        higher_state: StateDict | None = None,
        k1=0.001,
        k2=2,
        use_norms=True,
        foreach=False,
        device=None,
    ) -> None:
        self.first_iter = True
        self.use_norms = use_norms
        self.k1 = k1
        self.k2 = k2

        w_lk = None
        theta1 = 0.0
        theta2 = torch.inf

        if higher_state is not None:
            gr = higher_state["param_groups"][0]
            gradient = gr["gradient"]
            assert gradient is not None, (
                "Higher optimizer state should have a valid gradient"
            )
            if foreach:
                w_lk = deepcopy(gr["w_lk"])
            else:
                w_lk = torch.clone(gr["w_lk"])
                theta1, theta2 = self.compute_thetas(gradient, w_lk)

        defaults = {
            "theta1": theta1,
            "theta2": theta2,
            "gradient": None,
        }
        super().__init__(params, defaults)

        # TODO: look into parameter groups
        if len(self.param_groups) != 1:
            raise ValueError(
                "DD Adagrad doesn't support per-parameter options (parameter groups)"
            )

        group = self.param_groups[0]
        self._params = group["params"]
        # NOTE: this requires the model to be moved to device before calling this constructor
        self.device = device if device is not None else self._params[0].device

        if foreach:
            self.step = self.step_foreach

        if w_lk is not None:
            group["w_lk"] = w_lk
        else:
            if foreach:
                group["w_lk"] = [
                    torch.full_like(
                        t,
                        0.01,
                        device=self.device,
                    )
                    for t in self._params
                ]
            else:
                group["w_lk"] = torch.full_like(
                    flatten(self._params),
                    0.01,
                    device=self.device,
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

    # @torch.no_grad()
    # def get_thetas(self, params):
    #     group = self.param_groups[0]
    #     w_lk = group["w_lk"]
    #
    #     grad_flat = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
    #
    #     w_lk = (grad_flat**2 + w_lk**2).sqrt()
    #     delta = grad_flat.abs() / w_lk
    #
    #     group["w_lk"] = w_lk
    #
    #     # Prolongation
    #     s = torch.clamp(-grad_flat, -delta, delta)
    #
    #     theta1 = self.k1 * (grad_flat @ delta).abs().item()
    #     theta2 = self.k2 * s.norm().item()
    #
    #     return theta1, theta2

    @torch.no_grad()
    def step(self, closure: Callable):  # type: ignore[override]
        group = self.param_groups[0]
        w_lk = group["w_lk"].to(self.device)
        theta2 = group["theta2"]

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

        if theta2 < torch.inf and self.first_iter:
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
        with torch.enable_grad():
            grad_dot_s = grad_flat @ s_lk
            hvp = torch.autograd.grad(grad_dot_s, self._params, retain_graph=True)

        hvp_flat = flatten(hvp).to(self.device)
        # s^T @ B @ s
        curvature = s_lk @ hvp_flat

        # gamma
        lr = min(1.0, (-grad_flat @ s_lk / curvature).item()) if curvature > 0 else 1.0

        # apply step
        splits = torch.split(s_lk, [p.numel() for p in self._params])
        shaped_steps = [s.view_as(p) for s, p in zip(splits, self._params)]

        for p, step in zip(self._params, shaped_steps):
            p.add_(step.to(p.device), alpha=lr)
        self.first_iter = False

        return loss

    @torch.no_grad()
    def step_foreach(self, closure: Callable):
        group = self.param_groups[0]
        w_lk = group["w_lk"]

        with torch.enable_grad():
            loss = closure()
            gradient = torch.autograd.grad(
                loss,
                self._params,
                create_graph=True,
            )

        group["gradient"] = gradient

        w_new = torch._foreach_sqrt(
            torch._foreach_add(
                torch._foreach_pow(gradient, 2), torch._foreach_pow(w_lk, 2)
            )
        )

        if self.first_iter:
            w_new = torch._foreach_maximum(w_new, w_lk)

        delta = torch._foreach_div(torch._foreach_abs(gradient), w_new)

        group["w_lk"] = w_new

        s_lk = torch._foreach_clamp_min(
            torch._foreach_clamp_max(
                torch._foreach_neg(gradient), torch._foreach_neg(delta)
            ),
            delta,
        )

        with torch.enable_grad():
            # Taylor step
            grad_dot_s = [torch.sum(t) for t in torch._foreach_mul(gradient, s_lk)]

            hvp = torch.autograd.grad(grad_dot_s, self._params, retain_graph=True)

        curvature = [torch.sum(t) for t in torch._foreach_mul(s_lk, hvp)]

        # NOTE: lr is computed per layer
        lrs = [
            min(1.0, (torch.dot(-s.flatten(), g.flatten()) / curv).item())
            if curv.item() > 0
            else 1.0
            for s, g, curv in zip(s_lk, gradient, curvature)
        ]

        for p, step, lr in zip(self._params, s_lk, lrs):
            p.add_(step, alpha=lr)
        self.first_iter = False

        return loss
