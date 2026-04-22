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
        stop_early=False,
        foreach=False,
    ) -> None:
        self.first_iter = True
        self.stop_early = stop_early
        self.k1 = k1
        self.k2 = k2

        w_lk = None
        theta1 = 0.0
        theta2 = torch.inf

        if higher_state is not None:
            gr = higher_state["param_groups"][0]
            gradient: Tensor = gr["gradient"]
            assert gradient is not None, (
                "Higher optimizer state should have a valid gradient"
            )
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

        # FIXME:
        # if foreach:
        #     self.step = self.step_foreach
        #     group["w_lk"] = [torch.full_like(t, 0.01) for t in self._params]
        # else:
        #     group["w_lk"] = torch.full_like(flatten(self._params), 0.01)

        if w_lk is not None:
            group["w_lk"] = w_lk
        else:
            group["w_lk"] = torch.full_like(flatten(self._params), 0.01)

    @torch.no_grad()
    def compute_thetas(self, gradient, w_lk):

        delta = gradient.abs() / w_lk

        # Prolongation
        s = torch.clamp(-gradient, -delta, delta)

        theta1 = self.k1 * (gradient @ delta).abs().item()
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
        w_lk = group["w_lk"]
        theta1 = group["theta1"]
        theta2 = group["theta2"]

        with torch.enable_grad():
            loss = closure()
            gradient = torch.autograd.grad(
                loss,
                self._params,
                create_graph=True,
            )
            grad_flat = flatten(gradient)

        group["gradient"] = grad_flat

        w_lk = (grad_flat**2 + w_lk**2).sqrt()
        delta = grad_flat.abs() / w_lk

        if theta2 < torch.inf and self.first_iter:
            self.first_iter = False
            norm_delta = delta.norm()
            w_lk *= max(1, norm_delta / theta2)
            delta *= min(1, theta2 / norm_delta)

        group["w_lk"] = w_lk

        if self.stop_early and (grad_flat @ delta).abs().item() < theta1:
            return loss

        # Prolongation
        s_lk = torch.clamp(-grad_flat, -delta, delta)

        # Taylor step
        with torch.enable_grad():
            grad_dot_s = grad_flat @ s_lk
            hvp = torch.autograd.grad(grad_dot_s, self._params, retain_graph=True)

        hvp_flat = flatten(hvp)
        curvature = s_lk @ hvp_flat

        lr = min(1.0, (-s_lk @ grad_flat / curvature).item()) if curvature > 0 else 1.0

        # apply step
        numels = [p.numel() for p in self._params]
        splits = torch.split(s_lk, numels)
        shaped_steps = [s.view_as(p) for s, p in zip(splits, self._params)]

        for p, step in zip(self._params, shaped_steps):
            p.add_(step, alpha=lr)

        return loss

    # PERF: check if this is faster than the regular step
    @torch.no_grad()
    def step_foreach(self, closure: Callable):
        group = self.param_groups[0]
        w_lk = group["w_lk"]
        theta1 = group["theta1"]
        theta2 = group["theta2"]

        with torch.enable_grad():
            loss = closure()
            gradient = torch.autograd.grad(
                loss,
                self._params,
                create_graph=True,
            )

        w_lk = torch._foreach_sqrt(
            torch._foreach_add(
                torch._foreach_pow(gradient, 2), torch._foreach_pow(w_lk, 2)
            )
        )
        delta = torch._foreach_div(torch._foreach_abs(gradient), w_lk)

        if theta2 < torch.inf and self.first_iter:
            self.first_iter = False
            norm_delta = torch._foreach_norm(delta)
            w_lk = torch._foreach_mul(
                torch._foreach_maximum(torch._foreach_div(norm_delta, theta2), 1),
                w_lk,
            )
            delta = torch._foreach_mul(
                torch._foreach_minimum(torch._foreach_div(theta2, norm_delta), 1),
                delta,
            )

        group["w_lk"] = w_lk

        with torch.enable_grad():
            s_lk = torch._foreach_minimum(
                torch._foreach_maximum(
                    torch._foreach_neg(gradient), torch._foreach_neg(delta)
                ),
                delta,
            )

            # Taylor step
            grad_dot_s = [torch.sum(t) for t in torch._foreach_mul(gradient, s_lk)]

            hvp = torch.autograd.grad(grad_dot_s, self._params, retain_graph=True)

        curvature = [torch.sum(t) for t in torch._foreach_mul(s_lk, hvp)]

        lrs = []
        for s, g, curv in zip(s_lk, gradient, curvature):
            if curv.item() > 0:
                lrs.append(
                    min(1.0, (torch.dot(-s.flatten(), g.flatten()) / curv).item())
                )
            else:
                lrs.append(1.0)

        for p, step, lr in zip(self._params, s_lk, lrs):
            p.add_(step, alpha=lr)

        return loss
