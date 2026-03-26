# dynamic_ops.py
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Dict, List, Union

from gsplat.utils import normalized_quat_to_rotmat


@torch.no_grad()
def _update_param_with_optimizer_dim1(
    param_fn: Callable[[str, Tensor], Tensor],
    optimizer_fn: Callable[[str, Tensor], Tensor],
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    names: List[str] | None = None,
):
    if names is None:
        names = list(params.keys())

    for name in names:
        old_param = params[name]
        new_param = param_fn(name, old_param)
        params[name] = new_param

        if name not in optimizers:
            assert not old_param.requires_grad
            continue

        opt = optimizers[name]
        for i in range(len(opt.param_groups)):
            state = opt.state[old_param]
            del opt.state[old_param]

            for k, v in state.items():
                if k == "step":
                    continue
                if torch.is_tensor(v):
                    state[k] = optimizer_fn(k, v)

            opt.param_groups[i]["params"] = [new_param]
            opt.state[new_param] = state


@torch.no_grad()
def dynamic_duplicate(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,  # [N]
):
    device = mask.device
    sel = torch.where(mask)[0]

    if sel.numel() == 0:
        return

    def param_fn(name: str, p: Tensor) -> Tensor:
        # p is [T, N, ...]
        p_new = torch.cat([p, p[:, sel].clone()], dim=1)
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        # optimizer states have same shape as params
        zeros = torch.zeros_like(v[:, sel], device=device)
        return torch.cat([v, zeros], dim=1)

    _update_param_with_optimizer_dim1(param_fn, optimizer_fn, params, optimizers)

    # state tensors are [N]
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == mask.shape[0]:
            state[k] = torch.cat([v, v[sel]], dim=0)


@torch.no_grad()
def dynamic_remove(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,  # [N], True means remove
):
    keep = torch.where(~mask)[0]
    if keep.numel() == mask.numel():
        return

    def param_fn(name: str, p: Tensor) -> Tensor:
        p_new = p[:, keep]
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[:, keep]

    _update_param_with_optimizer_dim1(param_fn, optimizer_fn, params, optimizers)

    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == mask.shape[0]:
            state[k] = v[keep]


@torch.no_grad()
def dynamic_reset_opa(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    value: float,
):
    cap = torch.logit(torch.tensor(value, device=params["opacities"].device)).item()

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name != "opacities":
            raise ValueError(name)
        p_new = torch.clamp(p, max=cap)
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    _update_param_with_optimizer_dim1(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )


@torch.no_grad()
def dynamic_split(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    mask: Tensor,  # [N]
    revised_opacity: bool = False,
):
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    if sel.numel() == 0:
        return

    # params are [T, N, ...]
    scales = torch.exp(params["scales"][:, sel])              # [T, S, 3]
    quats = F.normalize(params["quats"][:, sel], dim=-1)      # [T, S, 4]
    T, S = scales.shape[:2]

    rotmats = normalized_quat_to_rotmat(quats.reshape(-1, 4)).reshape(T, S, 3, 3)

    # Shared canonical offsets across time for temporal coherence
    eps = torch.randn(2, S, 3, device=device)                 # [2, S, 3]
    local = scales[:, None] * eps[None]                       # [T, 2, S, 3]
    samples = torch.einsum("tsij,tbsj->tbsi", rotmats, local) # [T, 2, S, 3]

    parent_means = params["means"][:, sel]                    # [T, S, 3]
    child_means_0 = parent_means + samples[:, 0]
    child_means_1 = parent_means + samples[:, 1]

    child_scales = torch.log(scales / 1.6)                    # [T, S, 3]

    def param_fn(name: str, p: Tensor) -> Tensor:
        base = p[:, rest]

        if name == "means":
            p_new = torch.cat([base, child_means_0, child_means_1], dim=1)
        elif name == "scales":
            p_new = torch.cat([base, child_scales, child_scales], dim=1)
        elif name == "opacities" and revised_opacity:
            new_opa = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[:, sel]))
            new_opa = torch.logit(new_opa.clamp(1e-6, 1 - 1e-6))
            p_new = torch.cat([base, new_opa, new_opa], dim=1)
        else:
            p_new = torch.cat([base, p[:, sel], p[:, sel]], dim=1)

        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        zeros0 = torch.zeros_like(v[:, sel], device=device)
        zeros1 = torch.zeros_like(v[:, sel], device=device)
        return torch.cat([v[:, rest], zeros0, zeros1], dim=1)

    _update_param_with_optimizer_dim1(param_fn, optimizer_fn, params, optimizers)

    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == mask.shape[0]:
            state[k] = torch.cat([v[rest], v[sel], v[sel]], dim=0)