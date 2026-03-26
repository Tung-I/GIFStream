# dynamic_default_strategy.py
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .dynamic_ops import (
    dynamic_duplicate,
    dynamic_remove,
    dynamic_reset_opa,
    dynamic_split,
)


@dataclass
class DynamicDefaultStrategy:
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    revised_opacity: bool = False
    absgrad: bool = False
    verbose: bool = False
    key_for_gradient: str = "means2d"

    def initialize_state(self, scene_scale: float) -> Dict[str, Any]:
        return {
            "grad2d": None,       # [N]
            "count": None,        # [N]
            "radii": None,        # [N] or None
            "scene_scale": scene_scale,
        }

    def step_pre_backward(
        self,
        params,
        optimizers,
        state,
        step: int,
        info: Dict[str, Any],
    ):
        assert self.key_for_gradient in info
        info[self.key_for_gradient].retain_grad()

    def _update_state(self, params, state, info, packed: bool = False):
        for key in ["width", "height", "n_cameras", "radii", "gaussian_ids", self.key_for_gradient]:
            assert key in info, f"{key} missing"

        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()

        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        n_gaussian = params["means"].shape[1]  # [T, N, 3] -> Gaussian axis is dim=1

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        

        if packed:
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]
            if radii.ndim > 1:
                radii = radii.max(dim=-1).values
        else:
            radii_raw = info["radii"]
            # Case A: radii is scalar per visible Gaussian, e.g. [C, N] or [N]
            if radii_raw.ndim == grads.ndim - 1:
                sel = radii_raw > 0.0
                radii = radii_raw[sel]

            # Case B: radii is vector-valued per Gaussian, e.g. [C, N, 2]
            elif radii_raw.ndim == grads.ndim:
                sel = (radii_raw > 0.0).all(dim=-1)
                radii = radii_raw[sel].max(dim=-1).values

            else:
                raise RuntimeError(
                    f"Unexpected radii/grads shapes: radii={tuple(radii_raw.shape)}, grads={tuple(grads.shape)}"
                )

            # Robust even if sel is 1D or 2D
            nz = sel.nonzero(as_tuple=False)
            if nz.numel() == 0:
                return
            gs_ids = nz[:, -1]
            grads = grads[sel]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

        if self.refine_scale2d_stop_iter > 0:
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                radii / float(max(info["width"], info["height"]))
            )

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step: int) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)

        # Aggregate 3D size across time conservatively
        scale3d = torch.exp(params["scales"]).amax(dim=-1)     # [T, N]
        scale3d_ref = scale3d.mean(dim=0)                      # [N]
        is_grad_high = grads > self.grow_grad2d
        is_small = scale3d_ref <= self.grow_scale3d * state["scene_scale"]

        is_dupli = is_grad_high & is_small
        n_dupli = int(is_dupli.sum().item())

        is_split = is_grad_high & (~is_small)
        if self.refine_scale2d_stop_iter > 0 and step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = int(is_split.sum().item())

        if n_dupli > 0:
            dynamic_duplicate(params, optimizers, state, is_dupli)

        if n_dupli > 0:
            # appended children from duplicate should not be split in same round
            is_split = torch.cat(
                [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=is_split.device)],
                dim=0
            )

        if n_split > 0:
            dynamic_split(
                params, optimizers, state, is_split,
                revised_opacity=self.revised_opacity
            )

        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(self, params, optimizers, state, step: int) -> int:
        # Conservative synchronized prune:
        # only prune if low-opacity across the entire trajectory
        opa = torch.sigmoid(params["opacities"])               # [T, N]
        opa_ref = opa.max(dim=0).values                        # [N]

        is_prune = opa_ref < self.prune_opa

        if step > self.reset_every:
            scale3d = torch.exp(params["scales"]).amax(dim=-1) # [T, N]
            too_big = scale3d.max(dim=0).values > self.prune_scale3d * state["scene_scale"]
            if self.refine_scale2d_stop_iter > 0 and step < self.refine_scale2d_stop_iter:
                too_big |= state["radii"] > self.prune_scale2d
            is_prune |= too_big

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            dynamic_remove(params, optimizers, state, is_prune)
        return n_prune

    def step_post_backward(
        self,
        params,
        optimizers,
        state,
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            n_prune = self._prune_gs(params, optimizers, state, step)

            if self.verbose:
                print(
                    f"[dynamic refine] step={step} dup={n_dupli} split={n_split} "
                    f"prune={n_prune} now_N={params['means'].shape[1]}"
                )

            state["grad2d"].zero_()
            state["count"].zero_()
            if state["radii"] is not None:
                state["radii"].zero_()

        if step % self.reset_every == 0 and step > 0:
            dynamic_reset_opa(params, optimizers, value=self.prune_opa * 2.0)