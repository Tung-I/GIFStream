#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_eval_GIFStream.py

Goal
----
A *checkpoint-free* evaluator for GIFStream end2end compression.

It reconstructs the dynamic GS model *only* from:
  <compress_dir>/meta.json
  <compress_dir>/*.bin, *.png, *.npz
  <compress_dir>/nets.pt     (decoder weights + entropy-model weights + scaling)

Then it produces (under out_dir):
  1) payload_sizes.json          : per-attribute + total bytes/MB (from files in compress_dir)
  2) decode_timing.json          : per-attribute decode time + postprocess + totals
  3) render_timing.json          : per-image render time, rasterization-kernel time, totals/averages
  4) renders/*.png               : rendered RGB images from reconstructed model (val split by default)

Usage
-----
python examples/eval_end2end_GIFStream.py \
  --compress_dir results_ori_codec/flame_steak/GOP_0/r0/compression/rank0 \
  --out_dir      results_ori/flame_steak/GOP_0/r0/my_eval 


Notes
-----
- This script does NOT load any raw splats from ckpt_*.pt. It reconstructs splats only from compressed files.
- It DOES load nets.pt (required to reconstruct decoders + entropy models).
- It tries to read ../cfg.yml (two levels up from compress_dir) for dataset settings if present.
  If not found, you must provide --data_dir at minimum.
"""

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class _TolerantLoader(yaml.SafeLoader):
    """SafeLoader that ignores unknown YAML tags like !!python/object:*."""
    pass

def _construct_unknown(loader, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)

# “None” means: apply to any undefined/unknown tag
_TolerantLoader.add_constructor(None, _construct_unknown)

from torch import Tensor

from datasets.GIFStream_new import Dataset, Parser
from gsplat.compression import GIFStreamEnd2endCompression
from gsplat.compression_simulation.entropy_model import ConditionEntropy
from gsplat.rendering import rasterization, view_to_visible_anchors


# -------------------------
# Utilities
# -------------------------

def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _now() -> float:
    return time.time()

def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)

def inverse_sigmoid(x: Tensor) -> Tensor:
    # match trainer's inverse_sigmoid (with clamp)
    x = x.clamp(1e-7, 1 - 1e-7)
    return -torch.log(1.0 / x - 1.0)

def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    # copied from trainer
    if quaternion.dim() == 1:
        quaternion = quaternion.unsqueeze(0)
    w, x, y, z = quaternion.unbind(dim=-1)
    B = quaternion.size(0)
    rotation_matrix = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(B, 3, 3)
    return rotation_matrix


# -------------------------
# Payload accounting
# -------------------------

def compute_payload_by_attribute(compress_dir: str) -> Dict[str, Any]:
    """
    Groups files in compress_dir by attribute as used by GIFStreamEnd2endCompression.
    Includes nets.pt + meta.json because they are required to reconstruct the model.
    """
    patterns = {
        "meta": ["meta.json"],
        "nets": ["nets.pt"],
        "anchors": ["anchors_l.png", "anchors_u.png"],
        "scales": ["scales.bin"],
        "offsets": ["offsets.bin"],
        "factors": ["factors.bin"],
        "quats": ["quats.npz"],
        "opacities": ["opacities.npz"],
        "anchor_features": ["anchor_features_*.bin"],
        "time_features": ["time_features_*.bin"],
    }

    used_files: Dict[str, List[str]] = {}
    used_bytes: Dict[str, int] = {}

    all_files = sorted(
        [os.path.join(compress_dir, f) for f in os.listdir(compress_dir)]
    )
    all_files_set = set(all_files)

    accounted = set()

    for attr, pats in patterns.items():
        files: List[str] = []
        for p in pats:
            if "*" in p:
                files.extend(sorted(glob.glob(os.path.join(compress_dir, p))))
            else:
                fp = os.path.join(compress_dir, p)
                if os.path.exists(fp):
                    files.append(fp)
        files = sorted(list(dict.fromkeys(files)))
        used_files[attr] = [os.path.relpath(f, compress_dir) for f in files]
        used_bytes[attr] = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
        accounted.update(files)

    # Anything else present in compress_dir
    other_files = sorted(list(all_files_set - accounted))
    used_files["other"] = [os.path.relpath(f, compress_dir) for f in other_files]
    used_bytes["other"] = sum(os.path.getsize(f) for f in other_files if os.path.isfile(f))

    total_bytes = sum(used_bytes.values())

    out = {
        "compress_dir": compress_dir,
        "by_attribute_bytes": used_bytes,
        "by_attribute_mb": {k: _bytes_to_mb(v) for k, v in used_bytes.items()},
        "total_bytes": total_bytes,
        "total_mb": _bytes_to_mb(total_bytes),
        "by_attribute_files": used_files,
    }
    return out


# -------------------------
# Model reconstruction from compress_dir
# -------------------------

@dataclass
class InferredHParams:
    feature_dim: int
    n_offsets: int
    gop: int
    c_perframe: int
    c_channel: int        # AR block size for anchor_features coding
    p_channel: int        # AR block size for time_features coding
    time_dim: int
    phi: float

    # Optional flags (best-effort inference)
    view_adaptive: bool
    add_opacity_dist: bool
    add_cov_dist: bool
    add_color_dist: bool
    app_embed_dim: int    # if > 0, we inject zeros embedding (since app_module is not stored in nets.pt)


def infer_hparams_from_meta_and_decoders(meta: Dict[str, Any], dec_state: Dict[str, Any], phi_default: float) -> InferredHParams:
    # From meta shapes
    feature_dim = int(meta["anchor_features"]["shape"][1])
    n_offsets = int(meta["offsets"]["shape"][1])
    gop = int(meta["time_features"]["shape"][1])
    c_perframe = int(meta["time_features"]["shape"][2])

    c_channel = int(meta["anchor_features"]["channel"])
    p_channel = int(meta["time_features"]["channel"])

    # From mlp_motion first layer in_features
    # dec_state keys look like: "mlp_motion.0.weight"
    w_motion0 = dec_state["mlp_motion.0.weight"]
    in_motion = int(w_motion0.shape[1])
    time_dim = in_motion - feature_dim - c_perframe
    if time_dim < 0:
        raise ValueError(f"Cannot infer time_dim: got in_motion={in_motion}, feature_dim={feature_dim}, c_perframe={c_perframe}")

    # Infer view/dist/app flags from layer0 in_features (best-effort)
    def _infer_view_and_dist(in_features: int, base: int) -> Tuple[bool, bool]:
        extra = in_features - base
        view = extra in (3, 4)
        dist = extra in (1, 4)
        return view, dist

    base = feature_dim + c_perframe

    in_opacity = int(dec_state["mlp_opacity.0.weight"].shape[1])
    view_adaptive, add_opacity_dist = _infer_view_and_dist(in_opacity, base)

    in_cov = int(dec_state["mlp_cov.0.weight"].shape[1])
    view_cov, add_cov_dist = _infer_view_and_dist(in_cov, base)
    view_adaptive = view_adaptive or view_cov

    in_color = int(dec_state["mlp_color.0.weight"].shape[1])

    # For color: extra could be view(3) + color_dist(1) + app_embed_dim(>=0)
    extra_color = in_color - base
    # Prefer consistent view_adaptive
    view_dim = 3 if view_adaptive else 0

    # Decide color_dist dim
    # Try (color_dist=1) first if it makes app_embed_dim non-negative and "reasonable"
    candidates = []
    for color_dist_dim in (0, 1):
        app_embed_dim = extra_color - view_dim - color_dist_dim
        if app_embed_dim >= 0:
            candidates.append((color_dist_dim, app_embed_dim))
    if not candidates:
        # fallback: no view, no dist, app = extra_color if possible
        add_color_dist = False
        app_embed_dim = max(0, extra_color)
    else:
        # pick smallest app_embed_dim (most likely app_opt=False)
        color_dist_dim, app_embed_dim = sorted(candidates, key=lambda x: x[1])[0]
        add_color_dist = (color_dist_dim == 1)

    return InferredHParams(
        feature_dim=feature_dim,
        n_offsets=n_offsets,
        gop=gop,
        c_perframe=c_perframe,
        c_channel=c_channel,
        p_channel=p_channel,
        time_dim=time_dim,
        phi=phi_default,
        view_adaptive=view_adaptive,
        add_opacity_dist=add_opacity_dist,
        add_cov_dist=add_cov_dist,
        add_color_dist=add_color_dist,
        app_embed_dim=int(app_embed_dim),
    )


def build_decoders_from_state(dec_state: Dict[str, Any], device: str) -> nn.ModuleDict:
    """
    Rebuild decoder architectures purely from state_dict shapes,
    then load weights from dec_state.
    """
    # Infer key dims
    # mlp_opacity: Linear(in0 -> feat) -> ReLU -> Linear(feat -> n_offsets) -> Tanh
    w0 = dec_state["mlp_opacity.0.weight"]
    feat_dim = int(w0.shape[0])
    in0 = int(w0.shape[1])
    w2 = dec_state["mlp_opacity.2.weight"]
    n_offsets = int(w2.shape[0])

    mlp_opacity = nn.Sequential(
        nn.Linear(in0, feat_dim),
        nn.ReLU(True),
        nn.Linear(feat_dim, n_offsets),
        nn.Tanh(),
    )

    # mlp_cov: Linear(in0 -> feat) -> ReLU -> Linear(feat -> 7*n_offsets)
    w0 = dec_state["mlp_cov.0.weight"]
    in0_cov = int(w0.shape[1])
    w2 = dec_state["mlp_cov.2.weight"]
    out_cov = int(w2.shape[0])
    mlp_cov = nn.Sequential(
        nn.Linear(in0_cov, feat_dim),
        nn.ReLU(True),
        nn.Linear(feat_dim, out_cov),
    )

    # mlp_color: Linear(in0 -> feat) -> ReLU -> Linear(feat -> 3*n_offsets) -> Sigmoid
    w0 = dec_state["mlp_color.0.weight"]
    in0_col = int(w0.shape[1])
    w2 = dec_state["mlp_color.2.weight"]
    out_col = int(w2.shape[0])
    mlp_color = nn.Sequential(
        nn.Linear(in0_col, feat_dim),
        nn.ReLU(True),
        nn.Linear(feat_dim, out_col),
        nn.Sigmoid(),
    )

    # mlp_motion: Linear(in0 -> feat) -> ReLU -> Linear(feat -> 7)
    w0 = dec_state["mlp_motion.0.weight"]
    in0_mot = int(w0.shape[1])
    w2 = dec_state["mlp_motion.2.weight"]
    out_mot = int(w2.shape[0])
    mlp_motion = nn.Sequential(
        nn.Linear(in0_mot, feat_dim),
        nn.ReLU(True),
        nn.Linear(feat_dim, out_mot),
    )

    decoders = nn.ModuleDict(
        {
            "mlp_opacity": mlp_opacity,
            "mlp_cov": mlp_cov,
            "mlp_color": mlp_color,
            "mlp_motion": mlp_motion,
        }
    ).to(device)

    decoders.load_state_dict(dec_state, strict=True)
    decoders.eval()
    return decoders


def build_entropy_models_from_nets(
    meta: Dict[str, Any],
    nets: Dict[str, Any],
    device: str,
) -> Dict[str, Optional[nn.Module]]:
    """
    Reconstruct ConditionEntropy models needed for end2end decompress.
    Uses dims inferred from meta.json (no ckpt needed).
    """
    feature_dim = int(meta["anchor_features"]["shape"][1])
    n_offsets = int(meta["offsets"]["shape"][1])
    c_channel = int(meta["anchor_features"]["channel"])
    p_channel = int(meta["time_features"]["channel"])

    entropy_models: Dict[str, Optional[nn.Module]] = {}
    for k in meta.keys():
        entropy_models[k] = None

    # These are the ones created in GIFStreamCompressionSimulation when entropy_model_enable=True
    entropy_models["scales"] = ConditionEntropy(feature_dim, 18, 8).to(device)
    entropy_models["anchor_features"] = ConditionEntropy(3 * c_channel, 3 * c_channel, 12).to(device)
    entropy_models["offsets"] = ConditionEntropy(feature_dim, 9 * n_offsets, 16).to(device)
    entropy_models["factors"] = ConditionEntropy(feature_dim, 12, 8).to(device)
    entropy_models["time_features"] = ConditionEntropy(3 * p_channel, 3 * p_channel, 12).to(device)

    # Load weights if present
    for name in ["scales", "anchor_features", "offsets", "factors", "time_features"]:
        key = f"{name}_entropy_model"
        if key in nets and nets[key] is not None:
            entropy_models[name].load_state_dict(nets[key], strict=True)
        entropy_models[name].eval()

    return entropy_models


def decode_splats_with_timing(
    compress_dir: str,
    device: str,
    entropy_models: Dict[str, Optional[nn.Module]],
) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
    """
    Decode all attributes from compressed_dir with per-attribute timing.

    Returns:
      splats: dict of reconstructed tensors (same semantic domains as trainer after decompress())
      timing: dict of decode times
    """
    compressor = GIFStreamEnd2endCompression()
    meta_fp = os.path.join(compress_dir, "meta.json")
    with open(meta_fp, "r") as f:
        meta = json.load(f)

    timing: Dict[str, float] = {}

    # 1) decode anchor_features first (required as condition for others)
    _sync(device)
    t0 = _now()
    fn = compressor._get_decompress_fn("anchor_features")
    anchor_features = fn(
        compress_dir,
        "anchor_features",
        meta["anchor_features"],
        entropy_model=entropy_models["anchor_features"],
        device=device,
    )
    _sync(device)
    timing["decode_anchor_features_sec"] = _now() - t0

    splats: Dict[str, Tensor] = {"anchor_features": anchor_features}

    # 2) decode all other params
    for param_name, param_meta in meta.items():
        if param_name == "anchor_features":
            continue

        fn = compressor._get_decompress_fn(param_name)

        _sync(device)
        t0 = _now()
        out = fn(
            compress_dir,
            param_name,
            param_meta,
            anchor_features=splats["anchor_features"],
            entropy_model=entropy_models.get(param_name, None),
            device=device,
        )
        _sync(device)
        timing[f"decode_{param_name}_sec"] = _now() - t0

        splats[param_name] = out

    # 3) Postprocess (match GIFStreamEnd2endCompression.decompress)
    #    - mask padded rows via quats == 0
    #    - re-voxelize anchors
    #    - recover full time_features via choose_idx from factors[:,0]
    #    - inverse_sigmoid factors
    _sync(device)
    t0 = _now()

    # Ensure anchors on device for later ops (png decode returns CPU)
    for k, v in list(splats.items()):
        if isinstance(v, torch.Tensor) and str(v.device) != device:
            splats[k] = v.to(device)

    # mask
    mask = (splats["quats"].any(dim=1) != 0)
    for k, v in list(splats.items()):
        if k != "time_features":
            splats[k] = v[mask]

    # re-voxelize
    voxel_size = meta["anchors"]["voxel_size"]
    splats["anchors"] = torch.round(splats["anchors"] / voxel_size) * voxel_size

    # recover time_features to [N, GOP, C]
    choose_idx = splats["factors"][:, 0] > 0
    gop = meta["time_features"]["shape"][1]
    c_perframe = meta["time_features"]["shape"][2]
    full_tf = torch.zeros((len(splats["anchors"]), gop, c_perframe), device=device)
    # decoded time_features currently corresponds to choose_idx==True rows
    full_tf[choose_idx] = splats["time_features"]
    splats["time_features"] = full_tf

    # factors back to logits for internal storage (trainer does this)
    splats["factors"] = inverse_sigmoid(splats["factors"])

    _sync(device)
    timing["postprocess_sec"] = _now() - t0

    timing["decode_total_sec"] = sum(v for k, v in timing.items() if k.startswith("decode_") or k == "postprocess_sec")
    return splats, timing


# -------------------------
# Rendering (using reconstructed model)
# -------------------------

class GIFStreamEvaluator(nn.Module):
    def __init__(
        self,
        splats: Dict[str, Tensor],
        decoders: nn.ModuleDict,
        h: InferredHParams,
        device: str,
        knn: bool = False,
        n_knn: int = 6,
        packed: bool = False,
        antialiased: bool = False,
        camera_model: str = "pinhole",
    ) -> None:
        super().__init__()
        self.device = device
        self.splats = splats
        self.decoders = decoders
        self.h = h
        self.knn = knn
        self.n_knn = n_knn
        self.packed = packed
        self.antialiased = antialiased
        self.camera_model = camera_model
        self.indices: Optional[Tensor] = None

        # App embedding is NOT stored in nets.pt; if model expects it, we inject zeros.
        self.app_embed_dim = h.app_embed_dim

        # Infer mlp_color expected input dim from weights (authoritative)
        mlp_color0 = self.decoders["mlp_color"][0]
        assert isinstance(mlp_color0, torch.nn.Linear)
        self.mlp_color_in_dim = int(mlp_color0.in_features)

        # Infer how many dims are already in time_adaptive_features (anchor_features + time_features [+ view_dir])
        self.anchor_feature_dim = int(self.splats["anchor_features"].shape[1])  # should be 24
        self.c_perframe = int(self.splats["time_features"].shape[2])           # should be 4

        # If your eval implements view_adaptive, account for it here; otherwise keep 0.
        self.view_dim = 3 if self.h.view_adaptive else 0

        # Base dims in time_adaptive_features (what you feed mlp_color *before* app embedding)
        base_dim = self.anchor_feature_dim + self.c_perframe + self.view_dim

        # The remaining dims must be app embedding (or other extras). For your case, this becomes 6.
        self.app_embed_dim_needed = max(0, self.mlp_color_in_dim - base_dim)

        print(f"[Eval] mlp_color expects {self.mlp_color_in_dim} dims; "
            f"base_dim={base_dim}; app_embed_dim_needed={self.app_embed_dim_needed}")

    @torch.no_grad()
    def decoding_features(
        self,
        camtoworlds: Tensor,
        time_val: float,
        visible_anchor_mask: Tensor,
        camera_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # match trainer (eval path): no coarse-to-fine window, use exact frame
        feat_start = int(time_val * (self.h.gop - 1))
        pre = feat_start
        aft = feat_start + 1

        selected_features = self.splats["anchor_features"][visible_anchor_mask]            # [M, C]
        selected_anchors = self.splats["anchors"][visible_anchor_mask]                    # [M, 3]
        selected_scales = torch.exp(self.splats["scales"][visible_anchor_mask])           # [M, 6]
        selected_time_features = self.splats["time_features"][visible_anchor_mask][:, pre:aft].mean(dim=1) \
            if aft - pre > 1 else self.splats["time_features"][visible_anchor_mask][:, feat_start]         # [M, Cp]

        # factors: trainer eval uses fake_quantize_factors(q_aware=False) even when not compression_sim
        # Here we replicate that behavior minimally: use sigmoid(factors_logits).
        selected_factors = torch.sigmoid(self.splats["factors"][visible_anchor_mask])     # [M, 4]

        cam_pos = camtoworlds[:, :3, 3]
        view_dir = selected_anchors - cam_pos
        length = view_dir.norm(dim=1, keepdim=True).clamp_min(1e-8)
        view_dir_normalized = view_dir / length

        if self.h.view_adaptive:
            feature_view_dir = torch.cat([selected_features, view_dir_normalized], dim=1)
        else:
            feature_view_dir = selected_features

        # time positional embedding
        i = torch.ones((1), dtype=torch.float32, device=self.device)
        # time_dim must be even
        half = self.h.time_dim // 2
        time_emb = torch.cat(
            [torch.sin((self.h.phi ** n) * torch.pi * i * time_val) for n in range(half)] +
            [torch.cos((self.h.phi ** n) * torch.pi * i * time_val) for n in range(half)],
            dim=0,
        ).to(self.device)  # [time_dim]

        time_feature_factor = selected_factors[:, 0:1]
        motion_factor = selected_factors[:, 1:2]
        pruning_factor = selected_factors[:, 3:4]

        # apply pruning_factor to last 3 dims (anisotropy part) as in trainer
        selected_scales = torch.cat([selected_scales[:, :3], selected_scales[:, 3:] * pruning_factor], dim=-1)

        # Features for mlps
        time_adaptive_features = torch.cat([feature_view_dir, selected_time_features * time_feature_factor], dim=-1)
        # motion net uses (selected_features + time_features) + time_embedding
        time_adaptive_features_ = torch.cat(
            [torch.cat([selected_features, selected_time_features * time_feature_factor], dim=-1),
             time_emb.unsqueeze(0).expand(time_adaptive_features.shape[0], -1)],
            dim=-1,
        )

        k = self.h.n_offsets

        # MLP opacity
        neural_opacity = self.decoders["mlp_opacity"](time_adaptive_features)  # [M, k]
        neural_opacity = neural_opacity.view(-1, 1) * pruning_factor.view(-1, 1).expand((-1, k)).reshape((-1, 1))

        # MLP color (inject zeros app embedding if needed)
        # time_adaptive_features: [M, base_dim]
        color_in = time_adaptive_features

        if self.app_embed_dim_needed > 0:
            # If you have a real app embedding module loaded, use it; otherwise use zeros.
            if getattr(self, "app_module", None) is not None and camera_ids is not None:
                emb = self.app_module(camera_ids.to(self.device)).view(1, -1)  # [1, D]
            else:
                emb = torch.zeros((1, self.app_embed_dim_needed), device=self.device)

            emb = emb.expand(color_in.shape[0], -1)  # [M, D]
            color_in = torch.cat([color_in, emb], dim=-1)  # [M, base_dim + D]

        # Final safety check
        if color_in.shape[1] != self.mlp_color_in_dim:
            raise RuntimeError(
                f"mlp_color input dim mismatch: got {color_in.shape[1]}, "
                f"expected {self.mlp_color_in_dim}. "
                f"(base_dim={time_adaptive_features.shape[1]}, app_needed={self.app_embed_dim_needed})"
            )

        neural_colors = self.decoders["mlp_color"](color_in).view(-1, 3)

        # MLP cov
        neural_scale_rot = self.decoders["mlp_cov"](time_adaptive_features).view(-1, 7)  # [M*k, 7]

        # motion
        motion = self.decoders["mlp_motion"](time_adaptive_features_) * motion_factor  # [M, 7]

        return {
            "neural_opacity": neural_opacity,
            "neural_colors": neural_colors,
            "neural_scale_rot": neural_scale_rot,
            "motion": motion,
            "selected_scales": selected_scales,
            "selected_factors": selected_factors,
            "selected_anchors": selected_anchors,
        }

    @torch.no_grad()
    def get_neural_gaussians(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        time_val: float,
        camera_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        rasterize_mode = "antialiased" if self.antialiased else "classic"

        # visible anchors
        visible_anchor_mask = view_to_visible_anchors(
            means=self.splats["anchors"],
            quats=self.splats["quats"],
            scales=torch.exp(self.splats["scales"][:, :3]),
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.packed,
            rasterize_mode=rasterize_mode,
        )

        selected_offsets = self.splats["offsets"][visible_anchor_mask]  # [M, k, 3]

        results = self.decoding_features(camtoworlds, time_val, visible_anchor_mask, camera_ids=camera_ids)

        neural_opacity = results["neural_opacity"]
        neural_colors = results["neural_colors"]
        neural_scale_rot = results["neural_scale_rot"]
        motion = results["motion"]
        selected_scales = results["selected_scales"]
        selected_anchors = results["selected_anchors"]

        # select gaussians with positive opacity
        neural_selection_mask = (neural_opacity > 0.0).view(-1)

        # motion: anchor offset + anchor rotation
        anchor_offset = motion[:, -7:-4]
        selected_anchors = selected_anchors + anchor_offset
        anchor_rot = F.normalize(0.1 * motion[:, -4:] + torch.tensor([[1, 0, 0, 0]], device=self.device))
        anchor_R = quaternion_to_rotation_matrix(anchor_rot)

        # transform offsets by scale + anchor rotation
        M = selected_offsets.shape[0]
        k = selected_offsets.shape[1]
        offs = selected_offsets.view(-1, k, 3) * selected_scales.unsqueeze(1)[:, :, :3]
        offs = torch.bmm(offs, anchor_R.reshape((-1, 3, 3)).transpose(1, 2)).reshape((-1, 3))

        scales_rep = selected_scales.unsqueeze(1).repeat(1, k, 1).view(-1, 6)
        anchors_rep = selected_anchors.unsqueeze(1).repeat(1, k, 1).view(-1, 3)

        # apply selection
        selected_opacity = neural_opacity[neural_selection_mask].squeeze(-1)
        selected_colors = neural_colors[neural_selection_mask]
        selected_scale_rot = neural_scale_rot[neural_selection_mask]
        offs = offs[neural_selection_mask]
        scales_rep = scales_rep[neural_selection_mask]
        anchors_rep = anchors_rep[neural_selection_mask]

        # final scales/quats
        scales = scales_rep[:, 3:] * torch.sigmoid(selected_scale_rot[:, :3])
        quats = F.normalize(selected_scale_rot[:, 3:7])

        means = anchors_rep + offs

        return {
            "means": means,
            "quats": quats,
            "scales": scales,
            "opacities": selected_opacity,
            "colors": selected_colors,
        }

    @torch.no_grad()
    def render_one(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        time_val: float,
        masks: Optional[Tensor] = None,
        camera_ids: Tensor = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Returns:
          render_colors: [1, H, W, 3]
          timing: dict with breakdown incl rasterization kernel time
        """
        timing: Dict[str, float] = {}

        # build gaussians
        _sync(self.device)
        t0 = _now()
        gauss = self.get_neural_gaussians(camtoworlds, Ks, width, height, time_val, camera_ids=camera_ids)
        _sync(self.device)
        timing["neural_gaussians_sec"] = _now() - t0

        # rasterization kernel timing
        rasterize_mode = "antialiased" if self.antialiased else "classic"

        _sync(self.device)
        t0 = _now()
        colors, alphas, _info = rasterization(
            means=gauss["means"],
            quats=gauss["quats"],
            scales=gauss["scales"],
            opacities=gauss["opacities"],
            colors=gauss["colors"],
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.packed,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model=self.camera_model,
        )
        _sync(self.device)
        timing["rasterization_sec"] = _now() - t0

        if masks is not None:
            colors[~masks] = 0

        timing["render_total_sec"] = timing["neural_gaussians_sec"] + timing["rasterization_sec"]
        return colors, timing


# -------------------------
# cfg.yml auto-detection
# -------------------------

def try_load_cfg_yml_near_compress_dir(compress_dir: str) -> Optional[Dict[str, Any]]:
    # compress_dir = .../r0/compression/rank0
    # cfg.yml typically at .../r0/cfg.yml
    candidate = os.path.abspath(os.path.join(compress_dir, "..", "..", "cfg.yml"))
    if os.path.exists(candidate):
        with open(candidate, "r") as f:
            # return yaml.safe_load(f)
            return yaml.load(f, Loader=_TolerantLoader)
    return None


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compress_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--max_images", type=int, default=-1, help="If >0, render only first N images.")
    ap.add_argument("--phi", type=float, default=2.0, help="Time embedding base (default matches trainer).")
    ap.add_argument("--data_dir", type=str, default="", help="Dataset root. If empty, try reading ../cfg.yml.")
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument("--test_every", type=int, default=8)
    ap.add_argument("--normalize_world_space", action="store_true")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--camera_model", type=str, default="pinhole", choices=["pinhole", "ortho", "fisheye"])
    ap.add_argument("--packed", action="store_true")
    ap.add_argument("--antialiased", action="store_true")
    args = ap.parse_args()

    compress_dir = args.compress_dir
    out_dir = args.out_dir
    device = args.device

    _mkdir(out_dir)
    _mkdir(os.path.join(out_dir, "renders"))
    # _mkdir(os.path.join(out_dir, "stats"))

    # 1) Payload sizes
    payload = compute_payload_by_attribute(compress_dir)
    with open(os.path.join(out_dir, "payload_sizes.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # 2) Load meta + nets.pt
    meta_fp = os.path.join(compress_dir, "meta.json")
    with open(meta_fp, "r") as f:
        meta = json.load(f)

    _sync(device)
    t0 = _now()
    nets = torch.load(os.path.join(compress_dir, "nets.pt"), map_location=device, weights_only=False)
    _sync(device)
    load_nets_sec = _now() - t0

    dec_state = nets["decoders"]

    # 3) Rebuild decoders + infer hparams
    decoders = build_decoders_from_state(dec_state, device=device)
    h = infer_hparams_from_meta_and_decoders(meta, dec_state, phi_default=args.phi)

    # 4) Rebuild entropy models (needed for end2end decompress)
    entropy_models = build_entropy_models_from_nets(meta, nets, device=device)

    # 5) Decode splats with per-attribute timing
    splats, decode_timing = decode_splats_with_timing(compress_dir, device=device, entropy_models=entropy_models)
    decode_timing["load_nets_sec"] = load_nets_sec
    decode_timing["total_reconstruction_sec"] = load_nets_sec + decode_timing["decode_total_sec"]

    with open(os.path.join(out_dir, "decode_timing.json"), "w") as f:
        json.dump(decode_timing, f, indent=2)

    # 6) Dataset config
    cfg_yml = try_load_cfg_yml_near_compress_dir(compress_dir)
    if args.data_dir:
        data_dir = args.data_dir
        data_factor = args.data_factor
        test_every = args.test_every
        normalize = args.normalize_world_space
        start_frame = args.start_frame
        camera_model = args.camera_model
    elif cfg_yml is not None:
        data_dir = cfg_yml.get("data_dir", "")
        data_factor = int(cfg_yml.get("data_factor", args.data_factor))
        test_every = int(cfg_yml.get("test_every", args.test_every))
        normalize = bool(cfg_yml.get("normalize_world_space", False))
        start_frame = int(cfg_yml.get("start_frame", 0))
        camera_model = str(cfg_yml.get("camera_model", args.camera_model))
    else:
        raise ValueError("Could not find cfg.yml near compress_dir and --data_dir not provided. Please pass --data_dir.")

    knn = bool(cfg_yml.get("knn", False)) if cfg_yml else False
    n_knn = int(cfg_yml.get("n_knn", 6)) if cfg_yml else 6

    # 7) Load dataset
    parser = Parser(
        data_dir=data_dir,
        factor=data_factor,
        normalize=normalize,
        test_every=test_every,
        first_frame=start_frame,
    )
    dataset = Dataset(
        parser,
        split=args.split,
        patch_size=None,
        load_depths=False,
        test_set=cfg_yml.get("test_set", None) if cfg_yml else None,
        remove_set=cfg_yml.get("remove_set", None) if cfg_yml else None,
        GOP_size=h.gop,
        start_frame=start_frame,
    )

    # 8) Evaluator
    evaluator = GIFStreamEvaluator(
        splats=splats,
        decoders=decoders,
        h=h,
        device=device,
        knn=knn,
        n_knn=n_knn,
        packed=args.packed,
        antialiased=args.antialiased,
        camera_model=camera_model,
    ).to(device)
    evaluator.eval()

    # -------------------------
    # Quality metrics (match trainer)
    # -------------------------
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # match trainer default (alex). Optionally read cfg.yml if you want.
    lpips_net = "vgg"
    # if cfg_yml is not None:
    #     lpips_net = str(cfg_yml.get("lpips_net", "alex"))

    if lpips_net == "alex":
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False).to(device)
    elif lpips_net == "vgg":
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)
    else:
        raise ValueError(f"Unknown lpips_net: {lpips_net}")

    metrics = {"psnr": [], "ssim": [], "lpips": []}

    # 9) Render loop with raster timing
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    render_stats = {
        "compress_dir": compress_dir,
        "out_dir": out_dir,
        "device": device,
        "split": args.split,
        "num_images": 0,
        "total_render_sec": 0.0,
        "total_rasterization_sec": 0.0,
        "avg_render_sec": None,
        "avg_rasterization_sec": None,
        "per_image": [],
        "inferred_hparams": h.__dict__,
    }

    for i, data in enumerate(loader):
        if args.max_images > 0 and i >= args.max_images:
            break

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None
        pixels = data["image"].to(device) / 255.0  # only for reference; not used
        H, W = pixels.shape[1:3]
        time_val = float(data["time"])
        camera_ids = data["camera_id"].to(device)

        with torch.inference_mode():
            colors, timing = evaluator.render_one(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=W,
                height=H,
                time_val=time_val,
                masks=masks,
                camera_ids=camera_ids,
            )

        colors = torch.clamp(colors, 0.0, 1.0)

        # Match trainer tensor layout: [1, H, W, 3] -> [1, 3, H, W]
        pixels_p = pixels.permute(0, 3, 1, 2)
        colors_p = colors.permute(0, 3, 1, 2)

        metrics["psnr"].append(psnr(colors_p, pixels_p).detach().cpu())
        metrics["ssim"].append(ssim(colors_p, pixels_p).detach().cpu())
        metrics["lpips"].append(lpips(colors_p, pixels_p).detach().cpu())

        img = (colors.squeeze(0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(out_dir, "renders", f"{args.split}_{i:04d}.png"), img)

        render_stats["num_images"] += 1
        render_stats["total_render_sec"] += float(timing["render_total_sec"])
        render_stats["total_rasterization_sec"] += float(timing["rasterization_sec"])
        render_stats["per_image"].append(
            {
                "idx": i,
                "time": time_val,
                "render_total_sec": float(timing["render_total_sec"]),
                "neural_gaussians_sec": float(timing["neural_gaussians_sec"]),
                "rasterization_sec": float(timing["rasterization_sec"]),
                "H": int(H),
                "W": int(W),
            }
        )

    if render_stats["num_images"] > 0:
        n = render_stats["num_images"]
        render_stats["avg_render_sec"] = render_stats["total_render_sec"] / n
        render_stats["avg_rasterization_sec"] = render_stats["total_rasterization_sec"] / n

    with open(os.path.join(out_dir, "render_timing.json"), "w") as f:
        json.dump(render_stats, f, indent=2)

    print(f"[OK] Wrote payload + timings + renders to: {out_dir}")
    print(f"     payload_sizes.json, decode_timing.json, render_timing.json, renders/*.png")


    quality = {}
    for k, v in metrics.items():
        if len(v) == 0:
            quality[k] = None
        else:
            quality[k] = float(torch.stack(v).mean().item())

    quality.update(
        {
            "num_images": int(render_stats["num_images"]),
            "split": args.split,
            "lpips_net": lpips_net,
        }
    )

    with open(os.path.join(out_dir, "quality.json"), "w") as f:
        json.dump(quality, f, indent=2)

    print(f"[OK] Wrote quality.json to: {os.path.join(out_dir, 'quality.json')}")

if __name__ == "__main__":
    main()