"""
python examples/eval_2dcodec_gifstream.py \
    --compress_dir results_ori_codec/flame_steak/GOP_0/r0/compression_2dcodec/qp28 \
    --out_dir results_ori_codec/flame_steak/GOP_0/r0/eval_2dcodec/qp28
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets.GIFStream_new import Dataset, Parser
from gsplat.rendering import rasterization, view_to_visible_anchors
from utils import find_k_neighbors


class _TolerantLoader(yaml.SafeLoader):
    pass


def _construct_unknown(loader, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


_TolerantLoader.add_constructor(None, _construct_unknown)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device=device)


def _now() -> float:
    return time.time()


def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


def quaternion_to_rotation_matrix(quaternion: Tensor) -> Tensor:
    if quaternion.dim() == 1:
        quaternion = quaternion.unsqueeze(0)
    w, x, y, z = quaternion.unbind(dim=-1)
    B = quaternion.size(0)
    rotation_matrix = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
            2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(B, 3, 3)
    return rotation_matrix


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def try_load_cfg_yml_near_compress_dir(compress_dir: str) -> Optional[Dict[str, Any]]:
    # For .../r0/compression_2dcodec/qp28 -> cfg at .../r0/cfg.yml
    candidate = os.path.abspath(os.path.join(compress_dir, "..", "..", "cfg.yml"))
    if not os.path.exists(candidate):
        return None

    with open(candidate, "r") as f:
        raw = f.read()

    raw = _strip_markdown_fences(raw)
    if not raw.strip():
        return None

    try:
        return yaml.load(raw, Loader=_TolerantLoader)
    except yaml.YAMLError as e:
        print(f"[WARN] Failed to parse cfg.yml at {candidate}: {e}")
        return None


# -----------------------------------------------------------------------------
# Side-info modules
# -----------------------------------------------------------------------------

def build_decoders_from_state(dec_state: Dict[str, Tensor], device: str) -> nn.ModuleDict:
    feat_dim = int(dec_state["mlp_opacity.0.weight"].shape[0])

    in_opa = int(dec_state["mlp_opacity.0.weight"].shape[1])
    out_opa = int(dec_state["mlp_opacity.2.weight"].shape[0])
    mlp_opacity = nn.Sequential(
        nn.Linear(in_opa, feat_dim),
        nn.ReLU(True),
        nn.Linear(feat_dim, out_opa),
        nn.Tanh(),
    )

    in_cov = int(dec_state["mlp_cov.0.weight"].shape[1])
    out_cov = int(dec_state["mlp_cov.2.weight"].shape[0])
    mlp_cov = nn.Sequential(
        nn.Linear(in_cov, feat_dim),
        nn.ReLU(True),
        nn.Linear(feat_dim, out_cov),
    )

    in_col = int(dec_state["mlp_color.0.weight"].shape[1])
    out_col = int(dec_state["mlp_color.2.weight"].shape[0])
    mlp_color = nn.Sequential(
        nn.Linear(in_col, feat_dim),
        nn.ReLU(True),
        nn.Linear(feat_dim, out_col),
        nn.Sigmoid(),
    )

    in_mot = int(dec_state["mlp_motion.0.weight"].shape[1])
    out_mot = int(dec_state["mlp_motion.2.weight"].shape[0])
    mlp_motion = nn.Sequential(
        nn.Linear(in_mot, feat_dim),
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


class AppEmbeddingTable(nn.Module):
    def __init__(self, weight: Tensor):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, camera_ids: Tensor) -> Tensor:
        camera_ids = camera_ids.long().view(-1)
        return self.weight[camera_ids]


# -----------------------------------------------------------------------------
# 2D-codec tensor decode
# -----------------------------------------------------------------------------

def _read_png_keep_channels(fp: str) -> np.ndarray:
    arr = imageio.imread(fp)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr



def _decode_png_tensor_from_meta(compress_dir: str, meta_item: Dict[str, Any]) -> np.ndarray:
    shape = tuple(meta_item["shape"])
    bitdepth = int(meta_item["bitdepth"])
    mins = np.asarray(meta_item["mins"], dtype=np.float32)
    maxs = np.asarray(meta_item["maxs"], dtype=np.float32)
    chunk_sizes = list(meta_item["chunk_sizes"])
    files = list(meta_item["files"])

    chunks = []
    fi = 0
    for csz in chunk_sizes:
        if bitdepth <= 8:
            q = _read_png_keep_channels(os.path.join(compress_dir, files[fi]))
            fi += 1
        else:
            lo = _read_png_keep_channels(os.path.join(compress_dir, files[fi]))
            hi = _read_png_keep_channels(os.path.join(compress_dir, files[fi + 1]))
            fi += 2
            q = (hi.astype(np.uint16) << 8) + lo.astype(np.uint16)
        q = q[..., :csz]
        chunks.append(q)

    q_all = np.concatenate(chunks, axis=-1)
    q_all = q_all.astype(np.float32) / float((1 << bitdepth) - 1)
    denom = np.maximum(maxs - mins, 1e-8)
    grid = q_all * denom.reshape(1, 1, -1) + mins.reshape(1, 1, -1)
    arr = grid.reshape(shape).astype(np.float32)
    return arr



def _decode_npz_tensor_from_meta(compress_dir: str, meta_item: Dict[str, Any]) -> np.ndarray:
    fp = os.path.join(compress_dir, meta_item["files"][0])
    arr = np.load(fp)["arr"]
    return arr.astype(np.float32)



def _decode_hevc_tensor_from_meta(
    compress_dir: str,
    meta_item: Dict[str, Any],
    ffmpeg_bin: str,
) -> np.ndarray:
    shape = tuple(meta_item["shape"])  # [N, T, P]
    bitdepth = int(meta_item["bitdepth"])
    mins = np.asarray(meta_item["mins"], dtype=np.float32)
    maxs = np.asarray(meta_item["maxs"], dtype=np.float32)
    chunk_sizes = list(meta_item["chunk_sizes"])
    files = list(meta_item["files"])

    decoded_chunks = []
    for csz, fn in zip(chunk_sizes, files):
        video_fp = os.path.join(compress_dir, fn)
        with tempfile.TemporaryDirectory(prefix="gifstream_2dcodec_decode_") as tmpdir:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                video_fp,
                os.path.join(tmpdir, "frame_%05d.png"),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            frame_paths = sorted(
                [os.path.join(tmpdir, x) for x in os.listdir(tmpdir) if x.endswith(".png")]
            )
            if len(frame_paths) == 0:
                raise RuntimeError(f"No frames decoded from HEVC file: {video_fp}")
            frames = []
            for fp in frame_paths:
                fr = imageio.imread(fp)
                if fr.ndim == 2:
                    fr = fr[..., None]
                frames.append(fr.astype(np.uint8))
            # [T, H, W, C] -> [H, W, T, C]
            arr = np.stack(frames, axis=0)
            arr = np.transpose(arr, (1, 2, 0, 3))
            decoded_chunks.append(arr[..., :csz])

    q_all = np.concatenate(decoded_chunks, axis=-1)
    q_all = q_all.astype(np.float32) / float((1 << bitdepth) - 1)
    denom = np.maximum(maxs - mins, 1e-8)
    grid = q_all * denom.reshape(1, 1, 1, -1) + mins.reshape(1, 1, 1, -1)

    N, T, P = shape
    H = W = int(round(math.sqrt(N)))
    if H * W != N:
        raise ValueError(f"Expected square padded anchor count for time_features, got N={N}")
    arr = grid.reshape(H * W, T, P).astype(np.float32)
    return arr



def decode_2dcodec_splats(
    compress_dir: str,
    device: str,
    ffmpeg_bin: str,
) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
    meta_fp = os.path.join(compress_dir, "meta.json")
    with open(meta_fp, "r") as f:
        meta = json.load(f)

    timing: Dict[str, float] = {}
    splats_np: Dict[str, np.ndarray] = {}

    # PNG tensors
    for name in ["anchors", "scales", "offsets", "factors", "anchor_features"]:
        _sync(device)
        t0 = _now()
        splats_np[name] = _decode_png_tensor_from_meta(compress_dir, meta[name])
        _sync(device)
        timing[f"decode_{name}_sec"] = _now() - t0

    # HEVC time feature stream
    _sync(device)
    t0 = _now()
    splats_np["time_features"] = _decode_hevc_tensor_from_meta(compress_dir, meta["time_features"], ffmpeg_bin=ffmpeg_bin)
    _sync(device)
    timing["decode_time_features_sec"] = _now() - t0

    # NPZ side tensors
    for name in ["quats", "opacities"]:
        _sync(device)
        t0 = _now()
        splats_np[name] = _decode_npz_tensor_from_meta(compress_dir, meta[name])
        _sync(device)
        timing[f"decode_{name}_sec"] = _now() - t0

    # Remove padded anchors using the same convention as posthoc compressor: padded quats are zeros.
    _sync(device)
    t0 = _now()
    quats = splats_np["quats"]
    mask = (np.any(quats != 0, axis=1))
    splats = {}
    for k, v in splats_np.items():
        splats[k] = torch.from_numpy(v[mask]).to(device)
    _sync(device)
    timing["postprocess_sec"] = _now() - t0

    timing["decode_total_sec"] = sum(v for k, v in timing.items() if k.startswith("decode_") or k == "postprocess_sec")
    return splats, timing


# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------

class GIFStream2DCodecEvaluator(nn.Module):
    def __init__(
        self,
        splats: Dict[str, Tensor],
        decoders: nn.ModuleDict,
        cfg: Dict[str, Any],
        device: str,
        app_module: Optional[AppEmbeddingTable] = None,
    ) -> None:
        super().__init__()
        self.splats = splats
        self.decoders = decoders
        self.cfg = cfg
        self.device = device
        self.app_module = app_module

        self.GOP_size = int(cfg.get("GOP_size", splats["time_features"].shape[1]))
        self.c_perframe = int(cfg.get("c_perframe", splats["time_features"].shape[2]))
        self.anchor_feature_dim = int(cfg.get("anchor_feature_dim", splats["anchor_features"].shape[1]))
        self.n_offsets = int(cfg.get("n_offsets", splats["offsets"].shape[1]))
        self.time_dim = int(cfg.get("time_dim", 16))
        self.phi = float(cfg.get("phi", 2.0))
        self.view_adaptive = bool(cfg.get("view_adaptive", False))
        self.knn = bool(cfg.get("knn", False))
        self.n_knn = int(cfg.get("n_knn", 6))
        self.packed = bool(cfg.get("packed", False))
        self.antialiased = bool(cfg.get("antialiased", False))
        self.camera_model = str(cfg.get("camera_model", "pinhole"))

        self.indices: Optional[Tensor] = None
        if self.knn:
            _, self.indices = find_k_neighbors(self.splats["anchors"], self.n_knn)
            self.indices = self.indices.to(device)

    @torch.no_grad()
    def decoding_features(
        self,
        camtoworlds: Tensor,
        time_val: float,
        visible_anchor_mask: Tensor,
        camera_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        feat_start = int(time_val * (self.GOP_size - 1))

        selected_features = self.splats["anchor_features"][visible_anchor_mask]
        selected_anchors = self.splats["anchors"][visible_anchor_mask]
        selected_scales = torch.exp(self.splats["scales"][visible_anchor_mask])
        selected_time_features = self.splats["time_features"][visible_anchor_mask][:, feat_start]
        # In 2D-codec posthoc compression, factors are already quantized probabilities, not logits.
        selected_factors = self.splats["factors"][visible_anchor_mask]

        if self.knn:
            selected_indices = self.indices[visible_anchor_mask].reshape(-1)
            knn_features = self.splats["anchor_features"][selected_indices].reshape(
                -1, self.n_knn, self.anchor_feature_dim
            ).mean(dim=1)
            knn_time_features = self.splats["time_features"][selected_indices][:, feat_start].reshape(
                -1, self.n_knn, self.c_perframe
            ).mean(dim=1)

        cam_pos = camtoworlds[:, :3, 3]
        view_dir = selected_anchors - cam_pos
        length = view_dir.norm(dim=1, keepdim=True).clamp_min(1e-8)
        view_dir_normalized = view_dir / length

        if self.view_adaptive:
            feature_view_dir = torch.cat([selected_features, view_dir_normalized], dim=1)
        else:
            feature_view_dir = selected_features

        i = torch.ones((1,), dtype=torch.float32, device=self.device)
        time_embedding = torch.cat(
            [torch.sin((self.phi ** n) * torch.pi * i * time_val) for n in range(self.time_dim // 2)]
            + [torch.cos((self.phi ** n) * torch.pi * i * time_val) for n in range(self.time_dim // 2)],
            dim=0,
        )

        time_feature_factor = selected_factors[:, 0].unsqueeze(-1)
        motion_factor = selected_factors[:, 1].unsqueeze(-1)
        knn_factor = selected_factors[:, 2].unsqueeze(-1)
        pruning_factor = selected_factors[:, 3].unsqueeze(-1)

        selected_scales = torch.cat(
            [selected_scales[:, :3], selected_scales[:, 3:] * pruning_factor], dim=-1
        )

        time_adaptive_features = torch.cat(
            [feature_view_dir, selected_time_features * time_feature_factor], dim=-1
        )
        if self.knn:
            time_adaptive_features_ = knn_factor * torch.cat(
                [selected_features, selected_time_features * time_feature_factor], dim=-1
            ) + (1 - knn_factor) * torch.cat([knn_features, knn_time_features], dim=-1)
        else:
            time_adaptive_features_ = torch.cat(
                [selected_features, selected_time_features * time_feature_factor], dim=-1
            )
        time_adaptive_features_ = torch.cat(
            [time_adaptive_features_, time_embedding.unsqueeze(0).expand(time_adaptive_features.shape[0], -1)],
            dim=1,
        )

        k = self.n_offsets

        neural_opacity = self.decoders["mlp_opacity"](time_adaptive_features)
        neural_opacity = neural_opacity.view(-1, 1) * pruning_factor.view(-1, 1).expand((-1, k)).reshape((-1, 1))

        if self.app_module is not None and camera_ids is not None:
            app_emb = self.app_module(camera_ids.to(self.device)).view((1, -1)).expand(time_adaptive_features.shape[0], -1)
            color_in = torch.cat([time_adaptive_features, app_emb], dim=-1)
        else:
            color_in = time_adaptive_features
        neural_colors = self.decoders["mlp_color"](color_in).view(-1, 3)

        neural_scale_rot = self.decoders["mlp_cov"](time_adaptive_features).view(-1, 7)
        motion = self.decoders["mlp_motion"](time_adaptive_features_) * motion_factor

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

        selected_offsets = self.splats["offsets"][visible_anchor_mask]
        results = self.decoding_features(camtoworlds, time_val, visible_anchor_mask, camera_ids=camera_ids)

        neural_opacity = results["neural_opacity"]
        neural_colors = results["neural_colors"]
        neural_scale_rot = results["neural_scale_rot"]
        motion = results["motion"]
        selected_scales = results["selected_scales"]
        selected_anchors = results["selected_anchors"]

        neural_selection_mask = (neural_opacity > 0.0).view(-1)

        anchor_offset = motion[:, -7:-4]
        selected_anchors = selected_anchors + anchor_offset
        anchor_rot = F.normalize(0.1 * motion[:, -4:] + torch.tensor([[1, 0, 0, 0]], device=self.device))
        anchor_rotation = quaternion_to_rotation_matrix(anchor_rot)

        selected_offsets = torch.bmm(
            selected_offsets.view(-1, self.n_offsets, 3) * selected_scales.unsqueeze(1)[:, :, :3],
            anchor_rotation.reshape((-1, 3, 3)).transpose(1, 2),
        ).reshape((-1, 3))

        scales_repeated = selected_scales.unsqueeze(1).repeat(1, self.n_offsets, 1).view(-1, 6)
        anchors_repeated = selected_anchors.unsqueeze(1).repeat(1, self.n_offsets, 1).view(-1, 3)

        selected_opacity = neural_opacity[neural_selection_mask].squeeze(-1)
        selected_colors = neural_colors[neural_selection_mask]
        selected_scale_rot = neural_scale_rot[neural_selection_mask]
        selected_offsets = selected_offsets[neural_selection_mask]
        scales_repeated = scales_repeated[neural_selection_mask]
        anchors_repeated = anchors_repeated[neural_selection_mask]

        scales = scales_repeated[:, 3:] * torch.sigmoid(selected_scale_rot[:, :3])
        rotation = F.normalize(selected_scale_rot[:, 3:7])
        means = anchors_repeated + selected_offsets

        return {
            "means": means,
            "colors": selected_colors,
            "opacities": selected_opacity,
            "scales": scales,
            "quats": rotation,
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
        camera_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        timing: Dict[str, float] = {}

        _sync(self.device)
        t0 = _now()
        gauss = self.get_neural_gaussians(camtoworlds, Ks, width, height, time_val, camera_ids=camera_ids)
        _sync(self.device)
        timing["neural_gaussians_sec"] = _now() - t0

        rasterize_mode = "antialiased" if self.antialiased else "classic"
        _sync(self.device)
        t0 = _now()
        colors, alphas, _ = rasterization(
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate GIFStream 2D-codec (PNG+HEVC) compressed outputs.")
    ap.add_argument("--compress_dir", type=str, required=True, help="Directory like .../compression_2dcodec/qp28")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for renders + stats")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val"])
    ap.add_argument("--max_images", type=int, default=-1)
    ap.add_argument("--ffmpeg_bin", type=str, default="ffmpeg")
    ap.add_argument("--lpips_net", type=str, default="", choices=["", "alex", "vgg"])
    return ap.parse_args()



def main() -> None:
    args = parse_args()
    compress_dir = args.compress_dir
    out_dir = args.out_dir
    device = args.device

    _mkdir(out_dir)
    _mkdir(os.path.join(out_dir, "renders"))
    _mkdir(os.path.join(out_dir, "stats"))

    cfg_yml = try_load_cfg_yml_near_compress_dir(compress_dir)
    if cfg_yml is None:
        raise FileNotFoundError(f"Could not locate cfg.yml near {compress_dir}")

    # Load side info
    _sync(device)
    t0 = _now()
    nets = torch.load(os.path.join(compress_dir, "nets.pt"), map_location=device, weights_only=False)
    _sync(device)
    load_nets_sec = _now() - t0

    decoders = build_decoders_from_state(nets["decoders"], device=device)

    app_module = None
    if nets.get("app_module", None) is not None:
        app_state = nets["app_module"]
        weight_key = None
        for k in app_state.keys():
            if k.endswith("weight"):
                weight_key = k
                break
        if weight_key is not None:
            app_module = AppEmbeddingTable(app_state[weight_key].to(device)).to(device)
            app_module.eval()

    # Decode all transmitted tensors
    splats, decode_timing = decode_2dcodec_splats(compress_dir, device=device, ffmpeg_bin=args.ffmpeg_bin)
    decode_timing["load_nets_sec"] = load_nets_sec
    decode_timing["total_reconstruction_sec"] = load_nets_sec + decode_timing["decode_total_sec"]
    with open(os.path.join(out_dir, "decode_timing.json"), "w") as f:
        json.dump(decode_timing, f, indent=2)

    # Dataset / parser config from original cfg.yml
    parser = Parser(
        data_dir=cfg_yml["data_dir"],
        factor=int(cfg_yml.get("data_factor", 4)),
        normalize=bool(cfg_yml.get("normalize_world_space", False)),
        test_every=int(cfg_yml.get("test_every", 8)),
        first_frame=int(cfg_yml.get("start_frame", 0)),
    )
    dataset = Dataset(
        parser,
        split=args.split,
        patch_size=None,
        load_depths=False,
        test_set=cfg_yml.get("test_set", None),
        remove_set=cfg_yml.get("remove_set", None),
        GOP_size=int(cfg_yml.get("GOP_size", splats["time_features"].shape[1])),
        start_frame=int(cfg_yml.get("start_frame", 0)),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    evaluator = GIFStream2DCodecEvaluator(
        splats=splats,
        decoders=decoders,
        cfg=cfg_yml,
        device=device,
        app_module=app_module,
    ).to(device)
    evaluator.eval()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_net = args.lpips_net if args.lpips_net else str(cfg_yml.get("lpips_net", "alex"))
    if lpips_net not in ["alex", "vgg"]:
        raise ValueError(f"Unsupported LPIPS net: {lpips_net}")
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net, normalize=False).to(device)

    metrics = defaultdict(list)
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
    }

    for i, data in enumerate(loader):
        if args.max_images > 0 and i >= args.max_images:
            break

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        masks = data["mask"].to(device) if "mask" in data else None
        camera_ids = data["camera_id"].to(device)
        H, W = pixels.shape[1:3]
        time_val = float(data["time"])

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

        img = (colors.squeeze(0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        imageio.imwrite(os.path.join(out_dir, "renders", f"{args.split}_{i:04d}.png"), img)

        pixels_p = pixels.permute(0, 3, 1, 2)
        colors_p = colors.permute(0, 3, 1, 2)
        metrics["psnr"].append(psnr(colors_p, pixels_p))
        metrics["ssim"].append(ssim(colors_p, pixels_p))
        metrics["lpips"].append(lpips(colors_p, pixels_p))

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

    quality = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
    quality["num_images"] = int(render_stats["num_images"])
    quality["lpips_net"] = lpips_net
    quality["avg_render_sec"] = render_stats["avg_render_sec"]
    quality["avg_rasterization_sec"] = render_stats["avg_rasterization_sec"]

    payload_json = os.path.join(compress_dir, "payload_sizes.json")
    if os.path.exists(payload_json):
        with open(payload_json, "r") as f:
            payload = json.load(f)
        quality["payload_total_MB"] = payload.get("total_MB", None)
        quality["payload_total_bytes"] = payload.get("total_bytes", None)

    with open(os.path.join(out_dir, "quality.json"), "w") as f:
        json.dump(quality, f, indent=2)

    print(f"[OK] 2D-codec evaluation written to: {out_dir}")
    print(f"      PSNR  = {quality['psnr']:.4f}")
    print(f"      SSIM  = {quality['ssim']:.6f}")
    print(f"      LPIPS = {quality['lpips']:.6f}")
    if "payload_total_MB" in quality and quality["payload_total_MB"] is not None:
        print(f"      payload = {quality['payload_total_MB']:.3f} MB")


if __name__ == "__main__":
    main()
