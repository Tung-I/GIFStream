"""
python examples/compress_2dcodec_gifstream.py \
  --ckpt results_ori_codec/flame_steak/GOP_0/r0/ckpts/ckpt_29999_rank0.pt \
  --out_dir results_ori_codec/flame_steak/GOP_0/r0/compression_2dcodec/qp36 \
  --qp 36
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from gsplat.compression.sort import sort_anchors
from gsplat.compression_simulation.ops import fake_quantize_factors


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _bytes_to_mb(x: int) -> float:
    return float(x) / (1024.0 * 1024.0)


@dataclass
class CodecConfig:
    hevc_qp: int = 28
    video_fps: int = 30
    time_gate_eps: float = 0.0
    prune_gate_eps: float = 0.0
    ffmpeg_bin: str = "ffmpeg"


# -----------------------------------------------------------------------------
# Tensor preprocessing
# -----------------------------------------------------------------------------

def _to_cpu_splats(splats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in splats.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().clone()
        else:
            raise TypeError(f"Unsupported non-tensor field in splats: {k}")
    return out


def _pad_to_square(splats: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """
    Pad all per-anchor tensors to ceil(sqrt(N))^2 anchors, matching the end2end path.
    Returns (padded_splats, n_sidelen, pad_count).
    """
    n = int(splats["anchors"].shape[0])
    n_sidelen = int(math.ceil(n ** 0.5))
    n_square = n_sidelen * n_sidelen
    pad_count = n_square - n
    if pad_count == 0:
        return splats, n_sidelen, 0

    padded: Dict[str, torch.Tensor] = {}
    for k, v in splats.items():
        pad_shape = (pad_count,) + tuple(v.shape[1:])
        pad = torch.zeros(pad_shape, dtype=v.dtype)
        padded[k] = torch.cat([v, pad], dim=0)
    return padded, n_sidelen, pad_count



def prepare_splats_for_2dcodec(
    raw_splats: Dict[str, torch.Tensor],
    cfg: CodecConfig,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Prepare tensors for the simple codec baseline described in the supplement.

    Steps:
      1) normalize quats
      2) convert factor logits -> quantized probabilities using fake_quantize_factors(..., q_aware=False)
      3) prune anchors using the pruning gate
      4) zero time_features for inactive time-stream anchors
      5) pad to square and sort anchors for 2D packing
    """
    splats = _to_cpu_splats(raw_splats)
    splats["quats"] = F.normalize(splats["quats"], dim=-1)

    # Match the official eval path more closely than plain sigmoid.
    factors_prob = fake_quantize_factors(splats["factors"], q_aware=False).detach().cpu()
    splats["factors"] = factors_prob

    pruning_mask = factors_prob[:, 3] > cfg.prune_gate_eps
    if pruning_mask.sum().item() == 0:
        raise RuntimeError(
            "All anchors were pruned by the selected threshold. "
            "Try lowering --prune_gate_eps."
        )

    for k in list(splats.keys()):
        splats[k] = splats[k][pruning_mask]

    time_mask = splats["factors"][:, 0] > cfg.time_gate_eps
    splats["time_features"] = splats["time_features"].clone()
    splats["time_features"][~time_mask] = 0

    splats, n_sidelen, pad_count = _pad_to_square(splats)
    splats = sort_anchors(splats)

    meta_extra = {
        "n_sidelen": n_sidelen,
        "pad_count": pad_count,
        "num_anchors_after_prune": int(pruning_mask.sum().item()),
        "num_active_time_anchors": int(time_mask.sum().item()),
        "time_gate_eps": cfg.time_gate_eps,
        "prune_gate_eps": cfg.prune_gate_eps,
    }
    return splats, meta_extra


# -----------------------------------------------------------------------------
# Generic quantized PNG helpers
# -----------------------------------------------------------------------------

def _quantize_grid(grid: np.ndarray, bitdepth: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = grid.min(axis=(0, 1))
    maxs = grid.max(axis=(0, 1))
    denom = np.maximum(maxs - mins, 1e-8)
    norm = (grid - mins) / denom
    q = np.round(norm * ((1 << bitdepth) - 1))
    if bitdepth <= 8:
        q = q.astype(np.uint8)
    else:
        q = q.astype(np.uint16)
    return q, mins.astype(np.float32), maxs.astype(np.float32)



def _compress_tensor_to_png_chunks(
    out_dir: str,
    stem: str,
    tensor: torch.Tensor,
    n_sidelen: int,
    bitdepth: int,
) -> Dict[str, Any]:
    """
    Pack [N, ...] -> [H, W, Cflat], then split channels into <=3 channel PNG groups.
    For 16-bit, each group is split into low/high-byte PNGs.
    """
    arr = tensor.detach().cpu().numpy()
    shape = list(arr.shape)
    dtype = str(arr.dtype)
    flat = arr.reshape(n_sidelen, n_sidelen, -1)
    q, mins, maxs = _quantize_grid(flat, bitdepth)

    chunk_sizes: List[int] = []
    files: List[str] = []
    cflat = q.shape[-1]
    n_chunks = math.ceil(cflat / 3)

    for idx in range(n_chunks):
        c0 = idx * 3
        c1 = min((idx + 1) * 3, cflat)
        chunk = q[:, :, c0:c1]
        chunk_sizes.append(c1 - c0)
        if bitdepth <= 8:
            fp = os.path.join(out_dir, f"{stem}_{idx:03d}.png")
            imageio.imwrite(fp, chunk.squeeze(-1) if chunk.shape[-1] == 1 else chunk)
            files.append(os.path.basename(fp))
        else:
            lo = (chunk & 0xFF).astype(np.uint8)
            hi = ((chunk >> 8) & 0xFF).astype(np.uint8)
            fp_lo = os.path.join(out_dir, f"{stem}_{idx:03d}_l.png")
            fp_hi = os.path.join(out_dir, f"{stem}_{idx:03d}_u.png")
            imageio.imwrite(fp_lo, lo.squeeze(-1) if lo.shape[-1] == 1 else lo)
            imageio.imwrite(fp_hi, hi.squeeze(-1) if hi.shape[-1] == 1 else hi)
            files.extend([os.path.basename(fp_lo), os.path.basename(fp_hi)])

    return {
        "shape": shape,
        "dtype": dtype,
        "bitdepth": bitdepth,
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "chunk_sizes": chunk_sizes,
        "files": files,
    }


# -----------------------------------------------------------------------------
# HEVC helper for time_features
# -----------------------------------------------------------------------------

def _write_frame_png(frame: np.ndarray, path: str) -> None:
    if frame.ndim == 2:
        imageio.imwrite(path, frame)
    else:
        imageio.imwrite(path, frame)



def _compress_time_features_hevc(
    out_dir: str,
    stem: str,
    tensor: torch.Tensor,
    n_sidelen: int,
    bitdepth: int,
    video_fps: int,
    hevc_qp: int,
    ffmpeg_bin: str,
) -> Dict[str, Any]:
    """
    tensor: [N, T, P]
    Reshape each time index into an HxW image, split channels into <=3 channel chunks,
    and compress each chunk as an HEVC video.
    """
    arr = tensor.detach().cpu().numpy()  # [N, T, P]
    shape = list(arr.shape)
    dtype = str(arr.dtype)
    n, tlen, p = arr.shape
    flat = arr.reshape(n_sidelen, n_sidelen, tlen, p)

    mins = flat.min(axis=(0, 1, 2))
    maxs = flat.max(axis=(0, 1, 2))
    denom = np.maximum(maxs - mins, 1e-8)
    norm = (flat - mins.reshape(1, 1, 1, -1)) / denom.reshape(1, 1, 1, -1)
    q = np.round(norm * ((1 << bitdepth) - 1)).astype(np.uint8)

    files: List[str] = []
    chunk_sizes: List[int] = []
    n_chunks = math.ceil(p / 3)

    for idx in range(n_chunks):
        c0 = idx * 3
        c1 = min((idx + 1) * 3, p)
        chunk_sizes.append(c1 - c0)

        with tempfile.TemporaryDirectory(prefix=f"{stem}_{idx:03d}_", dir=out_dir) as tmpdir:
            for t in range(tlen):
                frame = q[:, :, t, c0:c1]
                if frame.shape[-1] == 1:
                    # Store as RGB with replicated channels to avoid gray-specific codec quirks.
                    frame = np.repeat(frame, 3, axis=-1)
                elif frame.shape[-1] == 2:
                    pad = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=frame.dtype)
                    frame = np.concatenate([frame, pad], axis=-1)
                fp = os.path.join(tmpdir, f"frame_{t:05d}.png")
                _write_frame_png(frame, fp)

            out_mp4 = os.path.join(out_dir, f"{stem}_{idx:03d}.mp4")
            cmd = [
                ffmpeg_bin,
                "-y",
                "-framerate",
                str(video_fps),
                "-i",
                os.path.join(tmpdir, "frame_%05d.png"),
                "-c:v",
                "libx265",
                "-pix_fmt",
                "yuv444p",
                "-x265-params",
                f"qp={hevc_qp}",
                out_mp4,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            files.append(os.path.basename(out_mp4))

    return {
        "shape": shape,
        "dtype": dtype,
        "bitdepth": bitdepth,
        "mins": mins.astype(np.float32).tolist(),
        "maxs": maxs.astype(np.float32).tolist(),
        "chunk_sizes": chunk_sizes,
        "files": files,
        "fps": video_fps,
        "qp": hevc_qp,
    }


# -----------------------------------------------------------------------------
# Lossless side tensors
# -----------------------------------------------------------------------------

def _save_npz(out_dir: str, stem: str, tensor: torch.Tensor) -> Dict[str, Any]:
    fp = os.path.join(out_dir, f"{stem}.npz")
    np.savez_compressed(fp, arr=tensor.detach().cpu().numpy())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.detach().cpu().numpy().dtype),
        "files": [os.path.basename(fp)],
    }


# -----------------------------------------------------------------------------
# Payload accounting
# -----------------------------------------------------------------------------

def summarize_payload(out_dir: str) -> Dict[str, Any]:
    """
    Summarize payload sizes by attribute.

    Important: keep mapping entries as *patterns or filenames*, not pre-expanded paths.
    Otherwise relative paths returned by glob can accidentally get joined with out_dir twice.
    """
    mapping = {
        "meta": ["meta.json"],
        "nets": ["nets.pt"],
        "anchors": ["anchors_*.png"],
        "scales": ["scales_*.png"],
        "offsets": ["offsets_*.png"],
        "factors": ["factors_*.png"],
        "anchor_features": ["anchor_features_*.png"],
        "time_features": ["time_features_*.mp4"],
        "quats": ["quats.npz"],
        "opacities": ["opacities.npz"],
    }

    rows: Dict[str, Dict[str, Any]] = {}
    total = 0

    for key, patterns in mapping.items():
        abs_files: List[str] = []
        for pattern in patterns:
            if os.path.isabs(pattern):
                matches = sorted(glob.glob(pattern)) if "*" in pattern else ([pattern] if os.path.exists(pattern) else [])
            else:
                full_pattern = os.path.join(out_dir, pattern)
                matches = sorted(glob.glob(full_pattern)) if "*" in pattern else ([full_pattern] if os.path.exists(full_pattern) else [])
            abs_files.extend(matches)

        # deduplicate while preserving order
        abs_files = list(dict.fromkeys(abs_files))

        size_b = sum(os.path.getsize(fp) for fp in abs_files if os.path.isfile(fp))
        total += size_b
        rows[key] = {
            "files": [os.path.basename(fp) for fp in abs_files],
            "bytes": int(size_b),
            "MB": _bytes_to_mb(size_b),
        }

    for key in rows:
        rows[key]["percent"] = 0.0 if total == 0 else 100.0 * rows[key]["bytes"] / total

    return {
        "by_attribute": rows,
        "total_bytes": int(total),
        "total_MB": _bytes_to_mb(total),
    }


# -----------------------------------------------------------------------------
# Main compression driver
# -----------------------------------------------------------------------------

def run_posthoc_2dcodec(
    ckpt_path: str,
    out_dir: str,
    cfg: CodecConfig,
) -> None:
    _mkdir(out_dir)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "splats" not in ckpt or "decoders" not in ckpt:
        raise ValueError(f"Checkpoint does not look like a GIFStream training checkpoint: {ckpt_path}")

    splats_raw = ckpt["splats"]
    splats, prep_meta = prepare_splats_for_2dcodec(splats_raw, cfg)
    n_sidelen = prep_meta["n_sidelen"]

    meta: Dict[str, Any] = {
        "codec_mode": "gifstream_2dcodec_png_hevc",
        "source_checkpoint": ckpt_path,
        "prep": prep_meta,
    }

    # Supplement-inspired storage scheme:
    # x, S1, S2, {o_i}, M -> 16-bit PNG
    # f -> 8-bit PNG
    # f_t -> 8-bit HEVC
    meta["anchors"] = _compress_tensor_to_png_chunks(out_dir, "anchors", splats["anchors"], n_sidelen, bitdepth=16)
    meta["scales"] = _compress_tensor_to_png_chunks(out_dir, "scales", splats["scales"], n_sidelen, bitdepth=16)
    meta["offsets"] = _compress_tensor_to_png_chunks(out_dir, "offsets", splats["offsets"], n_sidelen, bitdepth=16)
    meta["factors"] = _compress_tensor_to_png_chunks(out_dir, "factors", splats["factors"], n_sidelen, bitdepth=16)
    meta["anchor_features"] = _compress_tensor_to_png_chunks(out_dir, "anchor_features", splats["anchor_features"], n_sidelen, bitdepth=8)
    meta["time_features"] = _compress_time_features_hevc(
        out_dir,
        "time_features",
        splats["time_features"],
        n_sidelen=n_sidelen,
        bitdepth=8,
        video_fps=cfg.video_fps,
        hevc_qp=cfg.hevc_qp,
        ffmpeg_bin=cfg.ffmpeg_bin,
    )

    # Auxiliary side tensors retained losslessly for the released implementation.
    meta["quats"] = _save_npz(out_dir, "quats", splats["quats"])
    meta["opacities"] = _save_npz(out_dir, "opacities", splats["opacities"])

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    side_info = {
        "step": ckpt.get("step", None),
        "decoders": ckpt["decoders"],
        "app_module": ckpt.get("app_module", None),
        "codec_mode": "gifstream_2dcodec_png_hevc",
        "prep": prep_meta,
    }
    torch.save(side_info, os.path.join(out_dir, "nets.pt"))

    payload = summarize_payload(out_dir)
    with open(os.path.join(out_dir, "payload_sizes.json"), "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] 2D-codec compression written to: {out_dir}")
    print(f"      total payload: {payload['total_MB']:.3f} MB")
    for k, v in payload["by_attribute"].items():
        print(f"      {k:16s} {v['MB']:.3f} MB  ({v['percent']:.2f}%)")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Post-hoc GIFStream 2D codec compression (PNG + HEVC) from a saved training checkpoint."
    )
    ap.add_argument("--ckpt", type=str, required=True, help="Path to a GIFStream training checkpoint (*.pt)")
    ap.add_argument("--out_dir", type=str, required=True, help="Output compression directory")
    ap.add_argument("--qp", type=int, default=28, help="HEVC QP for time_features video")
    ap.add_argument("--fps", type=int, default=30, help="Nominal FPS for the HEVC bitstream")
    ap.add_argument("--time_gate_eps", type=float, default=0.0, help="Threshold on the time gate after fake_quantize_factors for keeping ft active")
    ap.add_argument("--prune_gate_eps", type=float, default=0.0, help="Threshold on the pruning gate after fake_quantize_factors for keeping anchors")
    ap.add_argument("--ffmpeg_bin", type=str, default="ffmpeg", help="ffmpeg executable name/path")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = CodecConfig(
        hevc_qp=args.qp,
        video_fps=args.fps,
        time_gate_eps=args.time_gate_eps,
        prune_gate_eps=args.prune_gate_eps,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    run_posthoc_2dcodec(args.ckpt, args.out_dir, cfg)
