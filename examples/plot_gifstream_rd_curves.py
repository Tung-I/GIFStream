#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_end2end_points(base_dir: Path, scene: str, gop: str) -> List[Dict]:
    """
    Expected layout:
      base_dir/scene/gop/r0/my_eval/payload_sizes.json
      base_dir/scene/gop/r0/my_eval/quality.json
      ... r1, r2, r3 ...
    """
    scene_dir = base_dir / scene / gop
    if not scene_dir.exists():
        raise FileNotFoundError(f"End2end scene dir not found: {scene_dir}")

    points = []
    for rate_dir in sorted(scene_dir.glob("r*")):
        if not rate_dir.is_dir():
            continue
        m = re.fullmatch(r"r(\d+)", rate_dir.name)
        if m is None:
            continue

        eval_dir = rate_dir / "my_eval"
        payload_fp = eval_dir / "payload_sizes.json"
        quality_fp = eval_dir / "quality.json"
        if not payload_fp.exists() or not quality_fp.exists():
            continue

        payload = load_json(payload_fp)
        quality = load_json(quality_fp)

        points.append(
            {
                "label": rate_dir.name,
                "rate_idx": int(m.group(1)),
                "payload_mb": float(payload["total_mb"]),
                "psnr": float(quality["psnr"]),
                "ssim": float(quality["ssim"]),
                "lpips": float(quality["lpips"]),
                "lpips_net": quality.get("lpips_net", None),
                "num_images": quality.get("num_images", None),
                "path": str(eval_dir),
            }
        )

    if not points:
        raise FileNotFoundError(
            f"No end2end points found under {scene_dir}. Expected r*/my_eval/payload_sizes.json and quality.json"
        )

    points.sort(key=lambda x: x["rate_idx"])
    return points


def parse_2dcodec_points(base_dir: Path, scene: str, gop: str, anchor_rate: str = "r0") -> List[Dict]:
    """
    Expected layout:
      base_dir/scene/gop/r0/eval_2dcodec/qp20/quality.json
      base_dir/scene/gop/r0/eval_2dcodec/qp28/quality.json
      ...
    """
    eval_root = base_dir / scene / gop / anchor_rate / "eval_2dcodec"
    if not eval_root.exists():
        raise FileNotFoundError(f"2dcodec eval root not found: {eval_root}")

    points = []
    for qp_dir in sorted(eval_root.glob("qp*")):
        if not qp_dir.is_dir():
            continue
        m = re.fullmatch(r"qp(\d+)", qp_dir.name)
        if m is None:
            continue

        quality_fp = qp_dir / "quality.json"
        if not quality_fp.exists():
            continue

        quality = load_json(quality_fp)
        if "payload_total_MB" not in quality:
            raise KeyError(f"Missing payload_total_MB in {quality_fp}")

        points.append(
            {
                "label": qp_dir.name,
                "qp": int(m.group(1)),
                "payload_mb": float(quality["payload_total_MB"]),
                "psnr": float(quality["psnr"]),
                "ssim": float(quality["ssim"]),
                "lpips": float(quality["lpips"]),
                "lpips_net": quality.get("lpips_net", None),
                "num_images": quality.get("num_images", None),
                "avg_render_sec": quality.get("avg_render_sec", None),
                "avg_rasterization_sec": quality.get("avg_rasterization_sec", None),
                "path": str(qp_dir),
            }
        )

    if not points:
        raise FileNotFoundError(
            f"No 2dcodec points found under {eval_root}. Expected qp*/quality.json"
        )

    points.sort(key=lambda x: x["qp"])
    return points


def save_points_summary(points: List[Dict], out_fp: Path) -> None:
    with open(out_fp, "w") as f:
        json.dump(points, f, indent=2)


def metric_direction_note(metric: str) -> str:
    return "lower is better" if metric.lower() == "lpips" else "higher is better"


def maybe_warn_lpips_nets(end2end_points: List[Dict], codec2d_points: List[Dict]) -> Optional[str]:
    e_nets = sorted({p.get("lpips_net") for p in end2end_points if p.get("lpips_net") is not None})
    c_nets = sorted({p.get("lpips_net") for p in codec2d_points if p.get("lpips_net") is not None})
    if not e_nets or not c_nets:
        return None
    if e_nets != c_nets:
        return (
            f"LPIPS backbone mismatch: end2end uses {e_nets}, 2dcodec uses {c_nets}. "
            f"LPIPS curves are therefore not strictly apples-to-apples."
        )
    return None


def plot_metric(
    end2end_points: List[Dict],
    codec2d_points: List[Dict],
    metric: str,
    out_fp: Path,
    title: str,
    subtitle: Optional[str] = None,
) -> None:
    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(111)

    x1 = [p["payload_mb"] for p in end2end_points]
    y1 = [p[metric] for p in end2end_points]
    l1 = [p["label"] for p in end2end_points]

    x2 = [p["payload_mb"] for p in codec2d_points]
    y2 = [p[metric] for p in codec2d_points]
    l2 = [p["label"] for p in codec2d_points]

    ax.plot(x1, y1, marker="o", linewidth=2, label="GIFStream end2end entropy coding")
    ax.plot(x2, y2, marker="s", linewidth=2, label="GIFStream 2D codec")

    for x, y, t in zip(x1, y1, l1):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(5, 6), fontsize=9)
    for x, y, t in zip(x2, y2, l2):
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(5, -12), fontsize=9)

    ax.set_xlabel("Total payload size (MB)")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    note = metric_direction_note(metric)
    if subtitle:
        fig.text(0.5, 0.01, f"{subtitle} | {note}", ha="center", fontsize=9)
    else:
        fig.text(0.5, 0.01, note, ha="center", fontsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_fp, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RD-curve comparison for GIFStream end2end vs 2D codec compression.")
    parser.add_argument("--base_dir", type=str, required=True, help="Root directory such as results_ori_codec")
    parser.add_argument("--scene", type=str, default="flame_steak")
    parser.add_argument("--gop", type=str, default="GOP_0")
    parser.add_argument("--anchor_rate", type=str, default="r0", help="Base checkpoint rate used for 2D codec results")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (base_dir / args.scene / args.gop / "rd_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    end2end_points = parse_end2end_points(base_dir, args.scene, args.gop)
    codec2d_points = parse_2dcodec_points(base_dir, args.scene, args.gop, anchor_rate=args.anchor_rate)

    save_points_summary(end2end_points, out_dir / "end2end_points.json")
    save_points_summary(codec2d_points, out_dir / "2dcodec_points.json")

    lpips_warning = maybe_warn_lpips_nets(end2end_points, codec2d_points)
    subtitle = lpips_warning

    title_prefix = f"GIFStream RD comparison: {args.scene} / {args.gop}"
    plot_metric(end2end_points, codec2d_points, "psnr", out_dir / "rd_curve_psnr.png", f"{title_prefix} (PSNR)", subtitle=None)
    plot_metric(end2end_points, codec2d_points, "ssim", out_dir / "rd_curve_ssim.png", f"{title_prefix} (SSIM)", subtitle=None)
    plot_metric(end2end_points, codec2d_points, "lpips", out_dir / "rd_curve_lpips.png", f"{title_prefix} (LPIPS)", subtitle=subtitle)

    summary = {
        "scene": args.scene,
        "gop": args.gop,
        "base_dir": str(base_dir),
        "anchor_rate": args.anchor_rate,
        "num_end2end_points": len(end2end_points),
        "num_2dcodec_points": len(codec2d_points),
        "lpips_warning": lpips_warning,
        "outputs": {
            "psnr": str(out_dir / "rd_curve_psnr.png"),
            "ssim": str(out_dir / "rd_curve_ssim.png"),
            "lpips": str(out_dir / "rd_curve_lpips.png"),
            "end2end_points": str(out_dir / "end2end_points.json"),
            "2dcodec_points": str(out_dir / "2dcodec_points.json"),
        },
    }
    with open(out_dir / "rd_plot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] RD plots written to: {out_dir}")
    print(f"      - {out_dir / 'rd_curve_psnr.png'}")
    print(f"      - {out_dir / 'rd_curve_ssim.png'}")
    print(f"      - {out_dir / 'rd_curve_lpips.png'}")
    if lpips_warning:
        print(f"[WARN] {lpips_warning}")


if __name__ == "__main__":
    main()
