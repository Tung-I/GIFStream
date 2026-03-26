"""
python examples/simple_trainer_dynamic_vanilla.py \
  --data_dir /work/pi_rsitaram_umass_edu/tungi/datasets/neural3d/flame_steak \
  --data_factor 2 \
  --GOP_size 60 \
  --start_frame 0 \
  --result_dir results_vanilla/flame_steak \
  --test_set 0 \
  --export_final_plys 
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from plyfile import PlyData, PlyElement
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from fused_ssim import fused_ssim

from datasets.GIFStream_new import Dataset, Parser
from gsplat.rendering import rasterization
from utils import knn, rgb_to_sh, set_random_seed

from gsplat.strategy import DynamicDefaultStrategy


@dataclass
class Config:
    # Data
    data_dir: str = "data/Neur3D/flame_steak"
    data_factor: int = 2
    result_dir: str = "results/flame_steak_dynamic_vanilla"
    test_every: int = 8
    normalize_world_space: bool = True
    global_scale: float = 1.0
    patch_size: Optional[int] = None

    # Video / GOP
    GOP_size: int = 60
    start_frame: int = 0
    test_set: Optional[List[int]] = field(default_factory=lambda: [0])
    remove_set: Optional[List[int]] = None

    # Optimization
    batch_size: int = 1
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    tb_every: int = 100
    seed: int = 42

    # Vanilla 3DGS params
    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    init_opa: float = 0.1
    init_scale: float = 1.0
    sh_degree: int = 0

    # Rendering / loss
    ssim_lambda: float = 0.2
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    antialiased: bool = False
    camera_model: str = "pinhole"
    random_bkgd: bool = False
    scale_reg: float = 0.0

    # Learning rates (mirroring the vanilla gsplat trainer defaults)
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20.0

    # Temporal regularization.
    # The topology is fixed across time, so the i-th Gaussian is treated as the
    # same trajectory across frames.
    lambda_mean_vel: float = 1e-4
    lambda_mean_acc: float = 5e-5
    lambda_scale_vel: float = 5e-5
    lambda_opa_vel: float = 1e-5
    lambda_quat_vel: float = 5e-5
    lambda_sh0_vel: float = 1e-5
    lambda_shN_vel: float = 1e-6

    enable_temporal_loss: bool = False

    # Utilities
    export_plys_on_save: bool = False
    export_final_plys: bool = True
    save_renders: bool = True
    ckpt: Optional[str] = None

def finite_check(name, x, step=None):
    if x is None:
        return
    if not torch.is_tensor(x):
        return
    x_det = x.detach()
    ok = torch.isfinite(x_det)
    if ok.all():
        if x_det.numel() > 0 and x_det.is_floating_point():
            print(
                f"[OK][step={step}] {name}: "
                f"shape={tuple(x_det.shape)} "
                f"min={x_det.min().item():.6g} "
                f"max={x_det.max().item():.6g} "
                f"mean={x_det.mean().item():.6g}"
            )
        else:
            print(f"[OK][step={step}] {name}: shape={tuple(x_det.shape)}")
    else:
        print(
            f"[BAD][step={step}] {name}: "
            f"shape={tuple(x_det.shape)} "
            f"nan={torch.isnan(x_det).sum().item()} "
            f"inf={torch.isinf(x_det).sum().item()}"
        )
        raise RuntimeError(f"Non-finite tensor detected in {name}")


class DynamicVanillaGS(torch.nn.Module):
    """
    A sequence of vanilla 3DGS models with shared topology across time.

    Each frame has its own means/scales/quats/opacities/SH coefficients,
    but the Gaussian indexing is shared across all frames. This makes the
    representation much easier to use for downstream post-hoc compression.
    """

    def __init__(
        self,
        parser: Parser,
        cfg: Config,
        scene_scale: float,
        device: str,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.T = cfg.GOP_size
        self.scene_scale = scene_scale

        if cfg.init_type == "sfm":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif cfg.init_type == "random":
            points = cfg.init_extent * scene_scale * (torch.rand((cfg.init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((cfg.init_num_pts, 3))
        else:
            raise ValueError("init_type must be 'sfm' or 'random'.")

        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

        N = points.shape[0]
        quats = torch.rand((N, 4))
        quats = F.normalize(quats, dim=-1)
        opacities = torch.logit(torch.full((N,), cfg.init_opa))

        colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        sh0 = colors[:, :1, :]
        shN = colors[:, 1:, :]

        # Copy the same initialization to every frame.
        self.splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(points.unsqueeze(0).repeat(self.T, 1, 1).contiguous()),
                "scales": torch.nn.Parameter(scales.unsqueeze(0).repeat(self.T, 1, 1).contiguous()),
                "quats": torch.nn.Parameter(quats.unsqueeze(0).repeat(self.T, 1, 1).contiguous()),
                "opacities": torch.nn.Parameter(opacities.unsqueeze(0).repeat(self.T, 1).contiguous()),
                "sh0": torch.nn.Parameter(sh0.unsqueeze(0).repeat(self.T, 1, 1, 1).contiguous()),
                "shN": torch.nn.Parameter(shN.unsqueeze(0).repeat(self.T, 1, 1, 1).contiguous()),
            }
        ).to(device)

    def frame_state(self, t: int) -> Dict[str, Tensor]:
        return {
            "means": self.splats["means"][t],
            "scales": self.splats["scales"][t],
            "quats": F.normalize(self.splats["quats"][t], dim=-1),
            "opacities": self.splats["opacities"][t],
            "sh0": self.splats["sh0"][t],
            "shN": self.splats["shN"][t],
        }

    def create_optimizers(self, batch_size: int) -> Dict[str, torch.optim.Optimizer]:
        BS = batch_size
        betas = (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999))
        eps = 1e-15 / math.sqrt(BS)
        return {
            "means": torch.optim.Adam(
                [{"params": self.splats["means"], "lr": self.cfg.lr_means * self.scene_scale * math.sqrt(BS)}],
                eps=eps,
                betas=betas,
            ),
            "scales": torch.optim.Adam(
                [{"params": self.splats["scales"], "lr": self.cfg.lr_scales * math.sqrt(BS)}],
                eps=eps,
                betas=betas,
            ),
            "quats": torch.optim.Adam(
                [{"params": self.splats["quats"], "lr": self.cfg.lr_quats * math.sqrt(BS)}],
                eps=eps,
                betas=betas,
            ),
            "opacities": torch.optim.Adam(
                [{"params": self.splats["opacities"], "lr": self.cfg.lr_opacities * math.sqrt(BS)}],
                eps=eps,
                betas=betas,
            ),
            "sh0": torch.optim.Adam(
                [{"params": self.splats["sh0"], "lr": self.cfg.lr_sh0 * math.sqrt(BS)}],
                eps=eps,
                betas=betas,
            ),
            "shN": torch.optim.Adam(
                [{"params": self.splats["shN"], "lr": self.cfg.lr_shN * math.sqrt(BS)}],
                eps=eps,
                betas=betas,
            ),
        }


def quat_temporal_loss(q0: Tensor, q1: Tensor) -> Tensor:
    q0 = F.normalize(q0, dim=-1)
    q1 = F.normalize(q1, dim=-1)
    cos = (q0 * q1).sum(dim=-1).abs().clamp(max=1.0)
    return (1.0 - cos).mean()


def smooth_l1(x: Tensor, y: Tensor) -> Tensor:
    return F.smooth_l1_loss(x, y)


def compute_temporal_regularization(model: DynamicVanillaGS, t: int, cfg: Config) -> Tuple[Tensor, Dict[str, float]]:
    T = model.T
    terms: Dict[str, Tensor] = {}

    if t > 0:
        terms["mean_vel"] = smooth_l1(model.splats["means"][t], model.splats["means"][t - 1])
        terms["scale_vel"] = smooth_l1(model.splats["scales"][t], model.splats["scales"][t - 1])
        terms["opa_vel"] = smooth_l1(model.splats["opacities"][t], model.splats["opacities"][t - 1])
        terms["quat_vel"] = quat_temporal_loss(model.splats["quats"][t], model.splats["quats"][t - 1])
        terms["sh0_vel"] = smooth_l1(model.splats["sh0"][t], model.splats["sh0"][t - 1])
        # terms["shN_vel"] = smooth_l1(model.splats["shN"][t], model.splats["shN"][t - 1])

    if 0 < t < T - 1:
        v_prev = model.splats["means"][t] - model.splats["means"][t - 1]
        v_next = model.splats["means"][t + 1] - model.splats["means"][t]
        terms["mean_acc"] = smooth_l1(v_next, v_prev)

    loss = torch.zeros([], device=model.device)
    loss = loss + cfg.lambda_mean_vel * terms.get("mean_vel", torch.zeros([], device=model.device))
    loss = loss + cfg.lambda_mean_acc * terms.get("mean_acc", torch.zeros([], device=model.device))
    loss = loss + cfg.lambda_scale_vel * terms.get("scale_vel", torch.zeros([], device=model.device))
    loss = loss + cfg.lambda_opa_vel * terms.get("opa_vel", torch.zeros([], device=model.device))
    loss = loss + cfg.lambda_quat_vel * terms.get("quat_vel", torch.zeros([], device=model.device))
    loss = loss + cfg.lambda_sh0_vel * terms.get("sh0_vel", torch.zeros([], device=model.device))
    # loss = loss + cfg.lambda_shN_vel * terms.get("shN_vel", torch.zeros([], device=model.device))

    stats = {k: float(v.detach().item()) for k, v in terms.items()}
    return loss, stats


class Runner:
    def __init__(self, cfg: Config) -> None:
        assert cfg.batch_size == 1, "This script currently assumes batch_size=1."
        set_random_seed(cfg.seed)

        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(cfg.result_dir, "ckpts")
        self.render_dir = os.path.join(cfg.result_dir, "renders")
        self.stats_dir = os.path.join(cfg.result_dir, "stats")
        self.ply_dir = os.path.join(cfg.result_dir, "plys")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)
        os.makedirs(self.ply_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.result_dir, "tb"))

        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            first_frame=cfg.start_frame,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=False,
            test_set=cfg.test_set,
            remove_set=cfg.remove_set,
            GOP_size=cfg.GOP_size,
            start_frame=cfg.start_frame,
        )
        self.valset = Dataset(
            self.parser,
            split="val",
            test_set=cfg.test_set,
            remove_set=cfg.remove_set,
            GOP_size=cfg.GOP_size,
            start_frame=cfg.start_frame,
        )

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        self.model = DynamicVanillaGS(self.parser, cfg, self.scene_scale, self.device)
        self.optimizers = self.model.create_optimizers(cfg.batch_size)
        self.schedulers = {
            "means": torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / cfg.max_steps)
            )
        }

        if cfg.ckpt is not None:
            self.load_ckpt(cfg.ckpt)

        self.strategy = DynamicDefaultStrategy(
            prune_opa=0.005,
            grow_grad2d=0.0002,
            grow_scale3d=0.01,
            grow_scale2d=0.05,
            prune_scale3d=0.1,
            prune_scale2d=0.15,
            refine_start_iter=500,
            refine_stop_iter=15000,
            refine_every=100,
            reset_every=3000,
            verbose=True,
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)

    def save_ckpt(self, step: int) -> None:
        data = {
            "step": step,
            "splats": self.model.splats.state_dict(),
        }
        torch.save(data, os.path.join(self.ckpt_dir, f"ckpt_{step}.pt"))

    def load_ckpt(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.splats.load_state_dict(ckpt["splats"])
        print(f"Loaded checkpoint from {path}")

    def rasterize_frame(
        self,
        frame_idx: int,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict]:
        
        splats = self.model.frame_state(frame_idx)
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        render_colors, render_alphas, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=False,
            sparse_grad=False,
            rasterize_mode="antialiased" if self.cfg.antialiased else "classic",
            distributed=False,
            camera_model=self.cfg.camera_model,
            near_plane=self.cfg.near_plane,
            sh_degree=self.cfg.sh_degree, 
            far_plane=self.cfg.far_plane,
            render_mode="RGB",
        )

        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self) -> None:
        cfg = self.cfg
        with open(os.path.join(cfg.result_dir, "cfg.yml"), "w") as f:
            yaml.dump(vars(cfg), f)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        global_tic = time.time()
        pbar = tqdm.tqdm(range(cfg.max_steps))
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            masks = data.get("mask")
            masks = masks.to(self.device) if masks is not None else None
            height, width = pixels.shape[1:3]

            frame_idx = int(round(float(data["time"][0]) * (cfg.GOP_size - 1)))
            renders, alphas, info = self.rasterize_frame(
                frame_idx=frame_idx,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                masks=masks,
            )
            colors = renders[..., :3]
            # colors = torch.clamp(renders[..., :3], 0.0, 1.0)

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=self.device)
                colors = colors + bkgd * (1.0 - alphas)

            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
                padding="valid",
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            if cfg.scale_reg > 0.0:
                loss = loss + cfg.scale_reg * torch.abs(torch.exp(self.model.splats["scales"][frame_idx])).mean()

            if cfg.enable_temporal_loss:
                temporal_loss, temporal_stats = compute_temporal_regularization(self.model, frame_idx, cfg)
            # finite_check("temporal_loss", temporal_loss, step)

                loss = loss + temporal_loss

            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)

            self.strategy.step_pre_backward(
                self.model.splats, self.optimizers, self.strategy_state, step, info
            )

            loss.backward()

            self.strategy.step_post_backward(
                self.model.splats,
                self.optimizers,
                self.strategy_state,
                step,
                info,
                packed=self.cfg.packed,
            )

            for opt in self.optimizers.values():
                opt.step()
            for sched in self.schedulers.values():
                sched.step()

            desc = (
                f"loss={loss.item():.4f} | l1={l1loss.item():.4f} | "
                f"ssim={ssimloss.item():.4f} | t={frame_idx:02d}"
            )
            pbar.set_description(desc)

            if step % cfg.tb_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1", l1loss.item(), step)
                self.writer.add_scalar("train/ssim_loss", ssimloss.item(), step)
                if cfg.enable_temporal_loss:
                    self.writer.add_scalar("train/temporal", temporal_loss.item(), step)
                    for k, v in temporal_stats.items():
                        self.writer.add_scalar(f"train_temporal/{k}", v, step)
                self.writer.add_scalar("train/num_gaussians", self.model.splats["means"].shape[1], step)
                self.writer.flush()

            if step in [s - 1 for s in cfg.save_steps] or step == cfg.max_steps - 1:
                self.save_ckpt(step)
                stats = {
                    "step": step,
                    "elapsed_sec": time.time() - global_tic,
                    "num_gaussians": int(self.model.splats["means"].shape[1]),
                }
                with open(os.path.join(self.stats_dir, f"train_step{step:04d}.json"), "w") as f:
                    json.dump(stats, f, indent=2)
                if cfg.export_plys_on_save:
                    self.export_all_plys(subdir=f"step_{step:06d}")

            if step in [s - 1 for s in cfg.eval_steps]:
                self.eval(step)

        if cfg.export_final_plys:
            self.export_all_plys(subdir="final")

    @torch.no_grad()
    def eval(self, step: int) -> None:
        self.model.eval()
        loader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False, num_workers=1)
        psnrs: List[float] = []
        for i, data in enumerate(loader):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            masks = data.get("mask")
            masks = masks.to(self.device) if masks is not None else None
            height, width = pixels.shape[1:3]
            frame_idx = int(round(float(data["time"][0]) * (self.cfg.GOP_size - 1)))

            renders, _, _ = self.rasterize_frame(
                frame_idx=frame_idx,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                masks=masks,
            )
            colors = torch.clamp(renders[..., :3], 0.0, 1.0)
            mse = F.mse_loss(colors, pixels)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            psnrs.append(float(psnr.item()))

            if self.cfg.save_renders:
                pred = (colors[0].cpu().numpy() * 255.0).astype(np.uint8)
                imageio.imwrite(os.path.join(self.render_dir, f"val_step{step:06d}_{i:04d}.png"), pred)

        summary = {
            "step": step,
            "mean_psnr": float(np.mean(psnrs)) if psnrs else None,
            "num_views": len(psnrs),
        }
        print("Eval:", summary)
        with open(os.path.join(self.stats_dir, f"eval_step{step:06d}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if summary["mean_psnr"] is not None:
            self.writer.add_scalar("val/psnr", summary["mean_psnr"], step)
        self.writer.flush()
        self.model.train()

    def export_all_plys(self, subdir: str = "final") -> None:
        out_dir = os.path.join(self.ply_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        for t in range(self.cfg.GOP_size):
            save_frame_ply(self.model.splats, t, os.path.join(out_dir, f"frame_{t:03d}.ply"))
        print(f"Exported PLYs to {out_dir}")


def save_frame_ply(splats: torch.nn.ParameterDict, frame_idx: int, path: str) -> None:
    means = splats["means"][frame_idx].detach().cpu().numpy()
    normals = np.zeros_like(means)
    sh0 = (
        splats["sh0"][frame_idx]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    shN = (
        splats["shN"][frame_idx]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    opacities = splats["opacities"][frame_idx].detach().unsqueeze(1).cpu().numpy()
    scales = splats["scales"][frame_idx].detach().cpu().numpy()
    quats = F.normalize(splats["quats"][frame_idx].detach(), dim=-1).cpu().numpy()

    attrs = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ]
    attrs += [(f"f_dc_{i}", "f4") for i in range(sh0.shape[1])]
    attrs += [(f"f_rest_{i}", "f4") for i in range(shN.shape[1])]
    attrs += [("opacity", "f4")]
    attrs += [(f"scale_{i}", "f4") for i in range(scales.shape[1])]
    attrs += [(f"rot_{i}", "f4") for i in range(quats.shape[1])]

    elements = np.empty(means.shape[0], dtype=attrs)
    packed = np.concatenate([means, normals, sh0, shN, opacities, scales, quats], axis=1)
    elements[:] = list(map(tuple, packed))
    PlyData([PlyElement.describe(elements, "vertex")], text=False).write(path)


def main() -> None:
    cfg = tyro.cli(Config)
    runner = Runner(cfg)
    runner.train()


if __name__ == "__main__":
    main()
