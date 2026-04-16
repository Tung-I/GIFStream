"""
python examples/simple_trainer_dynamic_vanilla.py \
  --data_dir /work/pi_rsitaram_umass_edu/tungi/datasets/neural3d/flame_steak \
  --data_factor 2 \
  --GOP_size 60 \
  --start_frame 0 \
  --result_dir results_vanilla/flame_steak \
  --test_set 0 \
  --export_final_plys True
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from plyfile import PlyData, PlyElement
from torch import Tensor
import wandb
from fused_ssim import fused_ssim

from datasets.GIFStream_new import Dataset, Parser
from gsplat.rendering import rasterization
from utils import knn, rgb_to_sh, set_random_seed

from gsplat.strategy import DefaultStrategy

class FrameSubset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, target_frame: int, gop_size: int):
        self.base = base_dataset
        self.target_frame = target_frame
        self.gop_size = gop_size
        self.indices = []

        for i in range(len(base_dataset)):
            item = base_dataset[i]
            frame_idx = int(round(float(item["time"]) * (gop_size - 1)))
            if frame_idx == target_frame:
                self.indices.append(i)

        if len(self.indices) == 0:
            raise RuntimeError(f"No samples found for frame {target_frame}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

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

    # Dynamic default strategy
    strategy_prune_opa: float = 0.005
    strategy_grow_grad2d: float = 0.0002
    strategy_grow_scale3d: float = 0.01
    strategy_grow_scale2d: float = 0.05
    strategy_prune_scale3d: float = 0.1
    strategy_prune_scale2d: float = 0.15
    strategy_refine_scale2d_stop_iter: int = 0
    strategy_refine_start_iter: int = 500
    strategy_refine_stop_iter: int = 15_000
    strategy_reset_every: int = 3000
    strategy_refine_every: int = 100
    strategy_pause_refine_after_reset: int = 0
    strategy_revised_opacity: bool = False
    strategy_absgrad: bool = False
    strategy_verbose: bool = True

    # Temporal regularization.
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

    # WandB
    use_wandb: bool = True
    wandb_project: str = "dynamic-vanilla-3dgs"
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    wandb_resume: str = "allow"
    wandb_mode: str = "online"  # online / offline / disabled

    # Logging
    log_every: int = 100   # replace tb_every usage in train()

    # Sequential training
    iframe_steps: int = 30_000
    pframe_steps: int = 10_000
    iframe_eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    pframe_eval_every: int = 2_000
    eval_all_frames_at_stage_end: bool = True

    # Optional
    save_stage_ckpts: bool = True
    

def create_optimizers_for_static_splats(
    splats: torch.nn.ParameterDict,
    cfg: Config,
    scene_scale: float,
    batch_size: int,
) -> Dict[str, torch.optim.Optimizer]:
    BS = batch_size
    betas = (1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999))
    eps = 1e-15 / math.sqrt(BS)

    return {
        "means": torch.optim.Adam(
            [{"params": splats["means"], "lr": cfg.lr_means * scene_scale * math.sqrt(BS)}],
            eps=eps, betas=betas,
        ),
        "scales": torch.optim.Adam(
            [{"params": splats["scales"], "lr": cfg.lr_scales * math.sqrt(BS)}],
            eps=eps, betas=betas,
        ),
        "quats": torch.optim.Adam(
            [{"params": splats["quats"], "lr": cfg.lr_quats * math.sqrt(BS)}],
            eps=eps, betas=betas,
        ),
        "opacities": torch.optim.Adam(
            [{"params": splats["opacities"], "lr": cfg.lr_opacities * math.sqrt(BS)}],
            eps=eps, betas=betas,
        ),
        "sh0": torch.optim.Adam(
            [{"params": splats["sh0"], "lr": cfg.lr_sh0 * math.sqrt(BS)}],
            eps=eps, betas=betas,
        ),
        "shN": torch.optim.Adam(
            [{"params": splats["shN"], "lr": cfg.lr_shN * math.sqrt(BS)}],
            eps=eps, betas=betas,
        ),
    }

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

        # dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        # dist_avg = torch.sqrt(dist2_avg)
        # scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg.clamp_min(1e-12))
        scales = torch.log((dist_avg * cfg.init_scale).clamp_min(1e-6)).unsqueeze(-1).repeat(1, 3)

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

    @torch.no_grad()
    def make_static_splats_from_frame(self, t: int = 0) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(self.splats["means"][t].detach().clone()),
                "scales": torch.nn.Parameter(self.splats["scales"][t].detach().clone()),
                "quats": torch.nn.Parameter(self.splats["quats"][t].detach().clone()),
                "opacities": torch.nn.Parameter(self.splats["opacities"][t].detach().clone()),
                "sh0": torch.nn.Parameter(self.splats["sh0"][t].detach().clone()),
                "shN": torch.nn.Parameter(self.splats["shN"][t].detach().clone()),
            }
        ).to(self.device)
    
    @torch.no_grad()
    def repeat_from_static_splats(self, static_splats: torch.nn.ParameterDict) -> None:
        self.splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(static_splats["means"].detach()[None].repeat(self.T, 1, 1).contiguous()),
                "scales": torch.nn.Parameter(static_splats["scales"].detach()[None].repeat(self.T, 1, 1).contiguous()),
                "quats": torch.nn.Parameter(static_splats["quats"].detach()[None].repeat(self.T, 1, 1).contiguous()),
                "opacities": torch.nn.Parameter(static_splats["opacities"].detach()[None].repeat(self.T, 1).contiguous()),
                "sh0": torch.nn.Parameter(static_splats["sh0"].detach()[None].repeat(self.T, 1, 1, 1).contiguous()),
                "shN": torch.nn.Parameter(static_splats["shN"].detach()[None].repeat(self.T, 1, 1, 1).contiguous()),
            }
        ).to(self.device)

    def frame_state(self, t: int) -> Dict[str, Tensor]:
        return {
            "means": self.splats["means"][t],
            "scales": self.splats["scales"][t],
            "quats": F.normalize(self.splats["quats"][t], dim=-1),
            "opacities": self.splats["opacities"][t],
            "sh0": self.splats["sh0"][t],
            "shN": self.splats["shN"][t],
        }
    
    @torch.no_grad()
    def copy_frame(self, src: int, dst: int) -> None:
        for k in self.splats.keys():
            self.splats[k].data[dst].copy_(self.splats[k].data[src])

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
        
        self.wandb_run = None
        if cfg.use_wandb and cfg.wandb_mode != "disabled":
            self.wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_name,
                id=cfg.wandb_run_id,
                resume=cfg.wandb_resume,
                mode=cfg.wandb_mode,
                dir=cfg.result_dir,
                config=vars(cfg),
            )

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

        if cfg.ckpt is not None:
            self.load_ckpt(cfg.ckpt)

    def rasterize_static_splats(
        self,
        splats: torch.nn.ParameterDict,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        absgrad: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict]:
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)

        render_colors, render_alphas, info = rasterization(
            means=splats["means"],
            quats=F.normalize(splats["quats"], dim=-1),
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=absgrad,
            sparse_grad=False,
            rasterize_mode="antialiased" if self.cfg.antialiased else "classic",
            distributed=False,
            camera_model=self.cfg.camera_model,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            sh_degree=self.cfg.sh_degree,
            render_mode="RGB",
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train_iframe(self, global_step: int) -> int:
        cfg = self.cfg
        trainset0 = FrameSubset(self.trainset, target_frame=0, gop_size=cfg.GOP_size)
        valset0 = FrameSubset(self.valset, target_frame=0, gop_size=cfg.GOP_size)

        loader = torch.utils.data.DataLoader(
            trainset0, batch_size=1, shuffle=True, num_workers=4,
            persistent_workers=True, pin_memory=True
        )
        loader_iter = iter(loader)

        static_splats = self.model.make_static_splats_from_frame(0)
        static_optimizers = create_optimizers_for_static_splats(
            static_splats, cfg, self.scene_scale, cfg.batch_size
        )
        static_schedulers = {
            "means": torch.optim.lr_scheduler.ExponentialLR(
                static_optimizers["means"], gamma=0.01 ** (1.0 / cfg.iframe_steps)
            )
        }

        strategy = DefaultStrategy(
            prune_opa=cfg.strategy_prune_opa,
            grow_grad2d=cfg.strategy_grow_grad2d,
            grow_scale3d=cfg.strategy_grow_scale3d,
            grow_scale2d=cfg.strategy_grow_scale2d,
            prune_scale3d=cfg.strategy_prune_scale3d,
            prune_scale2d=cfg.strategy_prune_scale2d,
            refine_scale2d_stop_iter=cfg.strategy_refine_scale2d_stop_iter,
            refine_start_iter=cfg.strategy_refine_start_iter,
            refine_stop_iter=cfg.strategy_refine_stop_iter,
            reset_every=cfg.strategy_reset_every,
            refine_every=cfg.strategy_refine_every,
            pause_refine_after_reset=cfg.strategy_pause_refine_after_reset,
            revised_opacity=cfg.strategy_revised_opacity,
            absgrad=cfg.strategy_absgrad,
            verbose=cfg.strategy_verbose,
        )
        strategy_state = strategy.initialize_state(scene_scale=self.scene_scale)

        pbar = tqdm.tqdm(range(cfg.iframe_steps), desc="I-frame")
        for local_step in pbar:
            try:
                data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                data = next(loader_iter)

            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            masks = data.get("mask")
            masks = masks.to(self.device) if masks is not None else None
            height, width = pixels.shape[1:3]

            renders, alphas, info = self.rasterize_static_splats(
                static_splats, camtoworlds, Ks, width, height, masks=masks,
                absgrad=strategy.absgrad,
            )
            colors = torch.clamp(renders[..., :3], 0.0, 1.0)

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
                loss = loss + cfg.scale_reg * torch.abs(torch.exp(static_splats["scales"])).mean()

            for opt in static_optimizers.values():
                opt.zero_grad(set_to_none=True)

            strategy.step_pre_backward(static_splats, static_optimizers, strategy_state, local_step, info)
            loss.backward()
            strategy.step_post_backward(
                static_splats, static_optimizers, strategy_state, local_step, info, packed=cfg.packed
            )

            for opt in static_optimizers.values():
                opt.step()
            for sched in static_schedulers.values():
                sched.step()

            pbar.set_description(
                f"I-frame loss={loss.item():.4f} l1={l1loss.item():.4f} ssim={ssimloss.item():.4f}"
            )

            if local_step % cfg.log_every == 0:
                self.log_dict({
                    "iframe/train/loss": float(loss.item()),
                    "iframe/train/l1": float(l1loss.item()),
                    "iframe/train/ssim_loss": float(ssimloss.item()),
                    "iframe/train/num_gaussians": float(static_splats["means"].shape[0]),
                    "iframe/train/lr_means": float(static_optimizers["means"].param_groups[0]["lr"]),
                }, global_step)

            global_step += 1

            if (local_step + 1) in cfg.iframe_eval_steps:
                psnr0 = self.eval_static_frame_subset(static_splats, valset0)
                self.log_dict({"iframe/val/psnr": float(psnr0)}, global_step)

        # propagate the final I-frame topology + attributes to all frames
        self.model.repeat_from_static_splats(static_splats)

        # rebuild dynamic optimizers for sequential finetuning
        self.optimizers = self.model.create_optimizers(cfg.batch_size)
        self.schedulers = {
            "means": torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max(cfg.pframe_steps, 1))
            )
        }

        return global_step

    def train_one_pframe(self, frame_idx: int, global_step: int) -> int:
        cfg = self.cfg

        # warm start from previous frame
        self.model.copy_frame(frame_idx - 1, frame_idx)

        # fresh optimizer state for this stage
        self.optimizers = self.model.create_optimizers(cfg.batch_size)
        self.schedulers = {
            "means": torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max(cfg.pframe_steps, 1))
            )
        }

        train_subset = FrameSubset(self.trainset, target_frame=frame_idx, gop_size=cfg.GOP_size)

        loader = torch.utils.data.DataLoader(
            train_subset, batch_size=1, shuffle=True, num_workers=4,
            persistent_workers=True, pin_memory=True
        )
        loader_iter = iter(loader)

        pbar = tqdm.tqdm(range(cfg.pframe_steps), desc=f"P-frame {frame_idx:03d}")
        for local_step in pbar:
            try:
                data = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                data = next(loader_iter)

            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            masks = data.get("mask")
            masks = masks.to(self.device) if masks is not None else None
            height, width = pixels.shape[1:3]

            renders, alphas, _ = self.rasterize_frame(
                frame_idx=frame_idx,
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                masks=masks,
            )
            colors = torch.clamp(renders[..., :3], 0.0, 1.0)

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
                loss = loss + temporal_loss

            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            loss.backward()
            for opt in self.optimizers.values():
                opt.step()
            for sched in self.schedulers.values():
                sched.step()

            pbar.set_description(
                f"P{frame_idx:03d} loss={loss.item():.4f} l1={l1loss.item():.4f} ssim={ssimloss.item():.4f}"
            )

            if local_step % cfg.log_every == 0:
                self.log_dict({
                    f"frame_{frame_idx:03d}/train/loss": float(loss.item()),
                    f"frame_{frame_idx:03d}/train/l1": float(l1loss.item()),
                    f"frame_{frame_idx:03d}/train/ssim_loss": float(ssimloss.item()),
                    f"frame_{frame_idx:03d}/train/lr_means": float(self.optimizers["means"].param_groups[0]["lr"]),
                    "train/current_frame": float(frame_idx),
                    "train/num_gaussians": float(self.model.splats["means"].shape[1]),
                }, global_step)

            global_step += 1

            if (local_step + 1) % cfg.pframe_eval_every == 0 or (local_step + 1) == cfg.pframe_steps:
                psnr_t = self.eval_single_frame(frame_idx, global_step, log_prefix=f"frame_{frame_idx:03d}/val")
                self.log_dict({f"frame_{frame_idx:03d}/val/psnr": float(psnr_t)}, global_step)

        return global_step

    @torch.no_grad()
    def eval_static_frame_subset(self, static_splats, subset) -> float:
        loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=1)
        psnrs = []
        for data in loader:
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            masks = data.get("mask")
            masks = masks.to(self.device) if masks is not None else None
            height, width = pixels.shape[1:3]

            renders, _, _ = self.rasterize_static_splats(
                static_splats, camtoworlds, Ks, width, height, masks=masks, absgrad=False
            )
            colors = torch.clamp(renders[..., :3], 0.0, 1.0)
            mse = F.mse_loss(colors, pixels)
            psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
            psnrs.append(float(psnr.item()))
        return float(np.mean(psnrs))
    
    @torch.no_grad()
    def eval_single_frame(self, frame_idx: int, step: int, log_prefix: str = "val") -> float:
        subset = FrameSubset(self.valset, target_frame=frame_idx, gop_size=self.cfg.GOP_size)
        loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=1)
        psnrs = []

        for i, data in enumerate(loader):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            masks = data.get("mask")
            masks = masks.to(self.device) if masks is not None else None
            height, width = pixels.shape[1:3]

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

        mean_psnr = float(np.mean(psnrs))
        self.log_dict({f"{log_prefix}/psnr": mean_psnr}, step)
        return mean_psnr


    @torch.no_grad()
    def eval_all_frames(self, step: int) -> Dict[int, float]:
        metrics = {}
        for t in range(self.cfg.GOP_size):
            psnr_t = self.eval_single_frame(t, step, log_prefix=f"val/frame_{t:03d}")
            metrics[t] = psnr_t

        payload = {
            "val/mean_psnr": float(np.mean(list(metrics.values()))),
            "val/min_psnr": float(np.min(list(metrics.values()))),
            "val/max_psnr": float(np.max(list(metrics.values()))),
        }
        for t, v in metrics.items():
            payload[f"val/per_frame_psnr/frame_{t:03d}"] = float(v)
        self.log_dict(payload, step)
        return metrics

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

    def log_dict(self, payload: Dict, step: int) -> None:
        if self.wandb_run is not None:
            wandb.log(payload, step=step)

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

        global_step = 0
        global_step = self.train_iframe(global_step)

        if cfg.save_stage_ckpts:
            self.save_ckpt(global_step)
        if cfg.eval_all_frames_at_stage_end:
            self.eval_all_frames(global_step)

        for t in range(1, cfg.GOP_size):
            global_step = self.train_one_pframe(t, global_step)

            if cfg.save_stage_ckpts:
                self.save_ckpt(global_step)
            if cfg.eval_all_frames_at_stage_end:
                self.eval_all_frames(global_step)

        if cfg.export_final_plys:
            self.export_all_plys(subdir="final")

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
    cfg = tyro.cli(Config, config=(tyro.conf.FlagConversionOff,))
    runner = Runner(cfg)
    try:
        runner.train()
    finally:
        if runner.wandb_run is not None:
            runner.wandb_run.finish()


if __name__ == "__main__":
    main()
