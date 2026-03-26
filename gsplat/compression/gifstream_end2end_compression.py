import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from gsplat.compression.stream_helper import encode_x, decode_x, filesize

from gsplat.compression.outlier_filter import filter_splats
from gsplat.compression.sort import sort_splats, sort_anchors
from gsplat.utils import inverse_log_transform, log_transform
import math


@dataclass
class GIFStreamEnd2endCompression:
    """Uses quantization and sorting to compress splats into PNG files and uses
    K-means clustering to compress the spherical harmonic coefficents.

    .. warning::
        This class requires the `imageio <https://pypi.org/project/imageio/>`_,
        `plas <https://github.com/fraunhoferhhi/PLAS.git>`_
        and `torchpq <https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install>`_ packages to be installed.

    .. warning::
        This class might throw away a few lowest opacities splats if the number of
        splats is not a square number.

    .. note::
        The splats parameters are expected to be pre-activation values. It expects
        the following fields in the splats dictionary: "means", "scales", "quats",
        "opacities", "sh0", "shN". More fields can be added to the dictionary, but
        they will only be compressed using NPZ compression.

    References:
        - `Compact 3D Scene Representation via Self-Organizing Gaussian Grids <https://arxiv.org/abs/2312.13299>`_
        - `Making Gaussian Splats more smaller <https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/>`_

    Args:
        use_sort (bool, optional): Whether to sort splats before compression. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Default to True.
    """

    use_sort: bool = True
    verbose: bool = True

    def _get_compress_fn(self, param_name: str) -> Callable:
        compress_fn_map = {
            "anchors": _compress_png_16bit,
            "scales": _compress_end2end,
            "offsets": _compress_end2end,
            "anchor_features": _compress_end2end_ar,
            "factors": _compress_end2end,
            "time_features": _compress_end2end_ar,
        }
        if param_name in compress_fn_map:
            return compress_fn_map[param_name]
        else:
            return _compress_npz

    def _get_decompress_fn(self, param_name: str) -> Callable:
        decompress_fn_map = {
            "anchors": _decompress_png_16bit,
            "scales": _decompress_end2end,
            "offsets": _decompress_end2end,
            "anchor_features": _decompress_end2end_ar,
            "factors": _decompress_end2end,
            "time_features": _decompress_end2end_ar,
        }
        if param_name in decompress_fn_map:
            return decompress_fn_map[param_name]
        else:
            return _decompress_npz

    def compress(self, compress_dir: str, splats: Dict[str, Tensor], entropy_models: Dict[str, Module] = None, c_channel: int = 0, p_channel: int = 0, scaling = None, voxel_size = 0.01) -> None:
        """Run compression

        Args:
            compress_dir (str): directory to save compressed files
            splats (Dict[str, Tensor]): Gaussian splats to compress
        """
        
        # quantization scaling
        if scaling is None:
            scaling = {
                "anchors": None,
                "scales": 0.01,
                "quats": None,
                "opacities": None,
                "anchor_features": 1,
                "offsets": 0.01,
                "factors": 1/16,
                "time_features": 1,
            }

        # Param-specific preprocessing
        # splats["anchors"] = log_transform(splats["anchors"])
        splats["quats"] = F.normalize(splats["quats"], dim=-1)
        pruning_mask = splats["factors"][:,-1] > 0
        for k,v in splats.items():
            splats[k] = v[pruning_mask]

        n_gs = len(splats["anchors"])
        n_sidelen = math.ceil(n_gs**0.5)
        n_crop = n_gs - n_sidelen**2

        if n_crop != 0:
            splats = _crop_n_splats(splats, n_crop)
            print(
                f"Warning: Number of Gaussians was not square. Removed {n_crop} Gaussians."
            )

        if self.use_sort:
            splats = sort_anchors(splats)

        choose_idx = splats["factors"][:,0] > 0
        splats["time_features"] = splats["time_features"][choose_idx]

        meta = {}
        print(entropy_models.keys())
        for param_name in splats.keys():
            compress_fn = self._get_compress_fn(param_name)
            kwargs = {
                "n_sidelen": n_sidelen,
                "verbose": self.verbose,
                # quantized anchor_features are used
                "anchor_features": torch.round(splats["anchor_features"]/scaling["anchor_features"]) * scaling["anchor_features"],
                "c_channel": c_channel,
                "p_channel": p_channel,
                "scaling": scaling[param_name],
                "entropy_model": entropy_models[param_name],
                "voxel_size": voxel_size
            }

            meta[param_name] = compress_fn(
                compress_dir, param_name, splats[param_name], **kwargs
            )

        with open(os.path.join(compress_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

    def decompress(self, compress_dir: str, entropy_models, device) -> Dict[str, Tensor]:
        """Run decompression

        Args:
            compress_dir (str): directory that contains compressed files

        Returns:
            Dict[str, Tensor]: decompressed Gaussian splats
        """
        def inverse_sigmoid(x):
            return -torch.log(1/(x.clamp(1e-7,1-1e-7)) - 1)
        
        with open(os.path.join(compress_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        splats = {}
        decompress_fn = self._get_decompress_fn("anchor_features")
        kwargs = {
            "entropy_model": entropy_models["anchor_features"],
            "device": device,
        }
        splats["anchor_features"] = decompress_fn(compress_dir, "anchor_features", meta["anchor_features"], **kwargs)
        for param_name, param_meta in meta.items():
            if param_name == "anchor_features": continue
            decompress_fn = self._get_decompress_fn(param_name)
            kwargs = {
                "anchor_features": splats["anchor_features"],
                "entropy_model": entropy_models[param_name],
                "device": device
            }
            splats[param_name] = decompress_fn(compress_dir, param_name, param_meta, **kwargs)

        # Param-specific postprocessing
        # splats["anchors"] = inverse_log_transform(splats["anchors"])
        #* re-voxelize
        mask = (splats["quats"].any(dim=1) != 0)
        for k,v in splats.items():
            if k != "time_features":
                splats[k] = v[mask]
        voxel_size = meta["anchors"]["voxel_size"]
        splats["anchors"] = torch.round(splats["anchors"]/voxel_size)*voxel_size

        #* recover time features
        choose_idx = splats["factors"][:,0] > 0
        time_features = torch.zeros((len(splats["anchors"]),meta["time_features"]["shape"][1], meta["time_features"]["shape"][2]),device=device)
        time_features[choose_idx] = splats["time_features"]
        splats["time_features"] = time_features

        splats["factors"] = inverse_sigmoid(splats["factors"])
        return splats


def _crop_n_splats(splats: Dict[str, Tensor], n_crop: int) -> Dict[str, Tensor]:
    if n_crop > 0:
        opacities = splats["opacities"].view((-1))
        keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
        for k, v in splats.items():
            splats[k] = v[keep_indices]
        return splats
    else:
        for k, v in splats.items():
            splats[k] = torch.cat([v,torch.zeros([-n_crop]+list(v.shape[1:]), device = v.device)],dim=0)
        return splats


def _compress_png(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 8-bit quantization and lossless PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    img = img.squeeze()
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_png(compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs) -> Tensor:
    """Decompress parameters from PNG file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def _compress_png_kbit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, quantization: int = 8, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with k-bit quantization and lossless PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**quantization - 1)).round().astype(np.uint8)
    img = img << (8 - quantization)
    img = img.squeeze()
    if grid.shape[-1] > 4:
        for ind in range(grid.shape[-1]//3):
            imageio.imwrite(os.path.join(compress_dir, f"{param_name}_{ind}.png"), img[:,:,3*ind:3*ind+3])
    else:
        imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "quantization": quantization, 
    }
    return meta


def _decompress_png_kbit(compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs) -> Tensor:
    """Decompress parameters from PNG file.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    if np.prod(meta["shape"][1:]) > 4:
        for ind in range(np.prod(meta["shape"][1:])//3):
            tmp_img = imageio.imread(os.path.join(compress_dir, f"{param_name}_{ind}.png")) 
            img = tmp_img if ind == 0 else np.concatenate([img, tmp_img], axis=-1)
    else:
        img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img = img >> (8 - meta["quantization"])
    img_norm = img / (2**meta["quantization"] - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_png_16bit(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)

    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_l.png"), img_l.astype(np.uint8)
    )
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_u.png"), img_u.astype(np.uint8)
    )

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "voxel_size": kwargs["voxel_size"]
    }
    return meta


def _decompress_png_16bit(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from PNG files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    img_l = imageio.imread(os.path.join(compress_dir, f"{param_name}_l.png"))
    img_u = imageio.imread(os.path.join(compress_dir, f"{param_name}_u.png"))
    img_u = img_u.astype(np.uint16)
    img = (img_u << 8) + img_l

    img_norm = img / (2**16 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_npz(
    compress_dir: str, param_name: str, params: Tensor, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with numpy's NPZ compression."""
    npz_dict = {"arr": params.detach().cpu().numpy()}
    save_fp = os.path.join(compress_dir, f"{param_name}.npz")
    os.makedirs(os.path.dirname(save_fp), exist_ok=True)
    np.savez_compressed(save_fp, **npz_dict)
    meta = {
        "shape": params.shape,
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_npz(compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs) -> Tensor:
    """Decompress parameters with numpy's NPZ compression."""
    arr = np.load(os.path.join(compress_dir, f"{param_name}.npz"))["arr"]
    params = torch.tensor(arr)
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_end2end(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    params = params/kwargs["scaling"]
    anchor_features = kwargs["anchor_features"]
    entropy_model = kwargs["entropy_model"]
    output_path = os.path.join(compress_dir,f"{param_name}.bin")
    entropy_model.compress(params.flatten(1),anchor_features,output_path, adaptive=True)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "scaling": kwargs["scaling"]
    }
    return meta


def _decompress_end2end(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from PNG files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    
    anchor_features = kwargs["anchor_features"]
    entropy_model = kwargs["entropy_model"]
    output_path = os.path.join(compress_dir,f"{param_name}.bin")
    params = entropy_model.decompress(anchor_features, output_path, adaptive=True) * meta["scaling"]

    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def _compress_end2end_ar(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> Dict[str, Any]:
    """Compress parameters with 16-bit quantization and PNG compression.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        params (Tensor): parameters
        n_sidelen (int): image side length

    Returns:
        Dict[str, Any]: metadata
    """
    import imageio.v2 as imageio

    if torch.numel == 0:
        meta = {
            "shape": list(params.shape),
            "dtype": str(params.dtype).split(".")[1],
        }
        return meta

    params = params/kwargs["scaling"]
    channel = kwargs["c_channel"] if param_name == "anchor_features" else kwargs["p_channel"]
    N, f_channel = params.flatten(1).shape
    condition = torch.cat([torch.zeros((N,3*channel), device=params.device), params.flatten(1)],dim=-1)
    condition = torch.cat([condition.view((N,-1,channel))[:,x:-3+x] for x in range(3)],dim=-1)
    entropy_model = kwargs["entropy_model"]
    for ind in range(f_channel//channel):
        output_path = os.path.join(compress_dir,f"{param_name}_{ind:05d}.bin")
        entropy_model.compress(params.flatten(1)[:,ind*channel:ind*channel+channel],condition[:,ind],output_path)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "scaling": kwargs["scaling"],
        "length": f_channel // channel,
        "channel": channel
    }
    return meta


def _decompress_end2end_ar(
    compress_dir: str, param_name: str, meta: Dict[str, Any], **kwargs
) -> Tensor:
    """Decompress parameters from PNG files.

    Args:
        compress_dir (str): compression directory
        param_name (str): parameter field name
        meta (Dict[str, Any]): metadata

    Returns:
        Tensor: parameters
    """
    import imageio.v2 as imageio

    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"], dtype=getattr(torch, meta["dtype"]))
        return meta

    entropy_model = kwargs["entropy_model"]
    condition = torch.zeros((meta["shape"][0],meta["channel"] * 3), device=kwargs["device"])
    for ind in range(meta["length"]):
        output_path = os.path.join(compress_dir,f"{param_name}_{ind:05d}.bin")
        tmp = entropy_model.decompress(condition, output_path)
        condition = torch.cat([condition[:,meta["channel"]:], tmp],dim=-1)
        params = tmp if ind == 0 else torch.cat([params,tmp],dim=-1)

    params = params.reshape(meta["shape"]) * meta["scaling"]
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params

def save_ply(splats: torch.nn.ParameterDict, path: str):
    from plyfile import PlyData, PlyElement

    means = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(means)
    sh0 = splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    shN = splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = splats["opacities"].detach().unsqueeze(1).cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()

    def construct_list_of_attributes(splats):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(splats["sh0"].shape[1]*splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(splats["shN"].shape[1]*splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(splats)]

    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, sh0, shN, opacities, scales, quats), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def save_params_into_ply_file(
    splats, path
):
    """Save parameters of Gaussian Splats into .ply file"""
    ply_dir = f"{path}/ply"
    os.makedirs(ply_dir, exist_ok=True)
    ply_file = ply_dir + "/pruned_splats.ply"
    save_ply(splats, ply_file)
    print(f"Saved parameters of splats into file: {ply_file}.")