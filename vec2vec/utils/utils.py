import multiprocessing
import os, json
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from huggingface_hub.file_download import hf_hub_download
from safetensors.torch import load_file

from translators.IdentityBaseline import IdentityBaseline
from translators.MLPWithResidual import MLPWithResidual
from translators.LinearTranslator import LinearTranslator
from translators.TransformTranslator import TransformTranslator
from translators.transforms.UNetTransform import UNetTransform
from translators.transforms.UNet1dTransform import UNet1dTransform
from translators.SharedAE import SharedAETranslator
from translators.SharedAE import SharedAETranslator
from vec2text.models import InversionModel


def load_n_translator(cfg, encoder_dims):
    if cfg.style == 'identity':
        return IdentityBaseline(encoder_dims)
    if cfg.style == 'linear':
        return LinearTranslator(
            encoder_dims,
            cfg.normalize_embeddings,
            cfg.src_emb if hasattr(cfg, 'src_emb') else None,
            cfg.tgt_emb if hasattr(cfg, 'tgt_emb') else None
        )

    if cfg.style == 'shared_ae':
        # Build SharedAETranslator using supervised and unsupervised embedding dims
        assert hasattr(cfg, 'sup_emb') and hasattr(cfg, 'unsup_emb'), "shared_ae requires sup_emb and unsup_emb in config"
        assert cfg.sup_emb in encoder_dims and cfg.unsup_emb in encoder_dims, (
            "encoder_dims must include both sup_emb and unsup_emb for shared_ae"
        )
        d_s = encoder_dims[cfg.sup_emb]
        d_t = encoder_dims[cfg.unsup_emb]
        # Prefer d_z, fall back to d_adapter or latent_dims, then default 512
        d_z = (
            getattr(cfg, 'd_z', None)
            or getattr(cfg, 'd_adapter', None)
            or getattr(cfg, 'latent_dims', None)
            or 512
        )
        hidden_dim = (
            getattr(cfg, 'd_hidden', None)
            or getattr(cfg, 'd_transform', None)
            or 1024
        )
        depth = getattr(cfg, 'depth', 3)
        return SharedAETranslator(d_s=d_s, d_t=d_t, d_z=d_z, hidden_dim=hidden_dim, depth=depth)

    if cfg.style == 'n_simple':
        transform = nn.Linear(cfg.d_adapter, cfg.d_adapter)
    elif cfg.style == 'n_double':
        transform = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.d_adapter, cfg.d_adapter),
            nn.SiLU(),
            nn.Linear(cfg.d_adapter, cfg.d_adapter),
            nn.SiLU(),
        )
    elif cfg.style == 'res_mlp':
        transform = MLPWithResidual(
            depth=cfg.transform_depth,
            in_dim=cfg.d_adapter, 
            hidden_dim=cfg.d_transform, 
            out_dim=cfg.d_adapter, 
            norm_style=cfg.norm_style,
            weight_init=cfg.weight_init,
        )
    elif cfg.style == 'n_ae':
        transform = nn.Sequential(
            nn.Linear(cfg.d_adapter, cfg.latent_dims),
            nn.ReLU(),
            nn.Linear(cfg.latent_dims, cfg.d_adapter)
        )
    elif cfg.style == 'unet':
        transform = UNetTransform(cfg.d_adapter, cfg.d_adapter)
    elif cfg.style == 'unet1d':
        transform = UNet1dTransform(cfg.d_adapter, cfg.d_adapter)
    else:
        raise ValueError(f"Unknown style: {cfg.style}")

    return TransformTranslator(
        encoder_dims=encoder_dims,
        d_adapter=cfg.d_adapter,
        d_hidden=cfg.d_hidden,
        transform=transform,
        weight_init=cfg.weight_init,
        depth=cfg.depth,
        use_small_output_adapters=cfg.use_small_output_adapters if hasattr(cfg, 'use_small_output_adapters') else False,
        norm_style=cfg.norm_style if hasattr(cfg, 'norm_style') else 'batch',
    )


def get_inverters(emb_flags, device='cpu'):
    assert isinstance(emb_flags, list)
    inverters = {}
    for emb_flag in emb_flags:
        assert emb_flag in ['gtr', 'gte']
        if emb_flag == "gtr":
            # inversion_model = InversionModel.from_pretrained("jxm/gtr-32-noise-0.001")
            # inversion_model = InversionModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_corrector")
            inversion_model = InversionModel.from_pretrained("ielabgroup/vec2text_gtr-base-st_inversion")

        elif emb_flag == 'gte':
            inversion_model = InversionModel.from_pretrained("jxm/gte-32-noise-0.001")
        inversion_model.eval()
        inverters[emb_flag] = inversion_model.to(device)
    return inverters

def read_args(argv):
    cfg = {}
    # Handle unknown arguments
    for arg in argv:
        if arg.startswith("--"):
            key = arg.lstrip("--")
            # Attempt to parse value as int, float, or leave as string
            try:
                value = int(argv[argv.index(arg) + 1])
            except ValueError:
                try:
                    value = float(argv[argv.index(arg) + 1])
                except ValueError:
                    value = argv[argv.index(arg) + 1]
            cfg[key] = value
    return cfg


def get_world_size() -> int:
    try:
        return torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return 1

def get_num_proc() -> int:
    """Return a reasonable number of worker processes per device.

    - Uses GPU count to split CPU workers across devices when GPUs are present.
    - Falls back to CPU-only safe behavior when no GPU is available.
    """
    gpu_count: int = torch.cuda.device_count()
    devices = max(1, gpu_count)
    try:
        total_cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        total_cpus = multiprocessing.cpu_count()
    # Ensure at least 1 worker
    return max(1, total_cpus // devices)
    

def load_translator_from_hf(model_id):
    if os.path.isdir(model_id):
        print("Loading weights from local directory")
        model_file = os.path.join(model_id, 'model.safetensors')
        config_file = os.path.join(model_id, 'config.json')
    else:
        model_file = hf_hub_download(
            repo_id=model_id,
            filename='model.safetensors',
        )
        config_file = hf_hub_download(
            repo_id=model_id,
            filename='config.json',
        )
    state_dict = load_file(model_file)
    with open(config_file) as f:
        cfg = json.load(f)
    cfg = SimpleNamespace(**cfg)
    translator = load_n_translator(cfg, cfg.encoder_dims)
    translator.load_state_dict(state_dict, strict=False)
    return translator


def exit_on_nan(loss: torch.Tensor) -> None:
    if torch.isnan(loss).any():
        print("Loss is NaN! exiting")
        exit(1)


def save_everything(cfg, translator, opt, gans, save_dir):
    torch.save(translator.state_dict(), os.path.join(save_dir, 'model.pt'))
    torch.save(opt.state_dict(), os.path.join(save_dir, 'opt.pt'))
    for i, gan in enumerate(gans):
        torch.save(gan.discriminator.state_dict(), os.path.join(save_dir, f'gan_{i}.pt'))
        torch.save(gan.discriminator_opt.state_dict(), os.path.join(save_dir, f'gan_opt_{i}.pt'))
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(cfg.__dict__, f)
