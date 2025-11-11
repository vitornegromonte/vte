import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate
from tqdm import tqdm
import wandb

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from translators.SharedAE import SharedAETranslator, compute_losses

from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import get_num_proc, read_args, exit_on_nan
from utils.streaming_utils import load_streaming_embeddings, process_batch
from utils.wandb_logger import Logger


def _resolve_train_config_name(stem: str) -> str:
    """Resolve a config stem to a concrete path under configs/ with fuzzy matching.

    Returns absolute file path to the matched TOML config.
    """
    # If a direct path is provided, respect it
    if stem.endswith('.toml') or os.path.sep in stem:
        abs_path = stem if os.path.isabs(stem) else os.path.join(os.getcwd(), stem)
        if os.path.exists(abs_path):
            return abs_path
        raise FileNotFoundError(f"Config path '{stem}' not found (resolved to {abs_path}).")

    base_dir = os.path.dirname(__file__)
    cfg_dir = os.path.join(base_dir, 'configs')
    # Try exact
    exact = os.path.join(cfg_dir, f"{stem}.toml")
    if os.path.exists(exact):
        return exact
    # Fuzzy: prefix then substring
    if os.path.isdir(cfg_dir):
        files = [f for f in os.listdir(cfg_dir) if f.endswith('.toml')]
        prefix_matches = [f for f in files if f.startswith(stem)]
        if len(prefix_matches) == 1:
            return os.path.join(cfg_dir, prefix_matches[0])
        if not prefix_matches:
            substr = [f for f in files if stem in f]
            if len(substr) == 1:
                return os.path.join(cfg_dir, substr[0])
            elif len(substr) > 1:
                raise FileNotFoundError(f"Ambiguous config stem '{stem}' in configs. Candidates: {substr}")
        elif len(prefix_matches) > 1:
            raise FileNotFoundError(f"Ambiguous config stem '{stem}' in configs. Candidates: {prefix_matches}")
    raise FileNotFoundError(f"Config '{stem}.toml' not found under {cfg_dir} (tried fuzzy match as well).")


def training_loop_shared_ae(
    save_dir,
    accelerator,
    translator,
    sup_dataloader,
    unsup_dataloader,
    sup_encs,
    unsup_enc,
    cfg,
    opt,
    scheduler,
    logger=None,
    max_num_batches=None,
):
    device = accelerator.device
    if logger is None:
        logger = Logger(dummy=True)

    dataloader_pbar = tqdm(
        zip(sup_dataloader, unsup_dataloader),
        total=min(len(sup_dataloader), len(unsup_dataloader)),
        desc="Training (shared_ae)"
    )
    model_save_dir = os.path.join(save_dir, 'model.pt')

    translator.train()

    # Coefficients with safe defaults
    lambda_rec  = getattr(cfg, 'lambda_rec', 1.0)
    lambda_cyc  = getattr(cfg, 'lambda_cyc', 1.0)
    lambda_dist = getattr(cfg, 'lambda_dist', 0.2)
    lambda_stab = getattr(cfg, 'lambda_stab', 0.1)
    lambda_geo  = getattr(cfg, 'lambda_geo', 0.05)
    sinkhorn_eps = getattr(cfg, 'sinkhorn_eps', 0.1)

    if accelerator.is_main_process:
        print(
            f"[SharedAE Coeffs] rec={lambda_rec} | cyc={lambda_cyc} | dist={lambda_dist} | "
            f"stab={lambda_stab} | geo={lambda_geo} | sinkhorn_eps={sinkhorn_eps}"
        )

    for i, (sup_batch, unsup_batch) in enumerate(dataloader_pbar):
        if max_num_batches is not None and i >= max_num_batches:
            print(f"Early stopping at {i} batches")
            break

        with accelerator.accumulate(translator), accelerator.autocast():
            sup_ins = process_batch(sup_batch, sup_encs, cfg.normalize_embeddings, device)
            unsup_ins = process_batch(unsup_batch, unsup_enc, cfg.normalize_embeddings, device)
            x = sup_ins[cfg.sup_emb]
            y = unsup_ins[cfg.unsup_emb]

            out = translator(x, y)

            total, losses = compute_losses(
                out,
                x,
                y,
                lambda_rec=lambda_rec,
                lambda_cyc=lambda_cyc,
                lambda_dist=lambda_dist,
                lambda_stab=lambda_stab,
                lambda_geo=lambda_geo,
                use_ot=True,
                ot_eps=sinkhorn_eps,
            )

            exit_on_nan(total)
            opt.zero_grad()
            accelerator.backward(total)
            accelerator.clip_grad_norm_(translator.parameters(), getattr(cfg, 'max_grad_norm', 1.0))
            opt.step()
            scheduler.step()

            # Log
            metrics = {f"loss/{k}": (v.item() if hasattr(v, 'item') else float(v)) for k, v in losses.items()}
            metrics["loss/total"] = total.item() if hasattr(total, 'item') else float(total)
            metrics["learning_rate"] = opt.param_groups[0]["lr"]

            metrics["loss_w/rec"] = lambda_rec * (losses['rec_s'].item() + losses['rec_t'].item())
            metrics["loss_w/cyc"] = lambda_cyc * (losses['cyc_z_s'].item() + losses['cyc_z_t'].item())
            metrics["loss_w/dist"] = lambda_dist * (losses['ot_s'].item() + losses['ot_t'].item())
            metrics["loss_w/stab"] = lambda_stab * losses['vic'].item()
            metrics["loss_w/geo"] = lambda_geo * (losses['lap'].item() + losses['triplet'].item())

            for k, v in metrics.items():
                logger.logkv(k, v)
            logger.dumpkvs(force=(hasattr(cfg, 'force_dump') and cfg.force_dump))
            dataloader_pbar.set_postfix({k: round(v, 4) for k, v in metrics.items() if 'total' in k or k.endswith('rec') or k.endswith('cyc')})

    with open(save_dir + 'config.toml', 'w') as f:
        toml.dump(cfg.__dict__, f)
    torch.save(accelerator.unwrap_model(translator).state_dict(), model_save_dir)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    if len(argv) < 2:
        raise SystemExit("Usage: python train_shared_ae.py <config_name_or_path> [--overrides]")

    cfg_path = _resolve_train_config_name(argv[1])
    cfg = toml.load(cfg_path)
    unknown_cfg = read_args(argv)
    # Flatten tables + apply overrides
    cfg = SimpleNamespace(**{**{k: v for d in cfg.values() for k, v in d.items()}, **unknown_cfg})

    # Mixed precision guard
    if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision != 'no' and cfg.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        cfg.mixed_precision = 'fp16'
        cfg.gradient_accumulation_steps = 1
        print("Note: bf16 is not available on this hardware! Reverting to fp16 and setting accumulation steps to 1.")

    # set seeds
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)

    use_val_set = hasattr(cfg, 'val_size')

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision != 'no' else None,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )
    accelerator.dataloader_config.dispatch_batches = False

    # Save dir and logger
    if hasattr(cfg, 'force_wandb_name') and cfg.force_wandb_name:
        save_dir = cfg.save_dir.format(cfg.wandb_name)
    else:
        cfg.wandb_name = ','.join([f"{k[0]}:{v}" for k, v in unknown_cfg.items()]) if unknown_cfg else cfg.wandb_name
        save_dir = cfg.save_dir.format(getattr(cfg, 'latent_dims', cfg.wandb_name))

    logger = Logger(
        project=cfg.wandb_project,
        name=cfg.wandb_name,
        dummy=(cfg.wandb_project is None) or not (cfg.use_wandb),
        config=cfg,
    )

    print("Running Experiment (SharedAE):", cfg.wandb_name)

    # Load encoders
    sup_encs = { cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None) }
    encoder_dims = { cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb]) }

    os.makedirs(save_dir, exist_ok=True)
    assert hasattr(cfg, 'unsup_emb') and cfg.sup_emb != cfg.unsup_emb
    unsup_enc = { cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None) }
    unsup_dim = { cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb]) }

    # Build SharedAE with both dims
    encoder_dims_shared = {**encoder_dims, **unsup_dim}
    translator = SharedAETranslator(
        d_s=encoder_dims_shared[cfg.sup_emb],
        d_t=encoder_dims_shared[cfg.unsup_emb],
        d_z=getattr(cfg, 'd_z', getattr(cfg, 'latent_dims', 512)),
        hidden_dim=getattr(cfg, 'd_hidden', getattr(cfg, 'd_transform', 1024)),
        depth=getattr(cfg, 'depth', 3),
        norm_style='batch',
    )

    cfg.num_params = sum(x.numel() for x in translator.parameters())
    print("Number of parameters:", cfg.num_params)
    print("Number of *trainable* parameters:", sum(p.numel() for p in translator.parameters() if p.requires_grad))
    print(translator)

    # Dataset
    num_workers = min(get_num_proc(), 8)
    dset = load_streaming_embeddings(cfg.dataset)
    print(f"Using {num_workers} workers and {len(dset)} datapoints")
    dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    dset = dset_dict["train"]
    valset = dset_dict["test"]

    assert hasattr(cfg, 'num_points') or hasattr(cfg, 'unsup_points')
    dset = dset.shuffle(seed=cfg.train_dataset_seed)
    if hasattr(cfg, 'num_points'):
        assert cfg.num_points > 0 and cfg.num_points <= len(dset) // 2
        supset = dset.select(range(cfg.num_points))
        unsupset = dset.select(range(cfg.num_points, cfg.num_points * 2))
    elif hasattr(cfg, 'unsup_points'):
        unsupset = dset.select(range(min(cfg.unsup_points, len(dset))))
        supset = dset.select(range(min(cfg.unsup_points, len(dset)), len(dset) - len(unsupset)))

    supset = MultiencoderTokenizedDataset(
        dataset=supset,
        encoders=sup_encs,
        n_embs_per_batch=cfg.n_embs_per_batch,
        batch_size=cfg.bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )
    unsupset = MultiencoderTokenizedDataset(
        dataset=unsupset,
        encoders=unsup_enc,
        n_embs_per_batch=1,
        batch_size=cfg.bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )

    sup_dataloader = DataLoader(
        supset,
        batch_size=cfg.bs,
        num_workers=num_workers // 2,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=None,
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    unsup_dataloader = DataLoader(
        unsupset,
        batch_size=cfg.bs,
        num_workers=num_workers // 2,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=None,
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )

    # Optimizer & scheduler
    opt = torch.optim.Adam(translator.parameters(), lr=cfg.lr, fused=False, betas=(0.5, 0.999))

    steps_per_epoch = len(supset) // cfg.bs
    total_steps = steps_per_epoch * cfg.epochs / cfg.gradient_accumulation_steps
    warmup_length = (cfg.warmup_length if hasattr(cfg, 'warmup_length') else 100)

    def lr_lambda(step):
        if step < warmup_length:
            return min(1, step / warmup_length)
        else:
            if hasattr(cfg, 'no_scheduler') and cfg.no_scheduler:
                return 1
            return 1 - (step - warmup_length) / max(1, total_steps - warmup_length)

    scheduler = LambdaLR(opt, lr_lambda=lr_lambda)

    # Prepare with accelerator
    translator, opt, scheduler = accelerator.prepare(translator, opt, scheduler)
    sup_dataloader, unsup_dataloader = accelerator.prepare(sup_dataloader, unsup_dataloader)

    max_num_epochs = int(np.ceil(cfg.epochs))
    forced_max_batches = getattr(cfg, 'max_num_batches', None)

    for epoch in range(max_num_epochs):
        max_num_batches = forced_max_batches
        print(f"Epoch", epoch, "max_num_batches", max_num_batches, "max_num_epochs", max_num_epochs)
        if max_num_batches is None and (epoch + 1 >= max_num_epochs):
            max_num_batches = max(1, (cfg.epochs - epoch) * len(supset) // cfg.bs)
            print(f"Setting max_num_batches to {max_num_batches}")

        training_loop_shared_ae(
            save_dir=save_dir,
            accelerator=accelerator,
            translator=translator,
            sup_dataloader=sup_dataloader,
            unsup_dataloader=unsup_dataloader,
            sup_encs=sup_encs,
            unsup_enc=unsup_enc,
            cfg=cfg,
            opt=opt,
            scheduler=scheduler,
            logger=logger,
            max_num_batches=max_num_batches,
        )

    with open(save_dir + 'config.toml', 'w') as f:
        toml.dump(cfg.__dict__, f)


if __name__ == "__main__":
    main()
