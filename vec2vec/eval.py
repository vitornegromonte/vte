import os
import random
import toml
import json
from sys import argv
from types import SimpleNamespace
from typing import Tuple, List

import accelerate

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# from eval import eval_model
from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.dist import get_rank
from utils.eval_utils import eval_loop_
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings


def _resolve_checkpoint(load_dir: str, ckpt: str | None = None, default_name: str = 'model.pt') -> str:
    """Resolve a checkpoint path.

    Priority:
      1. Explicit ckpt path if it exists (absolute or relative).
      2. If ckpt is a filename and exists inside load_dir.
      3. load_dir/default_name
      4. Any nested 'model.pt' files under load_dir (choose most recent mtime)
      5. Any *.pt files under load_dir (choose most recent mtime)
    """
    if ckpt:
        # Absolute or relative explicit path
        if os.path.isfile(ckpt):
            print(f"[eval] Using explicit checkpoint path: {ckpt}")
            return ckpt
        # Treat as relative filename inside load_dir
        rel_path = os.path.join(load_dir, ckpt)
        if os.path.isfile(rel_path):
            print(f"[eval] Using checkpoint inside load_dir: {rel_path}")
            return rel_path
        print(f"[eval] Warning: Provided ckpt '{ckpt}' not found directly; falling back to discovery.")

    direct = os.path.join(load_dir, default_name)
    if os.path.isfile(direct):
        print(f"[eval] Found direct checkpoint: {direct}")
        return direct

    # Walk and collect candidates
    model_pt_candidates = []
    other_pt_candidates = []
    for root, dirs, files in os.walk(load_dir):
        if default_name in files:
            model_pt_candidates.append(os.path.join(root, default_name))
        for f in files:
            if f.endswith('.pt') and f != default_name:
                other_pt_candidates.append(os.path.join(root, f))

    if model_pt_candidates:
        # pick most recent
        best = max(model_pt_candidates, key=lambda p: os.path.getmtime(p))
        print(f"[eval] Discovered {len(model_pt_candidates)} 'model.pt' candidates; choosing most recent: {best}")
        return best
    if other_pt_candidates:
        best = max(other_pt_candidates, key=lambda p: os.path.getmtime(p))
        print(f"[eval] Discovered {len(other_pt_candidates)} '*.pt' candidates; choosing most recent: {best}")
        return best

    raise FileNotFoundError(f"No checkpoint (.pt) found under load_dir={load_dir}. Provide ckpt=<file.pt> or ensure model.pt exists.")


def _find_checkpoint_config(weight_path: str, root_dir: str) -> dict:
    """Search upward from weight_path directory to root_dir for a config.json capturing
    original architecture hyperparameters. Returns parsed dict or empty dict.
    """
    cur = os.path.dirname(weight_path)
    root_dir = os.path.abspath(root_dir)
    out_cfg = {}
    while True:
        candidate = os.path.join(cur, 'config.json')
        if os.path.isfile(candidate):
            try:
                import json
                with open(candidate) as f:
                    raw = json.load(f)
                # Keep only possible architecture keys
                for k in ['d_hidden', 'd_z', 'depth', 'latent_dims']:
                    if k in raw:
                        out_cfg[k] = raw[k]
                break
            except Exception as e:
                print(f"[eval] Warning: failed to parse {candidate}: {e}")
                break
        parent = os.path.dirname(cur)
        if parent == cur or not os.path.commonpath([parent, root_dir]).startswith(root_dir):
            break
        cur = parent
    return out_cfg


class SharedAEAdapter(nn.Module):
    """Adapter to expose a SharedAETranslator with the same interface used by eval_utils.

    Provides:
      - forward(ins, include_reps=False) -> (recons, translations[, reps])
      - translate_embeddings(emb, in_name, out_name)
    """
    def __init__(self, ae_module, sup_flag: str, unsup_flag: str, normalize_embeddings: bool = True):
        super().__init__()
        self.ae = ae_module
        self.sup_flag = sup_flag
        self.unsup_flag = unsup_flag
        self.normalize = normalize_embeddings

    def translate_embeddings(self, embeddings: torch.Tensor, in_name: str, out_name: str) -> torch.Tensor:
        if in_name == self.sup_flag:
            z = self.ae.encode_s(embeddings)
            out = self.ae.decode_s(z) if out_name == self.sup_flag else self.ae.decode_t(z)
        elif in_name == self.unsup_flag:
            z = self.ae.encode_t(embeddings)
            out = self.ae.decode_t(z) if out_name == self.unsup_flag else self.ae.decode_s(z)
        else:
            raise ValueError(f"Unknown input flag '{in_name}' for SharedAEAdapter.")
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
        return out

    def forward(
        self,
        ins: dict[str, torch.Tensor],
        in_set: set[str] = None,
        out_set: set[str] = None,
        include_reps: bool = False,
        noise_level: float = 0.0,
    ):
        in_set = in_set if in_set is not None else ins.keys()
        out_set = out_set if out_set is not None else ins.keys()
        recons: dict[str, torch.Tensor] = {}
        translations: dict[str, dict[str, torch.Tensor]] = {}
        reps: dict[str, torch.Tensor] = {}

        for flag in in_set:
            if flag not in ins:
                continue
            emb = ins[flag]
            if self.training and noise_level > 0.0:
                emb = emb + torch.randn_like(emb) * noise_level
                emb = F.normalize(emb, p=2, dim=1)

            if flag == self.sup_flag:
                z = self.ae.encode_s(emb)
                rec = self.ae.decode_s(z)
            elif flag == self.unsup_flag:
                z = self.ae.encode_t(emb)
                rec = self.ae.decode_t(z)
            else:
                continue

            if self.normalize:
                rec = F.normalize(rec, p=2, dim=1)
            recons[flag] = rec
            if include_reps:
                reps[flag] = z

            for target_flag in out_set:
                if target_flag == flag:
                    continue
                out = self.translate_embeddings(emb, flag, target_flag)
                if target_flag not in translations:
                    translations[target_flag] = {}
                translations[target_flag][flag] = out

        if include_reps:
            return recons, translations, reps
        return recons, translations


def _flatten_toml_tables(toml_dict: dict) -> dict:
    """Flatten a standard multi-table training config (like train.py does)."""
    if any(isinstance(v, dict) for v in toml_dict.values()):
        # typical structure: {section_name: {k:v}}
        flat = {}
        for section, sub in toml_dict.items():
            if isinstance(sub, dict):
                flat.update(sub)
        return flat
    return toml_dict

def _load_config_from_arg(arg: str) -> Tuple[dict, str]:
    """Return (config_dict, source_type) where source_type is 'run_dir' or 'config'.

    Cases:
      1. arg is a directory containing config.toml -> treat as run directory.
      2. arg refers to a name in configs/<name>.toml -> treat as config template.
    """
    # Case 1: run directory
    if os.path.isdir(arg) and os.path.exists(os.path.join(arg, 'config.toml')):
        cfg_path = os.path.join(arg, 'config.toml')
        return toml.load(cfg_path), 'run_dir'
    # Case 2: config name
    candidate = os.path.join('configs', f'{arg}.toml')
    if os.path.exists(candidate):
        return _flatten_toml_tables(toml.load(candidate)), 'config'

    # Fuzzy / prefix matching fallback: allow passing a shortened stem like 'shared_ae'
    configs_dir = 'configs'
    if os.path.isdir(configs_dir):
        all_cfg_files = [f for f in os.listdir(configs_dir) if f.endswith('.toml')]
        stem_matches = [f for f in all_cfg_files if f.startswith(arg)]
        if len(stem_matches) == 1:
            match_path = os.path.join(configs_dir, stem_matches[0])
            print(f"[eval] Resolved config stem '{arg}' -> '{stem_matches[0]}'")
            return _flatten_toml_tables(toml.load(match_path)), 'config'
        # If no prefix match, try substring match
        if not stem_matches:
            substr_matches = [f for f in all_cfg_files if arg in f]
            if len(substr_matches) == 1:
                match_path = os.path.join(configs_dir, substr_matches[0])
                print(f"[eval] Resolved config substring '{arg}' -> '{substr_matches[0]}'")
                return _flatten_toml_tables(toml.load(match_path)), 'config'
            elif len(substr_matches) > 1:
                raise FileNotFoundError(
                    f"Ambiguous config stem '{arg}'. Possible matches: {substr_matches}. Please specify one exactly (without .toml)."
                )
        elif len(stem_matches) > 1:
            raise FileNotFoundError(
                f"Ambiguous config stem '{arg}'. Possible matches: {stem_matches}. Please specify one exactly (without .toml)."
            )

    raise FileNotFoundError(
        f"Could not interpret first argument '{arg}' as a run directory (with config.toml) or a config name under configs/."
    )

def _augment_with_cli(cfg_dict: dict, argv_list: List[str]) -> dict:
    """Augment config dict with CLI overrides.

    Supports two forms:
      --key value
      key=value
    (the latter is convenient when calling with python -m vec2vec.eval ...)
    """
    overrides = {}
    # Existing --key value style handled by read_args
    overrides.update(read_args(argv_list))
    for token in argv_list[2:]:  # skip script and first positional
        if '=' in token and not token.startswith('--'):
            k, v = token.split('=', 1)
            # Try int/float cast
            v_lower = v.lower()
            if v_lower in ('true', 'false'):
                v_cast = (v_lower == 'true')
            elif v_lower in ('none', 'null'):
                v_cast = None
            else:
                try:
                    v_cast = int(v)
                except ValueError:
                    try:
                        v_cast = float(v)
                    except ValueError:
                        v_cast = v
            overrides[k] = v_cast
    cfg_dict.update(overrides)
    return cfg_dict

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    if len(argv) < 2:
        raise SystemExit("Usage: python -m vec2vec.eval <run_dir | config_name> [--overrides] [key=value ...]")

    base_cfg_dict, source_type = _load_config_from_arg(argv[1])
    base_cfg_dict = _augment_with_cli(base_cfg_dict, argv)

    # If using a run directory and load_dir not explicitly passed, set load_dir to that directory.
    if source_type == 'run_dir' and 'load_dir' not in base_cfg_dict:
        base_cfg_dict['load_dir'] = os.path.join(argv[1], '')  # ensure trailing slash

    cfg = SimpleNamespace(**base_cfg_dict)

    if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        cfg.mixed_precision = 'fp16'
        print("Note: bf16 is not available on this hardware!")

    # set seeds
    random.seed(cfg.seed + get_rank())
    torch.manual_seed(cfg.seed + get_rank())
    np.random.seed(cfg.seed + get_rank())
    torch.cuda.manual_seed(cfg.seed + get_rank())

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None
    )
    # https://github.com/huggingface/transformers/issues/26548
    accelerator.dataloader_config.dispatch_batches = False

    dset = load_streaming_embeddings(cfg.dataset)

    sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)}
    assert hasattr(cfg, 'unsup_emb'), 'unsup_emb must be in config'
    assert cfg.sup_emb != cfg.unsup_emb, 'sup_emb and unsup_emb must differ'
    unsup_enc = {cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)}

    encoder_dims = {cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])}
    unsup_dim = {cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb])}

    # We'll delay final shared_ae build until after possible checkpoint config recovery
    translator = None
    if cfg.style != 'shared_ae':
        translator = load_n_translator(cfg, encoder_dims)
        translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])

    if cfg.style != 'identity':
        assert cfg.unsup_emb not in sup_encs
        if cfg.style != 'shared_ae':
            # Only translators using adapter dicts expose in_adapters/out_adapters
            if hasattr(translator, 'in_adapters'):
                assert cfg.unsup_emb in translator.in_adapters, "unsup_emb missing from in_adapters"
            if hasattr(translator, 'out_adapters'):
                assert cfg.unsup_emb in translator.out_adapters, "unsup_emb missing from out_adapters"
        else:
            # Shared AE builds a joint latent space; adapters are not used.
            print('[eval] SharedAE style: skipping adapter presence assertions.')
        if translator is not None:
            cfg.num_params = sum(x.numel() for x in translator.parameters())
            print("Number of parameters:", cfg.num_params)

    dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    dset = dset_dict["train"]
    valset = dset_dict["test"]

    # Provide sane defaults if split parameters are missing
    if not (hasattr(cfg, 'num_points') or hasattr(cfg, 'unsup_points')):
        # Default to using val_size items for the unsupervised slice
        # This mirrors the existing selection logic and keeps eval lightweight.
        setattr(cfg, 'unsup_points', getattr(cfg, 'val_size', 1024))
        print(f"[eval] No num_points/unsup_points specified; defaulting unsup_points={cfg.unsup_points} for evaluation.")
    dset = dset.shuffle(seed=cfg.train_dataset_seed)
    if hasattr(cfg, 'num_points'):
        assert cfg.num_points > 0 and cfg.num_points <= len(dset) // 2
        supset = dset.select(range(cfg.num_points))
        unsupset = dset.select(range(cfg.num_points, cfg.num_points + cfg.val_size))
    elif hasattr(cfg, 'unsup_points'):
        # Select up to val_size items for unsupervised slice
        unsup_n = min(getattr(cfg, 'unsup_points', 0), getattr(cfg, 'val_size', len(dset)))
        unsup_n = max(unsup_n, 0)
        unsup_n = min(unsup_n, len(dset))
        unsupset = dset.select(range(unsup_n))
        # Use the remainder of the dataset (or as many as available) as the supervised slice
        start_sup = min(getattr(cfg, 'unsup_points', 0), len(dset))
        end_sup = max(start_sup, len(dset) - len(unsupset))
        supset = dset.select(range(start_sup, end_sup))

    num_workers = get_num_proc()
    evalset = MultiencoderTokenizedDataset(
        dataset=supset if hasattr(cfg, 'flip') and cfg.flip else unsupset,
        encoders={ **unsup_enc, **sup_encs },
        n_embs_per_batch=2,
        batch_size=cfg.val_bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )
    evalloader = DataLoader(
        evalset,
        batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    evalloader = accelerator.prepare(evalloader)

    if cfg.style != 'identity':
        assert hasattr(cfg, 'load_dir'), "Specify load_dir=<run_dir>/ via CLI or ensured by run directory argument."
        ckpt_attr = getattr(cfg, 'ckpt', None)
        weight_path = _resolve_checkpoint(cfg.load_dir, ckpt=ckpt_attr)
        print(f"[eval] Loading weights: {weight_path}")
        # If shared_ae, attempt to recover architecture params before instantiation
        if cfg.style == 'shared_ae' and translator is None:
            recovered = _find_checkpoint_config(weight_path, cfg.load_dir)
            if recovered:
                # Map legacy key latent_dims to d_z
                if 'latent_dims' in recovered and 'd_z' not in recovered:
                    recovered['d_z'] = recovered['latent_dims']
                for k, v in recovered.items():
                    if getattr(cfg, k, None) != v:
                        print(f"[eval] Restoring {k}={v} from checkpoint config (was {getattr(cfg,k,None)})")
                        setattr(cfg, k, v)
            encoder_dims_shared = {**encoder_dims, **unsup_dim}
            translator = load_n_translator(cfg, encoder_dims_shared)
            cfg.num_params = sum(x.numel() for x in translator.parameters())
            print("Number of parameters:", cfg.num_params)
        checkpoint_state = torch.load(weight_path, map_location='cpu')
        try:
            translator.load_state_dict(checkpoint_state, strict=False)
            # Wrap SharedAE in adapter for eval_utils API compatibility
            if cfg.style == 'shared_ae':
                translator = SharedAEAdapter(translator, cfg.sup_emb, cfg.unsup_emb, normalize_embeddings=cfg.normalize_embeddings)
        except RuntimeError as e:
            if cfg.style == 'shared_ae' and 'size mismatch' in str(e):
                print('[eval][warn] Size mismatch detected loading SharedAETranslator. Attempting automatic architecture inference...')
                # Infer latent (d_z) and hidden (d_hidden) dimensions from encoder layer weight shapes
                ckpt_latent = None  # d_z
                ckpt_hidden = None  # hidden_dim
                # Collect candidate last-layer keys for encoders
                for k, v in checkpoint_state.items():
                    if k.startswith('E_s.layers') and k.endswith('.0.weight'):
                        # Identify layer index
                        try:
                            layer_idx = int(k.split('E_s.layers.')[1].split('.')[0])
                        except Exception:
                            continue
                        # Heuristic: if weight shape[0] (out) != weight shape[1] (in), out is likely latent dim, in is hidden
                        out_dim, in_dim = v.shape[0], v.shape[1]
                        # In architecture, last layer maps hidden -> d_z (latent), earlier layers map hidden->hidden
                        if out_dim != in_dim:
                            ckpt_latent = out_dim
                            ckpt_hidden = in_dim
                        else:
                            # Could still be last layer if latent == hidden; keep as fallback
                            ckpt_latent = out_dim
                            ckpt_hidden = in_dim
                    if ckpt_latent is not None and ckpt_hidden is not None:
                        break
                if ckpt_latent is None or ckpt_hidden is None:
                    print('[eval][error] Could not infer (d_z, d_hidden) from checkpoint; re-raising.')
                    raise
                # Current model dims
                try:
                    current_latent = translator.E_s.layers[-1][0].weight.shape[0]
                    current_hidden = translator.E_s.layers[0][0].weight.shape[0]
                except Exception:
                    current_latent = None
                    current_hidden = None
                cfg_latent = getattr(cfg, 'd_z', None)
                cfg_hidden = getattr(cfg, 'd_hidden', None)
                need_rebuild = False
                if cfg_latent != ckpt_latent or current_latent != ckpt_latent:
                    print(f"[eval] Detected latent dim mismatch: cfg/cur={cfg_latent}/{current_latent} vs ckpt={ckpt_latent}")
                    need_rebuild = True
                if cfg_hidden is not None and cfg_hidden != ckpt_hidden:
                    print(f"[eval] Detected hidden dim mismatch: cfg={cfg_hidden} vs ckpt={ckpt_hidden}")
                    need_rebuild = True
                if current_hidden is not None and current_hidden != ckpt_hidden:
                    print(f"[eval] Current model hidden dim {current_hidden} != ckpt hidden dim {ckpt_hidden}")
                    need_rebuild = True
                if not need_rebuild:
                    print('[eval] Dimension inference suggests rebuild not strictly necessary, but mismatch persisted earlier; forcing rebuild.')
                # Rebuild translator with inferred dims
                setattr(cfg, 'd_z', ckpt_latent)
                setattr(cfg, 'd_hidden', ckpt_hidden)
                encoder_dims_shared = {**encoder_dims, **unsup_dim}
                translator = load_n_translator(cfg, encoder_dims_shared)
                load_report = translator.load_state_dict(checkpoint_state, strict=False)
                # Wrap after successful load
                translator = SharedAEAdapter(translator, cfg.sup_emb, cfg.unsup_emb, normalize_embeddings=cfg.normalize_embeddings)
                if isinstance(load_report, tuple):
                    missing, unexpected = load_report
                    if missing:
                        print(f"[eval] Warning: missing keys after rebuild (showing up to 10): {missing[:10]}")
                    if unexpected:
                        print(f"[eval] Warning: unexpected keys after rebuild (showing up to 10): {unexpected[:10]}")
            elif 'size mismatch' in str(e):
                # Handle TransformTranslator (res_mlp/n_*) mismatches by inferring dims
                print('[eval][warn] Size mismatch detected loading TransformTranslator. Attempting automatic architecture inference...')
                ckpt_d_adapter = None
                ckpt_d_transform = None
                ckpt_d_hidden = None
                # transform first layer gives d_transform (out) and d_adapter (in)
                for k, v in checkpoint_state.items():
                    if k.startswith('transform.layers.0.0.weight'):
                        ckpt_d_transform = int(v.shape[0])
                        ckpt_d_adapter = int(v.shape[1])
                        break
                # in_adapter first layer gives d_hidden (out)
                if ckpt_d_hidden is None:
                    for k, v in checkpoint_state.items():
                        if k.startswith('in_adapters.') and k.endswith('layers.0.0.weight'):
                            ckpt_d_hidden = int(v.shape[0])
                            break
                # Log discovered values
                print(f"[eval] Inferred from ckpt: d_adapter={ckpt_d_adapter}, d_transform={ckpt_d_transform}, d_hidden={ckpt_d_hidden}")
                # Update cfg if values were inferred
                if ckpt_d_adapter is not None:
                    setattr(cfg, 'd_adapter', ckpt_d_adapter)
                if ckpt_d_transform is not None:
                    setattr(cfg, 'd_transform', ckpt_d_transform)
                if ckpt_d_hidden is not None:
                    setattr(cfg, 'd_hidden', ckpt_d_hidden)
                # Rebuild translator and retry load
                translator = load_n_translator(cfg, encoder_dims)
                translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])
                load_report = translator.load_state_dict(checkpoint_state, strict=False)
                if isinstance(load_report, tuple):
                    missing, unexpected = load_report
                    if missing:
                        print(f"[eval] Warning: missing keys after rebuild (showing up to 10): {missing[:10]}")
                    if unexpected:
                        print(f"[eval] Warning: unexpected keys after rebuild (showing up to 10): {unexpected[:10]}")
            else:
                raise

    # Ensure SharedAE is wrapped even if earlier wrapping was skipped
    if cfg.style == 'shared_ae' and not isinstance(translator, SharedAEAdapter):
        translator = SharedAEAdapter(translator, cfg.sup_emb, cfg.unsup_emb, normalize_embeddings=cfg.normalize_embeddings)
    translator = accelerator.prepare(translator)
    # inverters = get_inverters(["gtr"], accelerator.device)
    inverters = None

    with torch.no_grad():
        translator.eval()
        val_res = {}
        recons, trans, heatmap_dict, text_recons, text_trans, _ =\
            eval_loop_(
                cfg,
                translator,
                {**sup_encs, **unsup_enc},
                evalloader,
                inverters=inverters,
                device=accelerator.device
            )
        val_res['recons'] = {}
        for flag, res in recons.items():
            for k, v in res.items():
                if k == 'cos':
                    val_res['recons'][f"rec_{flag}_{k}"] = v

        val_res['trans'] = {}
        for target_flag, d in trans.items():
            for flag, res in d.items():
                for k, v in res.items():
                    if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                        continue
                    val_res['trans'][f"{flag}_{target_flag}_{k}"] = v

        val_res['heatmap'] = {}
        if len(heatmap_dict) > 0:
            for k,v in heatmap_dict.items():
                if v.__class__.__name__ == 'Figure':
                    continue
                else:
                    val_res['heatmap'][f"{k} (avg. {cfg.top_k_batches} batches)"] = v
        
        val_res['text_recons'] = {}
        if len(text_recons) > 0:
            for flag, res in text_recons.items():
                for k,v in res.items():
                    val_res['text_recons'][f"text_{k}"] = v

        val_res['text_trans'] = {}
        if len(text_trans) > 0:
            for target_flag, d in text_trans.items():
                for flag, res in d.items():
                    for k, v in res.items():
                        if flag == cfg.unsup_emb and target_flag == cfg.unsup_emb:
                            continue
                        val_res['text_trans'][f"{flag}_{target_flag}_{k}"] = v

        # Optional: print a short summary to stdout for quick inspection
        if getattr(cfg, 'print_summary', False):
            def _print_section(title: str, dct: dict, limit: int = 8):
                if not dct:
                    return
                print(f"[eval][summary] {title} ({len(dct)} items)")
                shown = 0
                for k, v in dct.items():
                    print(f"  - {k}: {v}")
                    shown += 1
                    if shown >= limit:
                        remaining = len(dct) - shown
                        if remaining > 0:
                            print(f"  ... (+{remaining} more)")
                        break
            _print_section('reconstruction', val_res.get('recons', {}))
            _print_section('translation', val_res.get('trans', {}))
            _print_section('heatmap', val_res.get('heatmap', {}))
            _print_section('text_recons', val_res.get('text_recons', {}))
            _print_section('text_trans', val_res.get('text_trans', {}))

    if cfg.style == 'identity':
        fnm = f'results/baseline_{cfg.dataset.replace("/", "_")}_{cfg.unsup_emb}_{cfg.sup_emb}.json'
    elif hasattr(cfg, 'flip') and cfg.flip:
        fnm = f'results/{cfg.dataset.replace("/", "_")}_{cfg.sup_emb}_{cfg.unsup_emb}_ood.json'
    else:
        fnm = f'results/{cfg.dataset.replace("/", "_")}_{cfg.unsup_emb}_{cfg.sup_emb}.json'
    # Ensure results directory exists
    results_dir = os.path.dirname(fnm) or '.'
    os.makedirs(results_dir, exist_ok=True)
    with open(fnm, 'w') as f:
        # human readable
        f.write(json.dumps(val_res, indent=4))
    if getattr(cfg, 'print_summary', False):
        print(f"[eval] Wrote full results to {fnm}")


if __name__ == "__main__":
    main()