import os
import sys
import subprocess
import toml
from copy import deepcopy

# Simple orchestrator to run SharedAE ablations by toggling loss coefficients one-at-a-time.
# It generates ephemeral configs under runs/ablate_shared_ae/<name>.toml and invokes train.py.

BASE_CONFIG_STEM = os.environ.get("BASE_SHARED_AE_CONFIG", "shared_ae_tiny")
RUNS_DIR = os.path.join(os.path.dirname(__file__), 'runs', 'ablate_shared_ae')
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), 'configs')

# Default coefficient keys to ablate
DEFAULT_ABLATE_KEYS = [
    # (key, zero_value, pretty label)
    ("lambda_rec", 0.0, "no_rec"),
    ("lambda_cyc", 0.0, "no_cyc"),
    ("lambda_dist", 0.0, "no_dist"),
    ("lambda_stab", 0.0, "no_stab"),
    ("lambda_geo", 0.0, "no_geo"),
]

# Optional fast mode knobs
FAST_OVERRIDES = {
    # Limit batches per epoch for quick ablations
    "max_num_batches": int(os.environ.get("ABLATE_MAX_BATCHES", "128")),
    # Smaller val set for snappier eval, if present in config
    "val_size": int(os.environ.get("ABLATE_VAL_SIZE", "2048")),
}

# Force a fixed number of epochs for comparability across ablations (default: 3)
ABLATE_EPOCHS = int(os.environ.get("ABLATE_EPOCHS", "3"))
# Toggle post-training evaluation (1/true to enable, 0/false to disable)
DO_EVAL = os.environ.get("ABLATE_EVAL", "1").lower() not in {"0", "false", "no"}


def resolve_config_path(stem):
    # Prefix/prefix+substring resolution to mirror train.py helper
    exact = os.path.join(CONFIGS_DIR, f"{stem}.toml")
    if os.path.exists(exact):
        return exact
    files = [f for f in os.listdir(CONFIGS_DIR) if f.endswith('.toml')]
    prefix_matches = [f for f in files if f.startswith(stem)]
    if len(prefix_matches) == 1:
        return os.path.join(CONFIGS_DIR, prefix_matches[0])
    if not prefix_matches:
        substr = [f for f in files if stem in f]
        if len(substr) == 1:
            return os.path.join(CONFIGS_DIR, substr[0])
        elif len(substr) > 1:
            raise FileNotFoundError(f"Ambiguous config stem '{stem}'. Candidates: {substr}")
    elif len(prefix_matches) > 1:
        raise FileNotFoundError(f"Ambiguous config stem '{stem}'. Candidates: {prefix_matches}")
    raise FileNotFoundError(f"Config '{stem}.toml' not found under {CONFIGS_DIR}")


def _find_key_section(cfg: dict, key: str):
    """Return (section_name or None) where key currently exists in cfg.
    cfg is a nested dict from toml.load.
    """
    for section, sub in cfg.items():
        if isinstance(sub, dict) and key in sub:
            return section
    return None


def _get_coeff(cfg: dict, key: str, default=None):
    sec = _find_key_section(cfg, key)
    if sec is not None:
        return cfg[sec].get(key, default)
    # common location for SharedAE
    return cfg.get('translator', {}).get(key, default)


def _set_coeff(cfg: dict, key: str, value):
    sec = _find_key_section(cfg, key)
    if sec is not None:
        cfg[sec][key] = value
    else:
        cfg.setdefault('translator', {})[key] = value


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)

    base_cfg_path = resolve_config_path(BASE_CONFIG_STEM)
    base_cfg = toml.load(base_cfg_path)

    # Flatten sections for mutation
    flat = {}
    for section, sub in base_cfg.items():
        if isinstance(sub, dict):
            flat.update(sub)
        else:
            flat[section] = sub

    # Ensure style is shared_ae
    if flat.get("style") != "shared_ae":
        # Some configs may store under [translator]
        if base_cfg.get("translator", {}).get("style") != "shared_ae":
            raise SystemExit("Base config is not a shared_ae config.")

    # Bake in faster defaults if requested
    for k, v in FAST_OVERRIDES.items():
        flat[k] = v

    # Determine which keys to ablate (env override: ABLATE_LOSSES="lambda_rec,lambda_cyc,...")
    losses_env = os.environ.get("ABLATE_LOSSES")
    if losses_env:
        chosen_names = [s.strip() for s in losses_env.split(',') if s.strip()]
        # Map into (key, zero, label)
        name_to_tpl = {k: (k, 0.0, f"no_{k.split('lambda_')[-1]}") for k, _, _ in DEFAULT_ABLATE_KEYS}
        ablate_list = [name_to_tpl.get(n, (n, 0.0, f"no_{n}")) for n in chosen_names]
    else:
        ablate_list = DEFAULT_ABLATE_KEYS

    # Mode selection: drop (zero one), only (zero all others), both, or baseline only
    mode = os.environ.get("ABLATE_MODE", "drop").lower()
    if mode not in {"drop", "only", "both", "baseline"}:
        raise SystemExit("ABLATE_MODE must be one of: drop, only, both, baseline")

    def _write_and_run(cfg_dict: dict, label: str):
        cfg_dict.setdefault("logging", {})
        base_name = cfg_dict["logging"].get("wandb_name", "shared_ae")
        cfg_dict["logging"]["wandb_name"] = f"{base_name},{label}"
        cfg_dict["logging"]["save_dir"] = cfg_dict["logging"].get("save_dir", "./runs/{}/")
        # Force a comparable number of epochs for all runs
        _set_coeff(cfg_dict, "epochs", ABLATE_EPOCHS)
        out_path = os.path.join(RUNS_DIR, f"{BASE_CONFIG_STEM}-{label}.toml")
        with open(out_path, 'w') as f:
            toml.dump(cfg_dict, f)
        print(f"[ablate] Wrote {out_path}")
        # Choose module path based on cwd: if running inside vec2vec, use 'train'; otherwise 'vec2vec.train'
        base_dir = os.path.dirname(__file__)
        module_train = "train" if os.path.basename(base_dir) == "vec2vec" else "vec2vec.train"
        cmd = [sys.executable, "-m", module_train, os.path.splitext(os.path.basename(out_path))[0]]
        print("[ablate] Running:", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=base_dir)
        except subprocess.CalledProcessError as e:
            print(f"[ablate] Run failed for {label}: {e}")
            return

        # After successful training, automatically run evaluation on the produced checkpoint
        if DO_EVAL:
            # Resolve run directory from wandb_name and save_dir template (assumed ./runs/{}/)
            wandb_name = cfg_dict["logging"]["wandb_name"]
            # Use the same base_dir as training (vec2vec directory)
            run_dir = os.path.join(base_dir, "runs", wandb_name)
            module_eval = "eval" if os.path.basename(base_dir) == "vec2vec" else "vec2vec.eval"
            eval_cmd = [sys.executable, "-m", module_eval, os.path.abspath(run_dir), "print_summary=true"]
            print("[ablate] Evaluating:", " ".join(eval_cmd))
            try:
                subprocess.run(eval_cmd, check=True, cwd=base_dir)
            except subprocess.CalledProcessError as e:
                print(f"[ablate] Eval failed for {label}: {e}")

    # Always bake fast overrides (if any) into a working copy for derivation
    base_for_runs = deepcopy(base_cfg)
    for k, v in FAST_OVERRIDES.items():
        _set_coeff(base_for_runs, k, v)
    # Also force epochs for all derived configs
    _set_coeff(base_for_runs, "epochs", ABLATE_EPOCHS)

    # Baseline (no changes)
    if mode in {"baseline", "both"}:
        _write_and_run(deepcopy(base_for_runs), "baseline")
        if mode == "baseline":
            return

    # Drop-one and/or Only-one
    for key, zero, label in ablate_list:
        if mode in {"drop", "both"}:
            cfg_drop = deepcopy(base_for_runs)
            _set_coeff(cfg_drop, key, zero)
            _write_and_run(cfg_drop, label)
        if mode in {"only", "both"}:
            cfg_only = deepcopy(base_for_runs)
            # zero all ablated keys except the chosen one
            for k2, _, _ in ablate_list:
                if k2 != key:
                    _set_coeff(cfg_only, k2, 0.0)
            # keep the chosen key at its base value
            base_val = _get_coeff(base_cfg, key, None)
            if base_val is None:
                # choose a reasonable default if missing
                base_val = 1.0
            _set_coeff(cfg_only, key, base_val)
            _write_and_run(cfg_only, f"only_{key.split('lambda_')[-1]}")


if __name__ == "__main__":
    main()
