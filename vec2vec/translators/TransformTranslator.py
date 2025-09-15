import torch
import random
from torch import nn
from translators.MLPWithResidual import MLPWithResidual
from translators.AbsNTranslator import AbsNTranslator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformTranslator(AbsNTranslator):
    def __init__(
        self,
        encoder_dims: dict[str, int],
        d_adapter: int,
        d_hidden: int,
        transform: nn.Module,
        weight_init: str = 'kaiming',
        depth: int = 3,
        normalize_embeddings: bool = True,
        norm_style: str = 'layer',
        use_small_output_adapters: bool = False,
    ):
        super().__init__(encoder_dims, d_adapter, depth)

        self.d_hidden = d_hidden
        self.use_small_output_adapters = use_small_output_adapters
        self.norm_style = norm_style
        self.transform = transform
        self.weight_init = weight_init
        for flag, dims in encoder_dims.items():
            in_adapter, out_adapter = self._make_adapters(dims)
            self.in_adapters[flag] = in_adapter
            self.out_adapters[flag] = out_adapter
        self.normalize_embeddings = normalize_embeddings

    def translate_embeddings(
        self, embeddings: torch.Tensor, in_name: str, out_name: str,
    ) -> torch.Tensor:
        in_adapter = self.in_adapters[in_name]
        latents = self._get_latents(emb=embeddings, in_adapter=in_adapter)
        return self._out_project(latents,  self.out_adapters[out_name])

    def add_encoders(self, encoder_dims: dict[str, int], overwrite_embs: list[str] = None):
        for flag, dims in encoder_dims.items():
            if flag in self.in_adapters and (overwrite_embs is None or flag not in overwrite_embs):
                print(f"Skipping {flag}...")
                continue
            in_adapter, out_adapter = self._make_adapters(dims)
            self.in_adapters[flag] = in_adapter
            self.out_adapters[flag] = out_adapter

    def _make_adapters(self, dims):
        assert dims is not None
        return (
            MLPWithResidual(self.depth, dims, self.d_hidden, self.transform.in_dim, self.norm_style, weight_init=self.weight_init),
            MLPWithResidual(self.depth, self.transform.out_dim, self.d_hidden, dims, self.norm_style, weight_init=self.weight_init),
        )

    def _get_latents(self, emb: torch.Tensor, in_adapter: nn.Module) -> torch.Tensor:
        z = in_adapter(emb)
        return self.transform(z)

    def _out_project(self, emb: torch.Tensor, out_adapter: nn.Module) -> torch.Tensor:
        out = out_adapter(emb)
        if self.normalize_embeddings:
            out = out / out.norm(dim=1, keepdim=True)
        return out

    def forward(
        self,
        ins: dict[str, torch.Tensor],
        in_set: set[str] = None,
        out_set: set[str] = None,
        include_reps: bool = False,
        noise_level: float = 0.0,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        in_set = in_set if in_set is not None else ins.keys()
        out_set = out_set if out_set is not None else ins.keys()

        recons = {}
        translations = {}
        reps = recons.copy()

        for flag in in_set:
            noisy_emb = ins[flag]
            if self.training and noise_level > 0.0:
                noisy_emb += torch.randn_like(noisy_emb, device=noisy_emb.device) * noise_level
                noisy_emb = noisy_emb / noisy_emb.norm(p=2, dim=1, keepdim=True) # TODO check bool
            noisy_rep = self._get_latents(noisy_emb, self.in_adapters[flag])
            if include_reps:
                reps[flag] = noisy_rep
            for target_flag in out_set:
                if target_flag == flag:
                    recons[flag] = self._out_project(noisy_rep, self.out_adapters[target_flag])
                else:
                    if target_flag not in translations: translations[target_flag] = {}
                    translations[target_flag][flag] = self._out_project(noisy_rep, self.out_adapters[target_flag])

        if include_reps:
            return recons, translations, reps
        else:
            return recons, translations
