import time
import glob
import json
import gzip
import random
import dataclasses
from types import SimpleNamespace
from typing import Dict, Tuple, Optional

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.profiler import profile, ProfilerActivity, record_function
import torch.fx as fx
from torch.utils import checkpoint

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

CUDA_LAUNCH_BLOCKING=1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = torch.manual_seed(42)

def add_residual(input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Add residual connection with dimension matching."""
    if input_x.shape[1] != x.shape[1]:
        # This is a warning sign! Residuals should ideally have matching dims.
        # For simplicity, we'll implement the original logic but be aware of it.
        if input_x.shape[1] < x.shape[1]:
            padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device, dtype=x.dtype)
            input_x = torch.cat([input_x, padding], dim=1)
        elif input_x.shape[1] > x.shape[1]:
            input_x = input_x[:, :x.shape[1]]
    return x + input_x
  
class MLP(nn.Module):
    """
    A generic, configurable Multi-Layer Perceptron (MLP).
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        depth: int = 3,
        norm_style: str = 'layer',
        weight_init: str = 'kaiming',
        dropout_rate: float = 0.1,
        residual: bool = False,
        output_norm: bool = False
    ):
        """
        Initializes the MLP.

        Args:
            input_dim (int): Dimension of the input tensor.
            output_dim (int): Dimension of the output tensor.
            hidden_dim (int, optional): Dimension of hidden layers. If None, defaults to input_dim.
            depth (int): Total number of layers.
            norm_style (str): Normalization style ('layer' or 'batch').
            weight_init (str): Weight initialization scheme ('kaiming', 'xavier', 'orthogonal').
            dropout_rate (float): Dropout probability.
            residual (bool): If True, adds residual connections.
            output_norm (bool): If True, applies LayerNorm to the final output.
        """
        super().__init__()
        assert depth >= 1, "Depth must be at least 1."

        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.residual = residual

        # Paper says it used LayerNorm only but I want this class to be as generic as possible
        if norm_style == 'batch':
            norm_layer = nn.BatchNorm1d
        elif norm_style == 'layer':
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm style: {norm_style}")

        self.layers = nn.ModuleList()
        current_dim = input_dim

        for i in range(depth):
            is_last_layer = (i == depth - 1)
            target_dim = output_dim if is_last_layer else hidden_dim

            block = [nn.Linear(current_dim, target_dim)]

            if not is_last_layer:
                block.extend([
                    nn.SiLU(),
                    norm_layer(target_dim),
                    nn.Dropout(p=dropout_rate)
                ])

            self.layers.append(nn.Sequential(*block))
            current_dim = target_dim

        if output_norm:
            self.output_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        else:
            self.output_norm = None

        self.initialize_weights(weight_init)

    def initialize_weights(self, weight_init: str = 'kaiming'):
      """
      Initializes the weights of the MLP.

      Args:
          weight_init (str): Weight initialization scheme ('kaiming', 'xavier', 'orthogonal').
      """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                elif weight_init == 'kaiming':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif weight_init == 'orthogonal':
                    nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            input_x = x
            x = layer(x)
            if self.residual:
                x = add_residual(input_x, x)

        if self.output_norm is not None:
            x = self.output_norm(x)
        return x

class Adapter(nn.Module):
  """
  A generic adapter to project a tensor from an input dimension to a target dimension.

  """
  def __init__(self, input_dim: int, target_dim: int, dropout_rate: float):
    """
    Initializes the adapter

    Args:
      input_dim (int): The input dimension of the tensor.
      target_dim (int): The target dimension of the tensor.
      dropout_rate (float): The dropout rate to use.
    """
    super().__init__()
    self.input_dim = input_dim
    self.target_dim = target_dim
    self.droupt_rate = dropout_rate

    if input_dim == target_dim:
      self.adapter == nn.Identity()
    else:
      self.adapter = nn.Sequential(
          nn.Linear(input_dim, target_dim),
          nn.LayerNorm(target_dim),
          nn.SiLU(),
          nn.Dropout(dropout_rate)
      )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.adapter(x)
    
class Translator(nn.Module):
  """
  Main translator backbone
  """

  def __init__(self, sup_dim: int, hidden_dim: int, unsup_dim: int, depth: int):
    """
    Initializes the translator x  network.

    Args:
      sup_dim (int): The dimension of the supervised embeddings. (Supervised == known encoder (M_2))
      unsup_dim (int): The dimension of the unsupervised embeddings. (Unsupervised == unknown encoder (M_1))
      hidden_dim (int): The hidden dimension for the MLP.
      depth (int): The number of layers.
    """
    super().__init__()
    self.sup_dim = sup_dim
    self.unsup_sim = unsup_dim
    self.hidden_dim = hidden_dim
    self.depth = depth

    self.latent_dim = max(sup_dim, unsup_dim) # To avoid losing info, the latent space dimensions are defined as the max of the unsupervised and the supervised spaces.

    self.sup_input_adapter = Adapter(sup_dim, self.latent_dim, dropout_rate=0.1)
    self.unsup__input_adapter = Adapter(unsup_dim, self.latent_dim, dropout_rate=0.1)

    self.translator = MLP(
        input_dim = self.latent_dim,
        hidden_dim = hidden_dim,
        output_dim = self.latent_dim,
        depth = depth,
        norm_style = 'layer',
        weight_init = 'kaiming',
        dropout_rate = 0.1,
        residual = True,
        output_norm = False
    )

    self.sup_output_adapter = Adapter(self.latent_dim, sup_dim, dropout_rate=0.1)
    self.unsup_output_adapter = Adapter(self.latent_dim, unsup_dim, dropout_rate=0.1)

  def forward(self, x: torch.Tensor, source_type: str) -> torch.Tensor:
    """
    Forward pass of the translator.

    Args:
      x (torch.Tensor): The input embedding tensor.
      source_type (str): The source domain ('supervised' or 'unsupervised').

    Returns:
      torch.Tensor: The translated embedding in the opposite domain.
    """

    # assert source_type in ['sup', 'unsup'], f"Unknown source_type: {source_type}. Must be 'sup' or 'unsup'."

    if source_type == 'sup':
        # Path: sup -> latent -> unsup
        x = self.sup_input_adapter(x)
        x = self.translator(x)
        return self.unsup_output_adapter(x)

    elif source_type == 'unsup':
        # Path: unsup -> latent -> sup
        x = self.unsup_input_adapter(x)
        x = self.translator(x)
        return self.sup_output_adapter(x)
    else:
        raise ValueError(f"Unknown source_type: {source_type}. Must be 'sup' or 'unsup'.")

  def reconstruct(self, x: torch.Tensor, source_type: str) -> torch.Tensor:
    """
    Reconstruct the input embedding in the opposite domain.

    Args:
      x (torch.Tensor): The input embedding tensor.
      source_type (str): The source domain ('supervised' or 'unsupervised').
    """

    translated = self.forward(x, source_type)
    if source_type == 'sup':
        return self.forward(translated, 'unsup') # unsup -> sup
    elif source_type == 'unsup':
       return self.forward(translated, 'sup') # sup -> unsup

class Vec2Vec(nn.Module):
  """
  Space-specific adapters and the shared neural backbone
  """

  def __init__(self, sup_dim: int, unsup_dim: int , latent_dim: int, hidden_dim: int, depth:int ):
    super().__init__()

    # Input Adapters A1 and A2
    self.adapter_A1 = Adapter(sup_dim, latent_dim, droupt_rate=0.1)
    self.adapter_A2 = Adapter(unsup_dim, latent_dim, droupt_rate=0.1)

    # Translator backbone T
    self.translator = Translator(
        sup_dim = sup_dim,
        unsup_dim = unsup_dim,
        hidden_dim = hidden_dim,
        depth = depth
    )

    # Output Adapters B1 and B2
    self.adapter_B1 = Adapter(latent_dim, unsup_dim, droupt_rate=0.1)
    self.adapter_B2 = Adapter(latent_dim, sup_dim, droupt_rate=0.1)

  def translate_1_to_2(self, x: torch.Tensor) -> torch.Tensor:
    """
    Translate a vector from domain 1 to domain 2.

    Args:
      x (torch.Tensor): The input vector in domain 1.

    Returns:
      torch.Tensor: The translated vector in domain 2.
    """
    return self.adapter_B2(self.translator(self.adapter_A1(x)))

  def translate_2_to_1(self, x: torch.Tensor) -> torch.Tensor:
    """
    Translate a vector from domain 2 to domain 1.

    Args:
      x (torch.Tensor): The input vector in domain 2.

    Returns:
      torch.Tensor: The translated vector in domain 1.
    """
    return self.adapter_B1(self.translator(self.adapter_A2(x)))

  def reconstruct_1(self, x: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct a vector from domain 1.

    Args:
      x (torch.Tensor): The input vector in domain 1.

    Returns:
      torch.Tensor: The reconstructed vector in domain 1.
    """
    return self.adapter_B1(self.translator.reconstruct(self.adapter_A1(x)))

  def reconstruct_2(self, x: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct a vector from domain 2.

    Args:
      x (torch.Tensor): The input vector in domain 2.

    Returns:
      torch.Tensor: The reconstructed vector in domain 2.
    """
    return self.adapter_B2(self.translator.reconstruct(self.adapter_A2(x)))
  
  
class Discriminator(nn.Module):
  """
    A discriminator to classify embeddings in adversarial training.
  """
  def __init__(self, latent_dim: int, disc_dim: int, depth: int):
    super().__init__()

    self.latent_dim = latent_dim
    self.disc_dim = disc_dim
    self.depth = depth

    assert depth >= 1, "Depth must be at least 1"

    self.discriminator = MLP(
        input_dim = self.latent_dim,
        hidden_dim = self.disc_dim,
        output_dim = 1,
        depth = depth,
        norm_style = 'layer',
        weight_init = 'kaiming',
        dropout_rate = 0.1,
        residual = False,
        output_norm = False
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.discriminator(x)

# TO-DO: Refatorar, acho que tem algo de errado.
class VanillaGAN:
  """
  A vanilla GAN implementation, as proposed by Goodfellow et al. (2014).
  """
  def __init__(

      self,
      generator: nn.Module,
      discriminator: nn.Module,
      generator_opt: optim.Optimizer,
      discriminator_opt: optim.Optimizer,
      config: SimpleNamespace
      ):

      self.generator = generator
      self.discriminator = discriminator
      self.generator_opt = generator_opt
      self.discriminator_opt = discriminator_opt
      self.config = config

  def set_discriminator_requires_grad(self, rg: bool):
    """

    """
        for param in self.discriminator.parameters():
            param.requires_grad = rg

  def compute_gradient_penalty(self, d_out: torch.Tensor, d_in: torch.Tensor) -> torch.Tensor:
      """
      Computes the gradient penalty for the discriminator.
      """
      gradients = torch.autograd.grad(
          outputs=d_out.sum(), inputs=d_in,
          create_graph=True, retain_graph=True,
      )[0]
      return gradients.pow(2).reshape(gradients.shape[0], -1).sum(1).mean()

  def _step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> Tuple:

      real_data = real_data.detach().requires_grad_(True)

      d_real_logits = self.discriminator(real_data)
      d_fake_logits = self.discriminator(fake_data)

      real_labels = torch.full_like(d_real_logits, 1.0 - self.cfg.smooth)
      fake_labels = torch.full_like(d_fake_logits, 0.0)

      disc_loss = (F.binary_cross_entropy_with_logits(d_real_logits, real_labels) +
                    F.binary_cross_entropy_with_logits(d_fake_logits, fake_labels)) / 2

      disc_acc_real = (d_real_logits.sigmoid() > 0.5).float().mean().item()
      disc_acc_fake = (d_fake_logits.sigmoid() < 0.5).float().mean().item()

      r1_penalty = self.compute_gradient_penalty(d_out=d_real_logits, d_in=real_data)
      total_disc_loss = disc_loss + (r1_penalty * self.cfg.r1_penalty_coeff)

      self.discriminator_opt.zero_grad()
      self.accelerator.backward(total_disc_loss)
      if hasattr(self.cfg, 'max_grad_norm'):
          self.accelerator.clip_grad_norm_(self.discriminator.parameters(), self.cfg.max_grad_norm)
      self.discriminator_opt.step()

      return r1_penalty.detach(), disc_loss.detach(), disc_acc_real, disc_acc_fake

  def _step_generator(self, fake_data: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """
    Computes the loss for the generator.
    """
    d_fake_logits = self.discriminator(fake_data)
    real_labels = torch.full_like(d_fake_logits, 1.0)
    gen_loss = F.binary_cross_entropy_with_logits(d_fake_logits, real_labels)
    gen_acc = (d_fake_logits.sigmoid() > 0.5).float().mean().item()
    return gen_loss, gen_acc

  def step(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> Tuple:
      """Orchestrates a full adversarial step for one D."""
      self.generator.eval()
      self.discriminator.train()
      self.set_discriminator_requires_grad(True)
      r1_penalty, disc_loss, disc_acc_real, disc_acc_fake = self._step_discriminator(
          real_data=real_data.detach(), fake_data=fake_data.detach()
      )
      self.generator.train()
      self.discriminator.eval()
      self.set_discriminator_requires_grad(False)
      gen_loss, gen_acc = self._step_generator(fake_data=fake_data)

      return r1_penalty, disc_loss, gen_loss, disc_acc_real, disc_acc_fake, gen_acc

def reconstruction_loss(original_embeddings, reconstructed_embeddings):
    """
    Computes reconstruction loss based on cosine similarity.
    """
    return 1 - F.cosine_similarity(original_embeddings, reconstructed_embeddings, dim=1).mean()

def vsp_loss(original_batch, translated_batch):
    """
    Computes Vector Space Preservation loss.
    Ensures the internal geometry of a batch is preserved after translation.
    """
    EPS = 1e-10
    # Normalize embeddings to preserve angular relationships
    original_normed = original_batch / (original_batch.norm(dim=1, keepdim=True) + EPS)
    translated_normed = translated_batch / (translated_batch.norm(dim=1, keepdim=True) + EPS)

    # pairwise cosine similarity
    original_sims = original_normed @ original_normed.T
    translated_sims = translated_normed @ translated_normed.T

    return (original_sims - translated_sims).abs().mean()

def translation_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute translation loss using cosine similarity."""
    return 1 - F.cosine_similarity(source, target, dim=1).mean()

def cycle_consistency_loss(original: torch.Tensor, cycle_reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute cycle consistency loss."""
    return 1 - F.cosine_similarity(original, cycle_reconstructed, dim=1).mean()

model1_name = 'sentence-transformers/gtr-t5-base'
model2_name = 'thenlper/gte-base'

model1 = SentenceTransformer(model1_name)
model2 = SentenceTransformer(model2_name)

train_files = glob.glob('/content/NQ/v1.0/train/nq-train-*.jsonl.gz')
print(f"Found {len(train_files)} files.")

all_questions = []
print("Parsing JSONL files...")
for file_path in tqdm(train_files, desc="Reading files"):
    with gzip.open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # We use the question text as the input sequence
            all_questions.append(data['question_text'])

print(f"✅ Parsed a total of {len(all_questions):,} questions.")

print("🔀 Shuffling and creating disjoint datasets...")
random.seed(42) # for reproducibility
random.shuffle(all_questions)

num_samples_per_set = 1_000_000
if len(all_questions) < num_samples_per_set * 2:
    raise ValueError(f"Not enough data in NQ. Need {num_samples_per_set * 2}, but found {len(all_questions)}")

# Create the two disjoint sets for the two models
sup_data = all_questions[:num_samples_per_set]
unsup_data = all_questions[num_samples_per_set : num_samples_per_set * 2]

print(f"✅ Created sup_data with {len(sup_data):,} samples.")
print(f"✅ Created unsup_data with {len(unsup_data):,} samples.")

sup_embeddings = model1.encode(sup_data)
unsup_embeddings = model2.encode(unsup_data)

sup_embeddings = model1.encode(
    sup_data, 
    show_progress_bar=True,
    batch_size=128
)

unsup_embeddings = model2.encode(
    unsup_data, 
    show_progress_bar=True,
    batch_size=128
)
print(f"Shape of sup_embeddings: {sup_embeddings.shape}")
print(f"Shape of unsup_embeddings: {unsup_embeddings.shape}")