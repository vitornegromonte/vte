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


CUDA_LAUNCH_BLOCKING=1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = torch.manual_seed(42)

def generate_synthetic_embeddings(num_samples: int, dim: int, noise_level: float = 0.1) -> torch.Tensor:
    """Generate synthetic embeddings for demonstration."""
    # Generate random embeddings
    embeddings = torch.randn(num_samples, dim)

    # Normalize to unit vectors
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Add some structure by clustering
    num_clusters = 5
    cluster_centers = torch.randn(num_clusters, dim)
    cluster_centers = F.normalize(cluster_centers, p=2, dim=1)

    cluster_assignments = torch.randint(0, num_clusters, (num_samples,))
    for i in range(num_samples):
        cluster_idx = cluster_assignments[i]
        embeddings[i] = embeddings[i] + 0.5 * cluster_centers[cluster_idx]
        embeddings[i] = F.normalize(embeddings[i], p=2, dim=0)

    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(embeddings) * noise_level
        embeddings = embeddings + noise
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings

def create_dataloaders(config: SimpleNamespace) -> Tuple[DataLoader, DataLoader]:
    """Create synthetic dataloaders for training."""
    # Generate synthetic data
    sup_embeddings = generate_synthetic_embeddings(
        config.num_samples, config.sup_emb_dim, config.noise_level
    )
    unsup_embeddings = generate_synthetic_embeddings(
        config.num_samples, config.unsup_emb_dim, config.noise_level
    )

    # Create datasets
    sup_dataset = TensorDataset(sup_embeddings)
    unsup_dataset = TensorDataset(unsup_embeddings)

    # Create dataloaders
    sup_loader = DataLoader(sup_dataset, batch_size=config.batch_size, shuffle=True)
    unsup_loader = DataLoader(unsup_dataset, batch_size=config.batch_size, shuffle=True)

    return sup_loader, unsup_loader
  
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

def train(config: SimpleNamespace):
    """Main training function for Vec2Vec MVP."""

    print("Initializing models...")

    latent_dim = max(config.sup_emb_dim, config.unsup_emb_dim)

    translator = Translator(
        sup_dim=config.sup_emb_dim,
        unsup_dim=config.unsup_emb_dim,
        latent_dim=latent_dim,
        hidden_dim=config.translator_hidden_dim,
        depth=config.translator_depth,
        source_type=config.source_type
    ).to(device)

    sup_discriminator = Discriminator(
        latent_dim=config.sup_emb_dim,
        disc_dim=config.disc_dim,
        depth=config.discriminator_depth
    ).to(device)

    unsup_discriminator = Discriminator(
        latent_dim=config.unsup_emb_dim,
        disc_dim=config.disc_dim,
        depth=config.discriminator_depth
    ).to(device)

    translator_opt = optim.AdamW(translator.parameters(), lr=config.lr)
    sup_disc_opt = optim.AdamW(sup_discriminator.parameters(), lr=config.disc_lr)
    unsup_disc_opt = optim.AdamW(unsup_discriminator.parameters(), lr=config.disc_lr)

    sup_gan = VanillaGAN(
        generator=translator,
        discriminator=sup_discriminator,
        generator_opt=translator_opt,
        discriminator_opt=sup_disc_opt,
        config=config
    )

    unsup_gan = VanillaGAN(
        generator=translator,
        discriminator=unsup_discriminator,
        generator_opt=translator_opt,
        discriminator_opt=unsup_disc_opt,
        config=config
    )

    sup_loader, unsup_loader = create_dataloaders(config)

    print(f"Starting training for {config.epochs} epochs...")

    losses = {
        'rec_loss': [],
        'vsp_loss': [],
        'cc_loss': [],
        'disc_loss': [],
        'gen_loss': []
    }

    for epoch in range(config.epochs):
        epoch_losses = {k: [] for k in losses.keys()}

        sup_iter = iter(sup_loader)
        unsup_iter = iter(unsup_loader)

        num_batches = min(len(sup_loader), len(unsup_loader))

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config.epochs}"):
            try:
                sup_batch = next(sup_iter)
                unsup_batch = next(unsup_iter)
            except StopIteration:
                break

            sup_emb = sup_batch[0].to(device)
            unsup_emb = unsup_batch[0].to(device)

            if config.normalize_embeddings:
                sup_emb = F.normalize(sup_emb, p=2, dim=1)
                unsup_emb = F.normalize(unsup_emb, p=2, dim=1)

            # sup -> unsup
            sup_to_unsup = translator(sup_emb, 'sup')
            # unsup -> sup
            unsup_to_sup = translator(unsup_emb, 'unsup')

            sup_recon = translator.reconstruct(sup_emb, 'sup')
            unsup_recon = translator.reconstruct(unsup_emb, 'unsup')


            sup_cycle = translator(unsup_to_sup, 'sup')
            unsup_cycle = translator(sup_to_unsup, 'unsup')

            rec_loss = (reconstruction_loss(sup_emb, sup_recon) +
                       reconstruction_loss(unsup_emb, unsup_recon)) / 2

            sup_sims = torch.mm(sup_emb, sup_emb.t())
            sup_trans_sims = torch.mm(sup_to_unsup, sup_to_unsup.t())
            unsup_sims = torch.mm(unsup_emb, unsup_emb.t())
            unsup_trans_sims = torch.mm(unsup_to_sup, unsup_to_sup.t())

            vsp_loss = (vsp_loss(sup_sims, sup_trans_sims) +
                       vsp_loss(unsup_sims, unsup_trans_sims)) / 2

            cc_loss = (cycle_consistency_loss(sup_emb, sup_cycle) +
                      cycle_consistency_loss(unsup_emb, unsup_cycle)) / 2


            sup_disc_loss, _, _ = sup_gan.step_discriminator(sup_emb, unsup_to_sup)
            unsup_disc_loss, _, _ = unsup_gan.step_discriminator(unsup_emb, sup_to_unsup)
            disc_loss = (sup_disc_loss + unsup_disc_loss) / 2

            sup_gen_loss, _ = sup_gan.step_generator(unsup_to_sup)
            unsup_gen_loss, _ = unsup_gan.step_generator(sup_to_unsup)
            gen_loss = (sup_gen_loss + unsup_gen_loss) / 2

            total_loss = (config.loss_coefficient_rec * rec_loss +
                         config.loss_coefficient_vsp * vsp_loss +
                         config.loss_coefficient_cc * cc_loss +
                         config.loss_coefficient_gen * gen_loss)

            translator_opt.zero_grad()
            total_loss.backward()
            translator_opt.step()

            epoch_losses['rec_loss'].append(rec_loss.item())
            epoch_losses['vsp_loss'].append(vsp_loss.item())
            epoch_losses['cc_loss'].append(cc_loss.item())
            epoch_losses['disc_loss'].append(disc_loss.item())
            epoch_losses['gen_loss'].append(gen_loss.item())

        for k in losses.keys():
            losses[k].append(np.mean(epoch_losses[k]))

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}:")
            for k, v in losses.items():
                print(f"  {k}: {v[-1]:.4f}")

    return translator, losses
  
  
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


class Backbone(nn.Module):
  """
  Main translator backbone
  """

  def __init__(self,
               sup_dim: int,
               hidden_dim: int,
               unsup_dim: int,
               depth: int,
               source_type: str):
    """
    Initializes the translator shared backbone MLP.

    Args:
      sup_dim (int): The dimension of the supervised embeddings. (Supervised == known encoder (M_2))
      unsup_dim (int): The dimension of the unsupervised embeddings. (Unsupervised == unknown encoder (M_1))
      hidden_dim (int): The hidden dimension for the MLP.
      depth (int): The number of layers.
      source_type (str): The source domain ('supervised' or 'unsupervised').

    Returns:
      None
    """
    super().__init__()
    self.sup_dim = sup_dim
    self.unsup_sim = unsup_dim
    self.hidden_dim = hidden_dim
    self.depth = depth
    self.source_type = source_type
    self.latent_dim = max(sup_dim, unsup_dim) # To avoid losing info, the latent space dimensions are defined as the max of the unsupervised and the supervised spaces.

    self.sup_input_adapter = MLP(sup_dim, self.latent_dim, dropout_rate=0.1, depth = 1)
    self.unsup__input_adapter = MLP(unsup_dim, self.latent_dim, dropout_rate=0.1, depth = 1)

    self.translator = MLP(
        input_dim = self.latent_dim,
        hidden_dim = hidden_dim,
        output_dim = self.latent_dim,
        depth = depth,
        norm_style = 'layer',
        weight_init = 'kaiming',
        dropout_rate = 0.1,
        residual = True,
        output_norm = False,
        source_type = source_type
    )

    self.sup_output_adapter = MLP(self.latent_dim, sup_dim, dropout_rate=0.1, depth = 1)
    self.unsup_output_adapter = MLP(self.latent_dim, unsup_dim, dropout_rate=0.1, depth = 1)

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

class Translator(nn.Module):
  """
  Space-specific adapters and the shared neural backbone
  """

  def __init__(self, sup_dim: int, unsup_dim: int , latent_dim: int, hidden_dim: int, depth:int, source_type: str):
    super().__init__()
    self.source_type = source_type
    # Input Adapters A1 and A2
    self.adapter_A1 = MLP(sup_dim, latent_dim, dropout_rate=0.1, depth = 1)
    self.adapter_A2 = MLP(unsup_dim, latent_dim, dropout_rate=0.1, depth = 1)

    # Translator backbone T
    self.translator = Backbone(
        sup_dim = sup_dim,
        unsup_dim = unsup_dim,
        hidden_dim = hidden_dim,
        depth = depth,
        source_type = source_type
    )

    # Output Adapters B1 and B2
    self.adapter_B1 = MLP(latent_dim, unsup_dim, dropout_rate=0.1, depth = 1)
    self.adapter_B2 = MLP(latent_dim, sup_dim, dropout_rate=0.1, depth = 1)

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

  def forward(self, x: torch.Tensor, source_type: str) -> torch.Tensor:
    """
    Forward pass of the translator.

    Args:
      x (torch.Tensor): The input vector.
      source_type (str): The source domain ('sup' or 'unsup').

    Returns:
      torch.Tensor: The translated vector.
    """
    if source_type == 'sup':
        return self.translate_1_to_2(x)
    elif source_type == 'unsup':
        return self.translate_2_to_1(x)
    else:
      raise ValueError(f"Unknown source_type: {source_type}. Must be 'sup' or 'unsup'.")

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
  """
  Compute translation loss using cosine similarity.
  """
  return 1 - F.cosine_similarity(source, target, dim=1).mean()

def cycle_consistency_loss(original: torch.Tensor, cycle_reconstructed: torch.Tensor) -> torch.Tensor:
  """
  Compute cycle consistency loss.
  """
  return 1 - F.cosine_similarity(original, cycle_reconstructed, dim=1).mean()

config = SimpleNamespace(
    seed=42,
    normalize_embeddings=False,

    sup_emb_dim=768,  # Supervised embedding dimension (e.g., BERT)
    unsup_emb_dim=384,  # Unsupervised embedding dimension (e.g., GTE)

    source_type='sup',

    translator_depth=3,
    translator_hidden_dim=512,
    norm_style='layer',

    discriminator_depth=3,
    disc_dim=512,
    gan_style='vanilla',

    batch_size=64,
    lr=2e-5,
    disc_lr=1e-5,
    epochs=50,

    # Loss coefficients
    loss_coefficient_rec=1.0,  # Reconstruction loss
    loss_coefficient_vsp=1.0,  # Vector space preservation
    loss_coefficient_cc=10.0,  # Cycle consistency
    loss_coefficient_disc=1.0,  # Discriminator loss
    loss_coefficient_gen=1.0,  # Generator loss

    # GAN settings
    smooth=0.9,

    # Data settings
    num_samples=1000,
    noise_level=0.1
)

translator, losses = train(config)