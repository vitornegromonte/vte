import torch
import torch.nn as nn
from mlp import MLP

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