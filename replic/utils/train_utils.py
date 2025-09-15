import torch
import torch.nn.functional as F

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
