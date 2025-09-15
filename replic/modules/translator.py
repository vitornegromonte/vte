import torch
import torch.nn as nn
from mlp import MLP

class Translator(nn.Module):
    """
    - Input Adapters (A1, A2)
    - Shared Backbone (T)
    - Output Adapters (B1, B2)
    """
    def __init__(self, sup_dim: int, unsup_dim: int, latent_dim: int, hidden_dim: int, depth: int):
      super().__init__()

      # A1: sup -> latent
      self.input_adapter_sup = MLP(sup_dim, latent_dim, depth=1)
      # A2: unsup -> latent
      self.input_adapter_unsup = MLP(unsup_dim, latent_dim, depth=1)

      # T: Shared backbone
      self.backbone = MLP(
          input_dim=latent_dim,
          output_dim=latent_dim,
          hidden_dim=hidden_dim,
          depth=depth,
          residual=True
      )

      # B1: latent -> sup
      self.output_adapter_sup = MLP(latent_dim, sup_dim, depth=1)
      # B2: latent -> unsup
      self.output_adapter_unsup = MLP(latent_dim, unsup_dim, depth=1)

    def translate_sup_to_unsup(self, x: torch.Tensor) -> torch.Tensor:
      """ 
      F1 = B2 o T o A1 
      Translate a vector from the supervised to the unsupervised space.
      
      Args:
        x (torch.Tensor): The input vector in the supervised space.

      Returns:
        torch.Tensor: The translated vector in the unsupervised space.
      """
      return self.output_adapter_unsup(self.backbone(self.input_adapter_sup(x)))

    def translate_unsup_to_sup(self, x: torch.Tensor) -> torch.Tensor:
      """ 
      F2 = B1 o T o A2 
      Translate a vector from the unsupervised to the supervised space.

      Args:
        x (torch.Tensor): The input vector in the unsupervised space.

      Returns:
        torch.Tensor: The translated vector in the supervised space.       
      """
      return self.output_adapter_sup(self.backbone(self.input_adapter_unsup(x)))

    def reconstruct_sup(self, x: torch.Tensor) -> torch.Tensor:
      """
      R1 = B1 o T o A1

      Reconstructs a vector from the supervised space back to itself.

      Args:
        x (torch.Tensor): The input vector in domain 1.

      Returns:
        torch.Tensor: The reconstructed vector in domain 1.
      """

      return self.output_adapter_sup(self.backbone(self.input_adapter_sup(x)))

    def reconstruct_unsup(self, x: torch.Tensor) -> torch.Tensor:
      """
      R2 = B2 o T o A2

      Reconstructs a vector from the unsupervised space back to itself.
      
      Args:
        x (torch.Tensor): The input vector in domain 2.

      Returns:
        torch.Tensor: The reconstructed vector in domain 2.
      """
      return self.output_adapter_unsup(self.backbone(self.input_adapter_unsup(x)))

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
          # FIX: Call the correct method name
          return self.translate_sup_to_unsup(x)
      elif source_type == 'unsup':
          # FIX: Call the correct method name
          return self.translate_unsup_to_sup(x)
      else:
          raise ValueError(f"Unknown source_type: {source_type}. Must be 'sup' or 'unsup'.")