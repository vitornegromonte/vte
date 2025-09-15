import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for the Vec2Vec training run."""
    seed: int = 42
    normalize_embeddings: bool = False

    # Model dimensions
    sup_emb_dim: int = 768
    unsup_emb_dim: int = 384

    # Translator architecture
    translator_depth: int = 3
    
    translator_hidden_dim: int = 512
    
    # Discriminator architecture
    discriminator_depth: int = 3
    disc_dim: int = 512

    # Training parameters
    batch_size: int = 64
    lr: float = 2e-5
    disc_lr: float = 1e-5
    epochs: int = 50

    # Loss coefficients
    loss_coefficient_rec: float = 1.0
    loss_coefficient_vsp: float = 1.0
    loss_coefficient_cc: float = 10.0
    loss_coefficient_gen: float = 1.0

    # Data settings
    num_samples: int = 1000
    noise_level: float = 0.1

config = TrainingConfig()