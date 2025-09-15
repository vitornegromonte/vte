import torch
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, asdict
from typing import Tuple

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

def create_dataloaders(config: dataclass) -> Tuple[DataLoader, DataLoader]:
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