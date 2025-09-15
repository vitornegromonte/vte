"""
Vec2Vec Architecture MVP Implementation

This module implements a minimal viable product (MVP) of the Vec2Vec architecture
for translating between different embedding spaces using adversarial training.

Key Components:
1. Translator: Neural network that translates between embedding spaces
2. Discriminators: Adversarial networks for different objectives
3. Adapters: Input/output transformations for different embedding dimensions
4. Loss Functions: Multiple loss components for training
5. GAN Implementations: Vanilla, Least Squares, and Relativistic GANs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from typing import Dict, Tuple, Optional
import dataclasses
from types import SimpleNamespace

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration for MVP
config = SimpleNamespace(
    # General settings
    seed=42,
    normalize_embeddings=True,
    
    # Model architecture
    sup_emb_dim=768,  # Supervised embedding dimension (e.g., BERT)
    unsup_emb_dim=384,  # Unsupervised embedding dimension (e.g., GTE)
    
    # Translator settings
    translator_depth=3,
    translator_hidden_dim=512,
    norm_style='layer',
    
    # Discriminator settings
    discriminator_depth=3,
    discriminator_dim=512,
    gan_style='least_squares',
    
    # Training settings
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

print("Configuration:")
for key, value in vars(config).items():
    print(f"  {key}: {value}")

# ============================================================================
# 1. ADAPTERS
# ============================================================================

class InputAdapter(nn.Module):
    """Adapter for input embeddings to match translator input dimension."""
    
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        
        if input_dim == target_dim:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.Linear(input_dim, target_dim),
                nn.LayerNorm(target_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

class OutputAdapter(nn.Module):
    """Adapter for translator output to match target embedding dimension."""
    
    def __init__(self, input_dim: int, target_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        
        if input_dim == target_dim:
            self.adapter = nn.Identity()
        else:
            self.adapter = nn.Sequential(
                nn.Linear(input_dim, target_dim),
                nn.LayerNorm(target_dim),
                nn.SiLU(),
                nn.Dropout(0.1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)

# ============================================================================
# 2. TRANSLATOR
# ============================================================================

def add_residual(input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Add residual connection with dimension matching."""
    if input_x.shape[1] < x.shape[1]:
        padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device)
        input_x = torch.cat([input_x, padding], dim=1)
    elif input_x.shape[1] > x.shape[1]:
        input_x = input_x[:, :x.shape[1]]
    return x + input_x

class MLPWithResidual(nn.Module):
    """Multi-layer perceptron with residual connections."""
    
    def __init__(self, depth: int, in_dim: int, hidden_dim: int, out_dim: int,
                 norm_style: str = 'layer', output_norm: bool = False):
        super().__init__()
        self.depth = depth
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layers = nn.ModuleList()x  

        if norm_style == 'batch':
            norm_layer = nn.BatchNorm1d
        elif norm_style == 'layer':
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm style: {norm_style}")

        for layer_idx in range(self.depth):
            if layer_idx == 0:
                hidden_dim = out_dim if self.depth == 1 else hidden_dim
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.SiLU(),
                    )
                )
            elif layer_idx < self.depth - 1:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        norm_layer(hidden_dim),
                        nn.Dropout(p=0.1),
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Dropout(p=0.1),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, out_dim),
                    )
                )
        
        if output_norm:
            self.output_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        else:
            self.output_norm = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            input_x = x
            x = layer(x)
            x = add_residual(input_x, x)
        
        if self.output_norm is not None:
            x = self.output_norm(x)
        
        return x

class Translator(nn.Module):
    """Main translator network with input/output adapters."""
    
    def __init__(self, sup_dim: int, unsup_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        self.sup_dim = sup_dim
        self.unsup_dim = unsup_dim
        self.hidden_dim = hidden_dim
        
        # Use the larger dimension as the internal representation
        self.internal_dim = max(sup_dim, unsup_dim)
        
        # Input adapters
        self.sup_input_adapter = InputAdapter(sup_dim, self.internal_dim)
        self.unsup_input_adapter = InputAdapter(unsup_dim, self.internal_dim)
        
        # Core translator
        self.translator = MLPWithResidual(
            depth=depth,
            in_dim=self.internal_dim,
            hidden_dim=hidden_dim,
            out_dim=self.internal_dim,
            norm_style='layer'
        )
        
        # Output adapters
        self.sup_output_adapter = OutputAdapter(self.internal_dim, sup_dim)
        self.unsup_output_adapter = OutputAdapter(self.internal_dim, unsup_dim)
    
    def forward(self, x: torch.Tensor, source_type: str) -> torch.Tensor:
        """
        Forward pass through translator.
        
        Args:
            x: Input embeddings
            source_type: 'sup' or 'unsup' to indicate source type
        """
        # Input adaptation
        if source_type == 'sup':
            x = self.sup_input_adapter(x)
        else:
            x = self.unsup_input_adapter(x)
        
        # Translation
        x = self.translator(x)
        
        # Output adaptation
        if source_type == 'sup':
            return self.unsup_output_adapter(x)  # sup -> unsup
        else:
            return self.sup_output_adapter(x)    # unsup -> sup
    
    def reconstruct(self, x: torch.Tensor, source_type: str) -> torch.Tensor:
        """Reconstruct input by translating back."""
        translated = self.forward(x, source_type)
        if source_type == 'sup':
            return self.forward(translated, 'unsup')  # unsup -> sup
        else:
            return self.forward(translated, 'sup')    # sup -> unsup

# ============================================================================
# 3. DISCRIMINATORS
# ============================================================================

class Discriminator(nn.Module):
    """Discriminator network for adversarial training."""
    
    def __init__(self, latent_dim: int, discriminator_dim: int = 512, depth: int = 3):
        super().__init__()
        self.latent_dim = latent_dim
        assert depth >= 1, "Depth must be at least 1"
        
        self.layers = nn.ModuleList()
        if depth >= 2:
            layers = []
            layers.append(nn.Linear(latent_dim, discriminator_dim))
            layers.append(nn.Dropout(0.0))
            for _ in range(depth - 2):
                layers.append(nn.SiLU())
                layers.append(nn.Linear(discriminator_dim, discriminator_dim))
                layers.append(nn.LayerNorm(discriminator_dim))
                layers.append(nn.Dropout(0.0))
            layers.append(nn.SiLU())
            layers.append(nn.Linear(discriminator_dim, 1))
            self.layers.append(nn.Sequential(*layers))
        else:
            self.layers.append(nn.Linear(latent_dim, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================================
# 4. GAN IMPLEMENTATIONS
# ============================================================================

@dataclasses.dataclass
class VanillaGAN:
    """Vanilla GAN implementation."""
    
    generator: nn.Module
    discriminator: nn.Module
    generator_opt: optim.Optimizer
    discriminator_opt: optim.Optimizer
    config: SimpleNamespace
    
    def step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Train discriminator step."""
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        
        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device) * (1 - self.config.smooth)
        fake_labels = torch.ones((batch_size, 1), device=device) * self.config.smooth
        
        disc_loss_real = F.binary_cross_entropy_with_logits(d_real_logits, real_labels)
        disc_loss_fake = F.binary_cross_entropy_with_logits(d_fake_logits, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        
        disc_acc_real = (d_real_logits.sigmoid() < 0.5).float().mean().item()
        disc_acc_fake = (d_fake_logits.sigmoid() > 0.5).float().mean().item()
        
        self.discriminator_opt.zero_grad()
        disc_loss.backward()
        self.discriminator_opt.step()
        
        return disc_loss.detach(), disc_acc_real, disc_acc_fake
    
    def step_generator(self, fake_data: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Train generator step."""
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        real_labels = torch.zeros((batch_size, 1), device=device)
        gen_loss = F.binary_cross_entropy_with_logits(d_fake_logits, real_labels)
        gen_acc = (d_fake_logits.sigmoid() < 0.5).float().mean().item()
        
        return gen_loss, gen_acc    

@dataclasses.dataclass
class LeastSquaresGAN(VanillaGAN):
    """Least Squares GAN implementation."""
    
    def step_discriminator(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        """Train discriminator step for LSGAN."""
        real_data = real_data.detach().requires_grad_(True)
        fake_data = fake_data.detach().requires_grad_(True)
        
        d_real_logits = self.discriminator(real_data)
        d_fake_logits = self.discriminator(fake_data)
        
        device = d_real_logits.device
        batch_size = d_real_logits.size(0)
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)
        
        disc_loss_real = F.mse_loss(d_real_logits, real_labels)
        disc_loss_fake = F.mse_loss(d_fake_logits, fake_labels)
        disc_loss = (disc_loss_real + disc_loss_fake) / 2
        
        disc_acc_real = (d_real_logits > 0.5).float().mean().item()
        disc_acc_fake = (d_fake_logits < 0.5).float().mean().item()
        
        self.discriminator_opt.zero_grad()
        disc_loss.backward()
        self.discriminator_opt.step()
        
        return disc_loss.detach(), disc_acc_real, disc_acc_fake
    
    def step_generator(self, fake_data: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Train generator step for LSGAN."""
        d_fake_logits = self.discriminator(fake_data)
        device = fake_data.device
        batch_size = fake_data.size(0)
        real_labels = torch.ones((batch_size, 1), device=device)
        gen_loss = F.mse_loss(d_fake_logits, real_labels)
        gen_acc = (d_fake_logits > 0.5).float().mean().item()
        
        return gen_loss, gen_acc

# ============================================================================
# 5. LOSS FUNCTIONS
# ============================================================================

def reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute reconstruction loss using cosine similarity."""
    return 1 - F.cosine_similarity(original, reconstructed, dim=1).mean()

def translation_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute translation loss using cosine similarity."""
    return 1 - F.cosine_similarity(source, target, dim=1).mean()

def vector_space_preservation_loss(original_sims: torch.Tensor, translated_sims: torch.Tensor) -> torch.Tensor:
    """Compute vector space preservation loss."""
    return torch.abs(original_sims - translated_sims).mean()

def cycle_consistency_loss(original: torch.Tensor, cycle_reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute cycle consistency loss."""
    return 1 - F.cosine_similarity(original, cycle_reconstructed, dim=1).mean()

# ============================================================================
# 6. DATA GENERATION
# ============================================================================

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

# ============================================================================
# 7. TRAINING LOOP
# ============================================================================

def train_vec2vec(config: SimpleNamespace):
    """Main training function for Vec2Vec MVP."""
    
    print("Initializing models...")
    
    # Initialize translator
    translator = Translator(
        sup_dim=config.sup_emb_dim,
        unsup_dim=config.unsup_emb_dim,
        hidden_dim=config.translator_hidden_dim,
        depth=config.translator_depth
    ).to(device)
    
    # Initialize discriminators
    sup_discriminator = Discriminator(
        latent_dim=config.sup_emb_dim,
        discriminator_dim=config.discriminator_dim,
        depth=config.discriminator_depth
    ).to(device)
    
    unsup_discriminator = Discriminator(
        latent_dim=config.unsup_emb_dim,
        discriminator_dim=config.discriminator_dim,
        depth=config.discriminator_depth
    ).to(device)
    
    # Initialize optimizers
    translator_opt = optim.AdamW(translator.parameters(), lr=config.lr)
    sup_disc_opt = optim.AdamW(sup_discriminator.parameters(), lr=config.disc_lr)
    unsup_disc_opt = optim.AdamW(unsup_discriminator.parameters(), lr=config.disc_lr)
    
    # Initialize GANs
    sup_gan = LeastSquaresGAN(
        generator=translator,
        discriminator=sup_discriminator,
        generator_opt=translator_opt,
        discriminator_opt=sup_disc_opt,
        config=config
    )
    
    unsup_gan = LeastSquaresGAN(
        generator=translator,
        discriminator=unsup_discriminator,
        generator_opt=translator_opt,
        discriminator_opt=unsup_disc_opt,
        config=config
    )
    
    # Create dataloaders
    sup_loader, unsup_loader = create_dataloaders(config)
    
    # Training loop
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
        
        # Create iterators
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
            
            # Normalize embeddings if required
            if config.normalize_embeddings:
                sup_emb = F.normalize(sup_emb, p=2, dim=1)
                unsup_emb = F.normalize(unsup_emb, p=2, dim=1)
            
            # Forward passes
            # sup -> unsup
            sup_to_unsup = translator(sup_emb, 'sup')
            # unsup -> sup
            unsup_to_sup = translator(unsup_emb, 'unsup')
            
            # Reconstructions
            sup_recon = translator.reconstruct(sup_emb, 'sup')
            unsup_recon = translator.reconstruct(unsup_emb, 'unsup')
            
            # Cycle consistency
            sup_cycle = translator(unsup_to_sup, 'sup')
            unsup_cycle = translator(sup_to_unsup, 'unsup')
            
            # Compute losses
            rec_loss = (reconstruction_loss(sup_emb, sup_recon) + 
                       reconstruction_loss(unsup_emb, unsup_recon)) / 2
            
            # Vector space preservation
            sup_sims = torch.mm(sup_emb, sup_emb.t())
            sup_trans_sims = torch.mm(sup_to_unsup, sup_to_unsup.t())
            unsup_sims = torch.mm(unsup_emb, unsup_emb.t())
            unsup_trans_sims = torch.mm(unsup_to_sup, unsup_to_sup.t())
            
            vsp_loss = (vector_space_preservation_loss(sup_sims, sup_trans_sims) +
                       vector_space_preservation_loss(unsup_sims, unsup_trans_sims)) / 2
            
            # Cycle consistency
            cc_loss = (cycle_consistency_loss(sup_emb, sup_cycle) +
                      cycle_consistency_loss(unsup_emb, unsup_cycle)) / 2
            
            # Adversarial losses
            sup_disc_loss, _, _ = sup_gan.step_discriminator(sup_emb, unsup_to_sup)
            unsup_disc_loss, _, _ = unsup_gan.step_discriminator(unsup_emb, sup_to_unsup)
            disc_loss = (sup_disc_loss + unsup_disc_loss) / 2
            
            sup_gen_loss, _ = sup_gan.step_generator(unsup_to_sup)
            unsup_gen_loss, _ = unsup_gan.step_generator(sup_to_unsup)
            gen_loss = (sup_gen_loss + unsup_gen_loss) / 2
            
            # Total loss
            total_loss = (config.loss_coefficient_rec * rec_loss +
                         config.loss_coefficient_vsp * vsp_loss +
                         config.loss_coefficient_cc * cc_loss +
                         config.loss_coefficient_gen * gen_loss)
            
            # Backward pass
            translator_opt.zero_grad()
            total_loss.backward()
            translator_opt.step()
            
            # Log losses
            epoch_losses['rec_loss'].append(rec_loss.item())
            epoch_losses['vsp_loss'].append(vsp_loss.item())
            epoch_losses['cc_loss'].append(cc_loss.item())
            epoch_losses['disc_loss'].append(disc_loss.item())
            epoch_losses['gen_loss'].append(gen_loss.item())
        
        # Average losses for epoch
        for k in losses.keys():
            losses[k].append(np.mean(epoch_losses[k]))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}:")
            for k, v in losses.items():
                print(f"  {k}: {v[-1]:.4f}")
    
    return translator, losses

# ============================================================================
# 8. EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_translator(translator: nn.Module, config: SimpleNamespace):
    """Evaluate the trained translator."""
    translator.eval()
    
    # Generate test data
    test_sup = generate_synthetic_embeddings(100, config.sup_emb_dim, 0.05)
    test_unsup = generate_synthetic_embeddings(100, config.unsup_emb_dim, 0.05)
    
    test_sup = test_sup.to(device)
    test_unsup = test_unsup.to(device)
    
    if config.normalize_embeddings:
        test_sup = F.normalize(test_sup, p=2, dim=1)
        test_unsup = F.normalize(test_unsup, p=2, dim=1)
    
    with torch.no_grad():
        # Test translations
        sup_to_unsup = translator(test_sup, 'sup')
        unsup_to_sup = translator(test_unsup, 'unsup')
        
        # Test reconstructions
        sup_recon = translator.reconstruct(test_sup, 'sup')
        unsup_recon = translator.reconstruct(test_unsup, 'unsup')
        
        # Compute metrics
        sup_trans_loss = translation_loss(test_sup, unsup_to_sup).item()
        unsup_trans_loss = translation_loss(test_unsup, sup_to_unsup).item()
        sup_rec_loss = reconstruction_loss(test_sup, sup_recon).item()
        unsup_rec_loss = reconstruction_loss(test_unsup, unsup_recon).item()
        
        print(f"Evaluation Results:")
        print(f"  Sup -> Unsup Translation Loss: {sup_trans_loss:.4f}")
        print(f"  Unsup -> Sup Translation Loss: {unsup_trans_loss:.4f}")
        print(f"  Sup Reconstruction Loss: {sup_rec_loss:.4f}")
        print(f"  Unsup Reconstruction Loss: {unsup_rec_loss:.4f}")

def plot_training_losses(losses: Dict[str, list]):
    """Plot training losses."""
    plt.figure(figsize=(15, 10))
    
    for i, (loss_name, loss_values) in enumerate(losses.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(loss_values)
        plt.title(f'{loss_name.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Vec2Vec Architecture MVP")
    print("=" * 50)
    
    # Train the model
    translator, losses = train_vec2vec(config)
    
    # Evaluate the model
    evaluate_translator(translator, config)
    
    # Plot training losses
    plot_training_losses(losses)
    
    print("Training completed!") 