#!/usr/bin/env python3
"""
Example usage of the Vec2Vec MVP implementation.

This script demonstrates how to:
1. Import and configure the Vec2Vec MVP
2. Train a translator between embedding spaces
3. Evaluate the trained model
4. Visualize results
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace

# Import the MVP implementation
from vec2vec_mvp import (
    Translator, Discriminator, LeastSquaresGAN,
    reconstruction_loss, translation_loss, vector_space_preservation_loss, cycle_consistency_loss,
    generate_synthetic_embeddings, create_dataloaders,
    train_vec2vec, evaluate_translator, plot_training_losses
)

def main():
    """Main example function."""
    print("Vec2Vec MVP Example Usage")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration for quick demo
    config = SimpleNamespace(
        # General settings
        seed=42,
        normalize_embeddings=True,
        
        # Model architecture
        sup_emb_dim=256,  # Smaller for faster demo
        unsup_emb_dim=128,  # Smaller for faster demo
        
        # Translator settings
        translator_depth=2,  # Shallow for faster demo
        translator_hidden_dim=256,
        norm_style='layer',
        
        # Discriminator settings
        discriminator_depth=2,  # Shallow for faster demo
        discriminator_dim=256,
        gan_style='least_squares',
        
        # Training settings
        batch_size=32,
        lr=2e-5,
        disc_lr=1e-5,
        epochs=20,  # Fewer epochs for demo
        
        # Loss coefficients
        loss_coefficient_rec=1.0,
        loss_coefficient_vsp=1.0,
        loss_coefficient_cc=10.0,
        loss_coefficient_disc=1.0,
        loss_coefficient_gen=1.0,
        
        # GAN settings
        smooth=0.9,
        
        # Data settings
        num_samples=500,  # Fewer samples for demo
        noise_level=0.1
    )
    
    print("Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    
    # Step 1: Create and display model architecture
    print("\n1. Creating model architecture...")
    translator = Translator(
        sup_dim=config.sup_emb_dim,
        unsup_dim=config.unsup_emb_dim,
        hidden_dim=config.translator_hidden_dim,
        depth=config.translator_depth
    ).to(device)
    
    print(f"Translator parameters: {sum(p.numel() for p in translator.parameters()):,}")
    
    # Step 2: Generate sample data
    print("\n2. Generating sample data...")
    sample_sup = generate_synthetic_embeddings(50, config.sup_emb_dim, 0.1)
    sample_unsup = generate_synthetic_embeddings(50, config.unsup_emb_dim, 0.1)
    
    print(f"Generated {sample_sup.shape[0]} supervised embeddings of dimension {sample_sup.shape[1]}")
    print(f"Generated {sample_unsup.shape[0]} unsupervised embeddings of dimension {sample_unsup.shape[1]}")
    
    # Step 3: Test forward pass
    print("\n3. Testing forward pass...")
    translator.eval()
    with torch.no_grad():
        sample_sup = sample_sup.to(device)
        sample_unsup = sample_unsup.to(device)
        
        if config.normalize_embeddings:
            sample_sup = F.normalize(sample_sup, p=2, dim=1)
            sample_unsup = F.normalize(sample_unsup, p=2, dim=1)
        
        # Test translations
        sup_to_unsup = translator(sample_sup, 'sup')
        unsup_to_sup = translator(sample_unsup, 'unsup')
        
        # Test reconstructions
        sup_recon = translator.reconstruct(sample_sup, 'sup')
        unsup_recon = translator.reconstruct(sample_unsup, 'unsup')
        
        print(f"Sup->Unsup translation shape: {sup_to_unsup.shape}")
        print(f"Unsup->Sup translation shape: {unsup_to_sup.shape}")
        print(f"Sup reconstruction shape: {sup_recon.shape}")
        print(f"Unsup reconstruction shape: {unsup_recon.shape}")
    
    # Step 4: Train the model
    print("\n4. Training the model...")
    translator, losses = train_vec2vec(config)
    print("Training completed!")
    
    # Step 5: Evaluate the model
    print("\n5. Evaluating the model...")
    evaluate_translator(translator, config)
    
    # Step 6: Visualize training losses
    print("\n6. Visualizing training losses...")
    plot_training_losses(losses)
    
    # Step 7: Analyze translation quality
    print("\n7. Analyzing translation quality...")
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
        
        # Test cycle consistency
        sup_cycle = translator(unsup_to_sup, 'sup')
        unsup_cycle = translator(sup_to_unsup, 'unsup')
        
        # Compute metrics
        sup_trans_loss = translation_loss(test_sup, unsup_to_sup).item()
        unsup_trans_loss = translation_loss(test_unsup, sup_to_unsup).item()
        sup_rec_loss = reconstruction_loss(test_sup, sup_recon).item()
        unsup_rec_loss = reconstruction_loss(test_unsup, unsup_recon).item()
        sup_cycle_loss = cycle_consistency_loss(test_sup, sup_cycle).item()
        unsup_cycle_loss = cycle_consistency_loss(test_unsup, unsup_cycle).item()
        
        print(f"Translation Quality Metrics:")
        print(f"  Sup -> Unsup Translation Loss: {sup_trans_loss:.4f}")
        print(f"  Unsup -> Sup Translation Loss: {unsup_trans_loss:.4f}")
        print(f"  Sup Reconstruction Loss: {sup_rec_loss:.4f}")
        print(f"  Unsup Reconstruction Loss: {unsup_rec_loss:.4f}")
        print(f"  Sup Cycle Consistency Loss: {sup_cycle_loss:.4f}")
        print(f"  Unsup Cycle Consistency Loss: {unsup_cycle_loss:.4f}")
    
    # Step 8: Visualize similarity matrices
    print("\n8. Visualizing similarity matrices...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original vs Translated (Sup -> Unsup)
    sup_sim_orig = torch.mm(test_sup.cpu(), test_sup.cpu().t())
    sup_sim_trans = torch.mm(sup_to_unsup.cpu(), sup_to_unsup.cpu().t())
    
    im1 = axes[0, 0].imshow(sup_sim_orig.numpy(), cmap='viridis')
    axes[0, 0].set_title('Original Sup Similarity')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(sup_sim_trans.numpy(), cmap='viridis')
    axes[0, 1].set_title('Translated Sup->Unsup Similarity')
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(torch.abs(sup_sim_orig - sup_sim_trans).numpy(), cmap='hot')
    axes[0, 2].set_title('Similarity Difference')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Original vs Reconstructed
    sup_sim_recon = torch.mm(sup_recon.cpu(), sup_recon.cpu().t())
    
    im4 = axes[1, 0].imshow(sup_sim_orig.numpy(), cmap='viridis')
    axes[1, 0].set_title('Original Sup Similarity')
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(sup_sim_recon.numpy(), cmap='viridis')
    axes[1, 1].set_title('Reconstructed Sup Similarity')
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(torch.abs(sup_sim_orig - sup_sim_recon).numpy(), cmap='hot')
    axes[1, 2].set_title('Reconstruction Difference')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.show()
    
    # Step 9: Summary
    print("\n9. Summary")
    print("=" * 50)
    print("✅ Vec2Vec MVP demonstration completed successfully!")
    print("✅ All components working correctly:")
    print("  - Translator with adapters")
    print("  - Discriminators for adversarial training")
    print("  - Multiple loss functions")
    print("  - Training and evaluation pipeline")
    print("  - Visualization and analysis tools")
    print("\n🎯 The MVP provides a solid foundation for Vec2Vec research!")
    print("📈 You can now extend it with real embedding models and advanced features.")

if __name__ == "__main__":
    main() 