# Vec2Vec Architecture MVP

This directory contains a minimal viable product (MVP) implementation of the Vec2Vec architecture for translating between different embedding spaces using adversarial training.

## Overview

Vec2Vec uses GANs to learn mappings between different embedding spaces while preserving semantic relationships. The MVP implementation includes all key components:

1. **Translator**: Neural network that translates between embedding spaces
2. **Discriminators**: Adversarial networks for different objectives
3. **Adapters**: Input/output transformations for different embedding dimensions
4. **Loss Functions**: Multiple loss components for training
5. **GAN Implementations**: Vanilla, Least Squares, and Relativistic GANs

## Files

- `vec2vec_mvp.py`: Complete MVP implementation with all components
- `vec2vec_mvp.ipynb`: Jupyter notebook demonstrating the implementation

## Key Components

### 1. Adapters

```python
class InputAdapter(nn.Module):
    """Adapter for input embeddings to match translator input dimension."""
    
class OutputAdapter(nn.Module):
    """Adapter for translator output to match target embedding dimension."""
```

### 2. Translator

```python
class Translator(nn.Module):
    """Main translator network with input/output adapters."""
    
    def forward(self, x: torch.Tensor, source_type: str) -> torch.Tensor:
        # Translates between embedding spaces
        # source_type: 'sup' or 'unsup'
    
    def reconstruct(self, x: torch.Tensor, source_type: str) -> torch.Tensor:
        # Reconstructs input by translating back
```

### 3. Discriminator

```python
class Discriminator(nn.Module):
    """Discriminator network for adversarial training."""
```

### 4. GAN Implementations

```python
@dataclasses.dataclass
class VanillaGAN:
    """Vanilla GAN implementation."""
    
@dataclasses.dataclass
class LeastSquaresGAN(VanillaGAN):
    """Least Squares GAN implementation."""
```

### 5. Loss Functions

```python
def reconstruction_loss(original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute reconstruction loss using cosine similarity."""
    
def translation_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute translation loss using cosine similarity."""
    
def vector_space_preservation_loss(original_sims: torch.Tensor, translated_sims: torch.Tensor) -> torch.Tensor:
    """Compute vector space preservation loss."""
    
def cycle_consistency_loss(original: torch.Tensor, cycle_reconstructed: torch.Tensor) -> torch.Tensor:
    """Compute cycle consistency loss."""
```

## Usage

### Quick Start

1. **Install dependencies**:
```bash
pip install torch torchvision torchaudio transformers numpy matplotlib tqdm
```

2. **Run the MVP implementation**:
```bash
python vec2vec_mvp.py
```

3. **Use in Jupyter notebook**:
```python
from vec2vec_mvp import *
translator, losses = train_vec2vec(config)
```

### Configuration

The MVP uses a configuration object with the following key parameters:

```python
config = SimpleNamespace(
    # Model architecture
    sup_emb_dim=768,      # Supervised embedding dimension (e.g., BERT)
    unsup_emb_dim=384,    # Unsupervised embedding dimension (e.g., GTE)
    
    # Translator settings
    translator_depth=3,
    translator_hidden_dim=512,
    
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
    loss_coefficient_rec=1.0,   # Reconstruction loss
    loss_coefficient_vsp=1.0,   # Vector space preservation
    loss_coefficient_cc=10.0,   # Cycle consistency
    loss_coefficient_disc=1.0,  # Discriminator loss
    loss_coefficient_gen=1.0,   # Generator loss
)
```

### Training Process

The training loop includes:

1. **Forward passes**: Translate embeddings between spaces
2. **Reconstruction**: Reconstruct inputs through cycle translation
3. **Loss computation**: Multiple loss components
   - Reconstruction loss
   - Vector space preservation loss
   - Cycle consistency loss
   - Adversarial losses
4. **Backward passes**: Update translator and discriminators

### Evaluation

The MVP includes comprehensive evaluation:

```python
# Evaluate translation quality
evaluate_translator(translator, config)

# Plot training losses
plot_training_losses(losses)

# Analyze translation quality
# (see notebook for detailed analysis)
```

## Architecture Details

### Translator Architecture

```
Input Embedding → Input Adapter → MLP with Residual → Output Adapter → Output Embedding
```

- **Input/Output Adapters**: Handle dimension mismatches between embedding spaces
- **MLP with Residual**: Core translation network with residual connections
- **Layer Normalization**: Stabilizes training
- **SiLU Activation**: Smooth activation function

### Loss Components

1. **Reconstruction Loss**: Ensures translator can reconstruct inputs
2. **Vector Space Preservation**: Maintains embedding neighborhoods
3. **Cycle Consistency**: Ensures translation consistency
4. **Adversarial Losses**: Multiple discriminators for different objectives

### GAN Variants

1. **Vanilla GAN**: Standard binary cross-entropy loss
2. **Least Squares GAN**: MSE-based loss for better stability
3. **Relativistic GAN**: (Can be extended from base implementation)

## Key Features

✅ **Complete Implementation**: All core Vec2Vec components
✅ **Modular Design**: Easy to extend and modify
✅ **Multiple GAN Variants**: Vanilla and Least Squares GANs
✅ **Comprehensive Losses**: Reconstruction, VSP, cycle consistency
✅ **Dimension Handling**: Automatic adapter creation for different dimensions
✅ **Training Pipeline**: Complete training and evaluation
✅ **Visualization**: Loss plots and quality analysis
✅ **Synthetic Data**: Built-in data generation for testing

## Potential Improvements

1. **Add Relativistic GAN**: Extend GAN implementations
2. **Gradient Penalty**: Implement R1/R2 regularization
3. **Real Embeddings**: Support for BERT, GTE, GTR models
4. **Multi-GPU Training**: Distributed training support
5. **Early Stopping**: Model checkpointing and early stopping
6. **Advanced Metrics**: More sophisticated evaluation metrics
7. **Configuration Files**: TOML/YAML configuration support

## Research Applications

This MVP can be used for:

- **Embedding Alignment**: Align different embedding models
- **Cross-Modal Translation**: Translate between different modalities
- **Domain Adaptation**: Adapt embeddings across domains
- **Model Compression**: Reduce embedding dimensions while preserving quality
- **Research Prototyping**: Quick experimentation with Vec2Vec ideas

## Citation

If you use this implementation in your research, please cite the original Vec2Vec paper:

```bibtex
@misc{jha2025harnessinguniversalgeometryembeddings,
      title={Harnessing the Universal Geometry of Embeddings}, 
      author={Rishi Jha and Collin Zhang and Vitaly Shmatikov and John X. Morris},
      year={2025},
      eprint={2505.12540},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.12540}, 
}
```

## License

This implementation is provided for research and educational purposes. Please refer to the original Vec2Vec repository for licensing information. 