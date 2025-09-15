# Vec2Vec

> **Project page:** https://vec2vec.github.io/

Vec2Vec is a framework for training GANs (Generative Adversarial Networks) to convert between different embedding models. It allows for the transformation of embeddings from one latent space to another while preserving the semantic relationships between vectors.

## Overview

Vec2Vec uses adversarial training to learn mappings between different embedding spaces. It can translate embeddings from unsupervised models like GTE (General Text Embeddings) to supervised models like GTR (General Text Representations), allowing for better alignment and utility of various embedding types.

![Vec2Vec Universal Architecture](universal2.png)

## Configuration

Vec2Vec uses a toml configuration file with sections for general settings, translator architecture, discriminator parameters, training hyperparameters, GAN-specific settings, evaluation metrics, and logging options. These files are stored in the `configs/` folder of the repo.

## Usage

To run the main experiment using the `configs/[EXPERIMENT_NAME].toml` configuration, run:

```bash
python train.py [EXPERIMENT_NAME] --num_points [NUMBER OF POINTS] --epochs [EPOCHS]
```

Most of the experiments in the paper use `configs/unsupervised.toml`.

> [!NOTE]
> GAN training is *very* unstable (especially across backbones!) You may need to try multiple seeds for convergence!

### Command Line Arguments

Each entry in the toml configuration can be altered in two ways: (1) by directly changing the configuration file, or (2) adding a flag to the run command above.
The `train.py` script with accepts various parameters, including:

#### General Settings
- `--num_points`: Number of points to allocate to each encoder
- `--unsup_points`: Number of points to allocate to the unsupervised encoder (the supervised recieves the rest)
- `--unsup_emb`: Unsupervised embedding model (e.g., 'gte')
- `--sup_emb`: Supervised embedding model (e.g., 'gtr')
- `--dataset`: Dataset to use (e.g., "nq")
- `--epochs`: Number of epochs to train for
- `--seed`: Random seed for reproducibility
- `--sampling_seed`: Seed for sampling operations
- `--train_dataset_seed`: Seed for training dataset generation
- `--val_dataset_seed`: Seed for validation dataset generation

Please refer to the example `.toml` files for all possible settings (there are a lot!).

## Model Release
We are releasing the trained weights of the models used in the paper [here](https://github.com/rjha18/vec2vec/releases/tag/v1.0.0). To use the trained weights, use `translator.load_state_dict()`. For an example on usage, please refer to `eval.py`.

## The Paper
Our paper is available on ArXiv: [Harnessing the Universal Geometry of Embeddings](https://arxiv.org/abs/2505.12540) (Jha, Zhang, Shmatikov, and Morris, 2025). If you find the code useful, please use the following citation:

```
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
