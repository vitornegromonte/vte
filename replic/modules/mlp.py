import torch
import torch.nn as nn
from typing import Optional

class MLP(nn.Module):
    """
    A generic, configurable, and corrected Multi-Layer Perceptron (MLP)
    with proper residual connections.
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
        output_norm: bool = False,
    ):
        super().__init__()
        assert depth >= 1, "Depth must be at least 1."

        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.residual = residual

        # Define normalization layer type
        if norm_style == 'batch':
            norm_layer = nn.BatchNorm1d
        elif norm_style == 'layer':
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm style: {norm_style}")

        # Create all layers in a single nn.Sequential for the main path
        layers = []
        current_dim = input_dim
        for i in range(depth):
            is_last_layer = (i == depth - 1)
            target_dim = output_dim if is_last_layer else hidden_dim
            
            layers.append(nn.Linear(current_dim, target_dim))
            
            if not is_last_layer:
                layers.extend([
                    nn.SiLU(),
                    norm_layer(target_dim),
                    nn.Dropout(p=dropout_rate)
                ])
            current_dim = target_dim
            
        self.main_path = nn.Sequential(*layers)
        
        # Handle the residual connection path
        if self.residual:
            if self.input_dim != self.output_dim:
                # If dimensions differ, create a learnable projection
                self.projection = nn.Linear(self.input_dim, self.output_dim)
            else:
                # If dimensions match, the projection is just an identity mapping
                self.projection = nn.Identity()
        
        if output_norm:
            self.output_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        else:
            self.output_norm = None

        self.initialize_weights(weight_init)

    def initialize_weights(self, weight_init: str = 'kaiming'):
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
        transformed_x = self.main_path(x)
        
        if self.residual:
            residual_x = self.projection(x)
            output = transformed_x + residual_x
        else:
            output = transformed_x

        if self.output_norm is not None:
            output = self.output_norm(output)
            
        return output