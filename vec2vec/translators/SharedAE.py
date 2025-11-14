import math
from typing import Optional, List, Tuple, Type
import torch, torch.nn as nn, torch.nn.functional as F
from geomloss import SamplesLoss

class MLP(nn.Module):
    """
    Generic Multi-Layer Perceptron (MLP)
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
        activation: Type[nn.Module] = nn.SiLU,
    ):
        """
        Initializes the MLP.

        Args:
            input_dim (int): Dimension of the input tensor.
            output_dim (int): Dimension of the output tensor.
            hidden_dim (int, optional): Hidden layer dimension (defaults to input_dim).
            depth (int): Total number of layers (must be >= 1).
            norm_style (str): Normalization style ('layer' or 'batch').
            weight_init (str): Weight initialization scheme ('kaiming', 'xavier', 'orthogonal').
            dropout_rate (float): Dropout probability for intermediate layers.
            residual (bool): If True, adds residual connection.
            output_norm (bool): If True, applies LayerNorm on the output.
            activation (Type[nn.Module]): Activation class (e.g., nn.ReLU, nn.GELU, nn.SiLU).
        """
        super().__init__()
        assert depth >= 1, "Depth must be at least 1."

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or input_dim
        self.depth = depth
        self.residual = residual
        self.activation = activation

        # Select normalization layer
        if norm_style == 'batch':
            norm_layer = nn.BatchNorm1d
        elif norm_style == 'layer':
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm style: {norm_style}")

        # Define layers
        self.layers = nn.ModuleList()
        current_dim = input_dim

        for i in range(depth):
            is_last = (i == depth - 1)
            next_dim = output_dim if is_last else self.hidden_dim

            block = [nn.Linear(current_dim, next_dim)]

            if not is_last:
                block += [
                    activation(),
                    norm_layer(next_dim),
                    nn.Dropout(p=dropout_rate)
                ]

            self.layers.append(nn.Sequential(*block))
            current_dim = next_dim

        # Optional residual projection for shape mismatch
        self.projection = None
        if residual and input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)

        # Optional output normalization
        self.output_norm = nn.LayerNorm(output_dim, elementwise_affine=False) if output_norm else None

        # Initialize weights
        self.initialize_weights(weight_init)

    def initialize_weights(self, weight_init: str = 'kaiming'):
        """
        Initialize MLP weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if weight_init == 'xavier':
                    nn.init.xavier_normal_(module.weight)
                elif weight_init == 'kaiming':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif weight_init == 'orthogonal':
                    nn.init.orthogonal_(module.weight)
                else:
                    raise ValueError(f"Unknown weight_init: {weight_init}")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        """
        residual_x = x

        for layer in self.layers:
            x = layer(x)

        if self.residual:
            if self.projection is not None:
                residual_x = self.projection(residual_x)
            x = x + residual_x

        if self.output_norm is not None:
            x = self.output_norm(x)

        return x
    

def l2_normalize(
    x: torch.Tensor, 
    dim: int = -1, 
    eps: float = 1e-12) -> torch.Tensor:
    """
    L2-normalize along `dim` to keep gradients stable.
    """
    norm = x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return x / norm



class SharedAETranslator(nn.Module):
    """
    Shared Autoencoder Translator for unsupervised embedding space translation.
    Uses a shared projection layer after encoders to ensure both encoders 
    project to the same latent space.
    """

    def __init__(
        self, 
        d_s, 
        d_t, 
        d_z:int =512, 
        hidden_dim: int=1024, 
        depth: int=3):
        super().__init__()
        self.E_s = MLP(d_s, d_z, hidden_dim=hidden_dim, depth=depth, residual=True, activation = nn.GELU, weight_init='orthogonal')
        self.D_s = MLP(d_z, d_s, hidden_dim=hidden_dim, depth=depth, residual=True, activation = nn.GELU, weight_init='orthogonal')
        self.E_t = MLP(d_t, d_z, hidden_dim=hidden_dim, depth=depth, residual=True, activation = nn.GELU, weight_init='orthogonal')
        self.D_t = MLP(d_z, d_t, hidden_dim=hidden_dim, depth=depth, residual=True, activation = nn.GELU, weight_init='orthogonal')

        # Shared projection layer to guarantee same latent space
        self.shared_proj = MLP(d_z, d_z, hidden_dim=d_z, depth=1, residual=False, activation=nn.GELU, weight_init='orthogonal')
        
        self.z_norm = nn.LayerNorm(d_z)

    def encode_s(self, x): return self.z_norm(self.shared_proj(self.E_s(x)))
    def encode_t(self, y): return self.z_norm(self.shared_proj(self.E_t(y)))
    def decode_s(self, z): return self.D_s(z)
    def decode_t(self, z): return self.D_t(z)

    def forward(self, x, y):
        z_s, z_t = self.encode_s(x), self.encode_t(y)
        x_rec, y_rec = self.decode_s(z_s), self.decode_t(z_t)
        y_hat, x_hat = self.decode_t(z_s), self.decode_s(z_t)
        z_s_cyc, z_t_cyc = self.encode_t(y_hat), self.encode_s(x_hat)

        return {
            "z_s": z_s, "z_t": z_t,
            "x_rec": x_rec, "y_rec": y_rec,
            "y_hat": y_hat, "x_hat": x_hat,
            "z_s_cyc": z_s_cyc, "z_t_cyc": z_t_cyc,
        }
        
#############  Losses 

# Reconstruction
def loss_rec(
    x: torch.Tensor, 
    x_rec: torch.Tensor, 
    reduction: str = "mean") -> torch.Tensor:
    """ 
    Reconstruction loss between input x and reconstructed x_rec.
    """
    return F.mse_loss(x_rec, x, reduction=reduction)

# Cycle-in-Z
def cyc_z_loss(
    z: torch.Tensor, 
    z_cyc: torch.Tensor, 
    reduction: str = "mean") -> torch.Tensor:
    """
    Cycle-consistency loss in latent space between z and z_cyc.
    """
    return F.mse_loss(z_cyc, z, reduction=reduction)

#  VICReg (anti-collapse)
def vicreg_loss(
    z1: torch.Tensor, 
    z2: torch.Tensor, 
    *, 
    sim_coeff: float = 1.0, 
    var_coeff: float = 1.0, 
    cov_coeff: float = 0.1, 
    eps: float = 1e-3) -> torch.Tensor:    
    """
    VICReg loss between two sets of embeddings z1 and z2.
    
    Args:
        z1: (B, D) tensor
        z2: (B, D) tensor
        sim_coeff: weight for invariance term
        var_coeff: weight for variance term
        cov_coeff: weight for covariance term
        eps: small constant for numerical stability
    """
    
    assert z1.shape == z2.shape, "z1 and z2 must have same shape"
        
    # Invariance -> similarity loss
    sim_loss = F.mse_loss(z1, z2)

    # Variance
    def var_loss(
        z1: torch.Tensor, 
        z2: torch.Tensor) -> torch.Tensor:
        """
        Variance loss: encourage std_dev of each dim to be >= 1
        
        Args:
            z1: (B, D) tensor
            z2: (B, D) tensor
        
        Returns:
            std_loss: scalar tensor
        """
        std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + eps)
        std_loss = torch.mean(F.relu(1.0 - std_z1)) + torch.mean(F.relu(1.0 - std_z2))
        
        return std_loss
    
    def off_diagonal_term(z: torch.Tensor) -> torch.Tensor:
        """
        Off-diagonal elements of covariance matrix flattened.
        Args:
            z: (D, D) covariance matrix
        Returns:
            off_diag: flattened off-diagonal elements
        """
        D = z.shape[0]
        return z.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
        
    # Covariance loss
    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        """
        Covariance loss: encourage off-diagonal covariances to be small.
        Args:
            z: (B, D) tensor
        Returns:
            cov_loss: scalar tensor
        """
        N, D = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (N - 1)
        cov_loss = off_diagonal_term(cov).pow(2).sum() / D                    
        return cov_loss

    var_loss_val = var_loss(z1, z2)
    cov_loss = 0.5 * (covariance_term(z1) + covariance_term(z2))

    return sim_coeff * sim_loss + var_coeff * var_loss_val + cov_coeff * cov_loss



# Sinkhorn OT
def sinkhorn_divergence(a: torch.Tensor, b: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """
    Compute Sinkhorn divergence between two point clouds using geomloss.
    
    Args:
        a: First point cloud (B, D)
        b: Second point cloud (B, D)
        eps: Blur/regularization parameter (corresponds to temperature in OT)
    
    Returns:
        Sinkhorn divergence value
    """
    sinkhorn = SamplesLoss("sinkhorn", p=2, blur=eps, debias=True, backend='tensorized')
    return sinkhorn(a, b)

#  Example helpers 
def compute_losses(
    out: dict,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    lambda_rec=1.0,
    lambda_cyc=1.0,
    lambda_dist=0.5,
    lambda_stab=0.1,
    lambda_geo=0.2,
    use_ot=True,
    ot_eps=0.1,
) -> Tuple[torch.Tensor, dict]:

    """
    Aggregate default losses returning (total, details dict).
    This is a convenience for trainers; trainers may compute more specialized
    variants or use different reductions.
    """
    z_s = out['z_s']
    z_t = out['z_t']
    x_rec = out['x_rec']
    y_rec = out['y_rec']
    y_hat = out['y_hat']
    x_hat = out['x_hat']
    z_s_cyc = out['z_s_cyc']
    z_t_cyc = out['z_t_cyc']

    losses = {}
    losses['rec_s'] = loss_rec(x, x_rec)
    losses['rec_t'] = loss_rec(y, y_rec)
    losses['cyc_z_s'] = cyc_z_loss(z_s, z_s_cyc)
    losses['cyc_z_t'] = cyc_z_loss(z_t, z_t_cyc)

    if use_ot:
        losses['ot_t'] = sinkhorn_divergence(y_hat, y, eps=ot_eps)
        losses['ot_s'] = sinkhorn_divergence(x_hat, x, eps=ot_eps)
    else:
        losses['ot_t'] = torch.tensor(0.0, device=x.device)
        losses['ot_s'] = torch.tensor(0.0, device=x.device)

    losses['vic'] = vicreg_loss(z_s, z_t)
    # geometry local
    #losses['lap'] = knn_laplacian_loss(z_s, l2_normalize(y_hat, dim=-1))
    #losses['triplet'] = triplet_loss_source_neighbors(z_s, l2_normalize(y_hat, dim=-1))

    total = (
        lambda_rec * (losses['rec_s'] + losses['rec_t'])
        + lambda_cyc * (losses['cyc_z_s'] + losses['cyc_z_t'])
        + lambda_dist * (losses['ot_s'] + losses['ot_t'])
        + lambda_stab * losses['vic']
    )

    losses['total'] = total
    return total, losses