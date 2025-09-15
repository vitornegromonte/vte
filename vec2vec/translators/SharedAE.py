import math
from typing import Optional, List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F

def add_residual(input_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Add residual connection with dimension matching."""
    if input_x.shape[1] != x.shape[1]:
        # This is a warning sign! Residuals should ideally have matching dims.
        # For simplicity, we'll implement the original logic but be aware of it.
        if input_x.shape[1] < x.shape[1]:
            padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device, dtype=x.dtype)
            input_x = torch.cat([input_x, padding], dim=1)
        elif input_x.shape[1] > x.shape[1]:
            input_x = input_x[:, :x.shape[1]]
    return x + input_x
  
class MLP(nn.Module):
    """
    A generic, configurable Multi-Layer Perceptron (MLP).
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
        projection: bool = True,
        activation: nn.Module = nn.SiLU, 
    ):
        """
        Initializes the MLP.

        Args:
            input_dim (int): Dimension of the input tensor.
            output_dim (int): Dimension of the output tensor.
            hidden_dim (int, optional): Dimension of hidden layers. If None, defaults to input_dim.
            depth (int): Total number of layers.
            norm_style (str): Normalization style ('layer' or 'batch').
            weight_init (str): Weight initialization scheme ('kaiming', 'xavier', 'orthogonal').
            dropout_rate (float): Dropout probability.
            residual (bool): If True, adds residual connections.
            output_norm (bool): If True, applies LayerNorm to the final output.
        """
        super().__init__()

        assert depth >= 1, "Depth must be at least 1."

        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.residual = residual
        self.activation = activation
        
        if self.residual and self.input_dim != self.output_dim:
            self.projection = nn.Linear(self.input_dim, self.output_dim)

        # Paper says it used LayerNorm only but I want this class to be as generic as possible
        if norm_style == 'batch':
            norm_layer = nn.BatchNorm1d
        elif norm_style == 'layer':
            norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unknown norm style: {norm_style}")

        self.layers = nn.ModuleList()
        current_dim = input_dim

        for i in range(depth):
            is_last_layer = (i == depth - 1)
            target_dim = output_dim if is_last_layer else hidden_dim

            block = [nn.Linear(current_dim, target_dim)]

            if not is_last_layer:
                block.extend([
                    activation(),
                    norm_layer(target_dim),
                    nn.Dropout(p=dropout_rate)
                ])

            self.layers.append(nn.Sequential(*block))
            current_dim = target_dim

        if output_norm:
            self.output_norm = nn.LayerNorm(output_dim, elementwise_affine=False)
        else:
            self.output_norm = None

        self.initialize_weights(weight_init)

    def initialize_weights(self, weight_init: str = 'kaiming'):
      """
      Initializes the weights of the MLP.

      Args:
          weight_init (str): Weight initialization scheme ('kaiming', 'xavier', 'orthogonal').
      """
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
      residual_x = x

      for layer in self.layers:
          x = layer(x)
                    
      if self.residual:
          if self.input_dim != self.output_dim:
              residual_x = self.projection(residual_x)
          x = x + residual_x

      if self.output_norm is not None:
          x = self.output_norm(x)

      return x

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize along `dim`.
    Keep gradients stable.
    """
    norm = x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)
    return x / norm

class SharedAETranslator(nn.Module):
    """
    Shared Autoencoder Translator for unsupervised embedding space translation.
    """

    def __init__(
        self, 
        d_s, 
        d_t, 
        d_z=512, 
        hidden_dim=1024, 
        depth=3):
        super().__init__()
        self.E_s = MLP(d_s, d_z, hidden_dim=hidden_dim, depth=depth, norm_style="batch", residual=True, activation = nn.GELU)
        self.D_s = MLP(d_z, d_s, hidden_dim=hidden_dim, depth=depth, norm_style="batch", residual=True, activation = nn.GELU)
        self.E_t = MLP(d_t, d_z, hidden_dim=hidden_dim, depth=depth, norm_style="batch", residual=True, activation = nn.GELU)
        self.D_t = MLP(d_z, d_t, hidden_dim=hidden_dim, depth=depth, norm_style="batch", residual=True, activation = nn.GELU)

        self.z_norm = nn.LayerNorm(d_z)

    def encode_s(self, x): return self.z_norm(self.E_s(x))
    def encode_t(self, y): return self.z_norm(self.E_t(y))
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
        
#  Losses 
# Reconstruction
def loss_rec(x: torch.Tensor, x_rec: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.mse_loss(x_rec, x, reduction=reduction)

# Cycle-in-Z
def cyc_z_loss(z: torch.Tensor, z_cyc: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    return F.mse_loss(z_cyc, z, reduction=reduction)

#  VICReg (anti-collapse) 
def vicreg(z1: torch.Tensor, z2: torch.Tensor, *, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4) -> torch.Tensor:
    """VICReg-style composite loss between two views z1,z2. Returns scalar.

    - invariance (sim): mean squared error between z1 and z2
    - variance: hinge on std per-dim to encourage non-collapse
    - covariance: off-diagonal Frobenius norm of covariance matrix
    Reference: VICReg paper (something similar in spirit).
    """
    assert z1.shape == z2.shape, "z1 and z2 must have same shape"
    # Invariance (MSE)
    sim_loss = F.mse_loss(z1, z2)

    # Variance loss
    def variance_term(z: torch.Tensor) -> torch.Tensor:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        # target: each dim has std >= 1.0 -> hinge
        hinge = F.relu(1.0 - std)
        return hinge.mean()

    var_loss = 0.5 * (variance_term(z1) + variance_term(z2))

    # Covariance loss: off-diagonal elements of covariance matrix
    def covariance_term(z: torch.Tensor) -> torch.Tensor:
        B, D = z.shape
        z = z - z.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / (B - 1)
        diag = torch.diag_embed(torch.diagonal(cov))
        off_diag = cov - diag
        return (off_diag ** 2).sum() / D

    cov_loss = 0.5 * (covariance_term(z1) + covariance_term(z2))

    return sim_coeff * sim_loss + var_coeff * var_loss + cov_coeff * cov_loss


# Local geometry losses
def knn_laplacian_loss(z_src: torch.Tensor, z_tgt_translated: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Laplacian loss: construct k-NN graph on z_src, encourage translated points
    to respect same neighborhood affinities in target space.

    Efficient minibatch approx: use pairwise cosine affinities within batch.
    """
    # cosine similarity
    z_src_n = l2_normalize(z_src, dim=-1)
    sim = z_src_n @ z_src_n.T  # B x B
    B = sim.size(0)

    # for each row, get top-k indices (exclude self)
    knn_vals, knn_idx = sim.topk(k + 1, dim=-1)
    knn_idx = knn_idx[:, 1:]

    # build laplacian-style loss: for each i, sum ||y_i - y_j||^2 for neighbors j
    y = z_tgt_translated
    # pairwise squared distances
    y2 = (y ** 2).sum(dim=1, keepdim=True)
    dists = y2 + y2.T - 2 * (y @ y.T)

    loss = 0.0
    for i in range(B):
        neigh = knn_idx[i]  # k
        loss = loss + dists[i, neigh].mean()
    return loss / B


def triplet_loss_source_neighbors(z_src: torch.Tensor, z_tgt_translated: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """
    Triplet style: 
    for each i, positive is nearest neighbor in src (j), negative is random other.
    Implemented in-batch for simplicity.
    """
    z_src_n = l2_normalize(z_src, dim=-1)
    sim = z_src_n @ z_src_n.T
    B = sim.size(0)
    # get nearest neighbor (exclude self)
    knn_vals, knn_idx = sim.topk(2, dim=-1)
    pos_idx = knn_idx[:, 1]

    z_t = l2_normalize(z_tgt_translated, dim=-1)
    pos = z_t[pos_idx]
    anchor = z_t
    # sample negatives: pick random idxs (not equal to i)
    neg_idx = torch.randint(0, B - 1, (B,), device=z_src.device)
    # shift so that neg_idx != i
    neg_idx = (neg_idx + 1) % B
    neg = z_t[neg_idx]

    # cosine distances
    pos_sim = (anchor * pos).sum(dim=-1)
    neg_sim = (anchor * neg).sum(dim=-1)
    loss = F.relu(margin + neg_sim - pos_sim).mean()
    return loss


#  Sinkhorn (minibatch) 
def sinkhorn_log_stabilized(cost: torch.Tensor, eps: float = 0.1, n_iters: int = 50) -> torch.Tensor:
    """
    Compute Sinkhorn transport plan in log-domain for stability.

    Args:
        cost: (B, B) cost matrix between two batches
    Returns:
        transport matrix Pi (B,B)
    """
    # use uniform marginals
    B = cost.size(0)
    mu = torch.full((B,), 1.0 / B, device=cost.device)
    nu = mu

    # K = exp(-cost / eps)
    K = torch.exp(-cost / eps)
    # stabilize via row/col scaling (Sinkhorn)
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)
    K_t = K
    for _ in range(n_iters):
        u = mu / (K_t @ v)
        v = nu / (K_t.T @ u)
    P = torch.diag(u) @ K_t @ torch.diag(v)
    return P


def cost_matrix_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine-based cost matrix; cost = 1 - cosine.
    a: (B, D), b: (B, D)
    returns: (B, B)
    """
    a_n = l2_normalize(a, dim=-1)
    b_n = l2_normalize(b, dim=-1)
    C = 1.0 - (a_n @ b_n.T)
    return C


def sinkhorn_divergence(a: torch.Tensor, b: torch.Tensor, eps: float = 0.1, n_iters: int = 50) -> torch.Tensor:
    """Compute simple Sinkhorn divergence between minibatches a and b.

    We approximate S_eps(a,b) = OT_eps(a,b) - 0.5 OT_eps(a,a) - 0.5 OT_eps(b,b)
    where OT_eps = <P, C>

    NOTE: this is a minibatch estimator and should be used with care.
    """
    B = a.size(0)
    assert b.size(0) == B
    C_ab = cost_matrix_cosine(a, b)
    P_ab = sinkhorn_log_stabilized(C_ab, eps=eps, n_iters=n_iters)
    ot_ab = (P_ab * C_ab).sum()

    C_aa = cost_matrix_cosine(a, a)
    P_aa = sinkhorn_log_stabilized(C_aa, eps=eps, n_iters=n_iters)
    ot_aa = (P_aa * C_aa).sum()

    C_bb = cost_matrix_cosine(b, b)
    P_bb = sinkhorn_log_stabilized(C_bb, eps=eps, n_iters=n_iters)
    ot_bb = (P_bb * C_bb).sum()

    return ot_ab - 0.5 * (ot_aa + ot_bb)


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

    losses['vic'] = vicreg(z_s, z_t)
    # geometry local
    losses['lap'] = knn_laplacian_loss(z_s, l2_normalize(y_hat, dim=-1))
    losses['triplet'] = triplet_loss_source_neighbors(z_s, l2_normalize(y_hat, dim=-1))

    total = (
        lambda_rec * (losses['rec_s'] + losses['rec_t'])
        + lambda_cyc * (losses['cyc_z_s'] + losses['cyc_z_t'])
        + lambda_dist * (losses['ot_s'] + losses['ot_t'])
        + lambda_stab * losses['vic']
        + lambda_geo * (losses['lap'] + losses['triplet'])
    )

    losses['total'] = total
    return total, losses