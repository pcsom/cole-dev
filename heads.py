
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
import numpy as np
import xgboost as xgb

# helper functions

def exists(val):
    return val is not None

def l2norm(t):
    return F.normalize(t, dim = -1)

def default(val, d):
    return val if exists(val) else d

def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)
    
class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = torch.nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)
        return self.fn(x, **kwargs)
    
class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * torch.nn.functional.gelu(gates)

class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)

class Attention(torch.nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(context_dim, inner_dim, bias = False)
        self.to_kv = torch.nn.Linear(query_dim, inner_dim * 2, bias = False)

        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias = False)

    def forward(self, x, context = None):
        h = self.heads
        q = self.to_q(context)
        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)
    
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = 4096,
        latent_dim = 4096,
        cross_dim_head = 2048,
        num_cross_heads = 32,
        num_latents_value = 768,
        layers = 32,
    ):
        super().__init__()

        num_latents, latent_dim, cross_heads, cross_dim_head, dim = num_latents_value, latent_dim, num_cross_heads, cross_dim_head,hidden_dim

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.pos_emb = nn.Embedding(layers, dim)
        self.normalize = True

        # init latent_attention and latents
        self.cross_attend_blocks = torch.nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head),
                    context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim)),
        ])


    def forward(self, hiddens):
        pos_emb = self.pos_emb(torch.arange(hiddens.shape[1], device = hiddens.device))
        hiddens = hiddens + pos_emb

        cross_attn, cross_ff = self.cross_attend_blocks

        x = repeat(self.latents, 'n d -> b n d', b = hiddens.shape[0])
        x = cross_attn(hiddens, context = x) + x
        x = cross_ff(x) + x

        x = torch.mean(x, dim=1)
        if self.normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x

class MLPSurrogate(nn.Module):
    """3-layer MLP for architecture performance prediction."""
    
    def __init__(self, input_dim, hidden_dims=[128, 128, 128], output_dim=2, dropout=0.1):
        super(MLPSurrogate, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class XGBoostSurrogate:
    """
    XGBoost wrapper for architecture performance prediction.
    
    Note: This is not a PyTorch module but a sklearn-compatible wrapper.
    Provides the same interface as neural network surrogates.
    Supports CUDA acceleration when device='cuda'.
    """
    
    def __init__(self, input_dim, output_dim=2,
                 n_estimators=2000,
                 max_depth=6, 
                 learning_rate=0.01, 
                #  min_child_weight=5,
                #  gamma=0.1,
                 subsample=0.8,
                 colsample_bytree=0.5,
                 random_state=42, 
                 device='cpu'):
        """
        Args:
            input_dim: Input feature dimension (unused, kept for API compatibility)
            output_dim: Number of targets (1 or 2)
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random seed
            device: Device to use ('cpu' or 'cuda')
        """
        self.output_dim = output_dim
        self.device = device
        
        # Create separate XGBoost models for each target if multi-output
        xgb_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            # 'min_child_weight': min_child_weight,
            # 'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state,
            'tree_method': 'hist',
            'device': device,
            'verbosity': 0
        }

        if output_dim == 1:
            self.models = [xgb.XGBRegressor(**xgb_params)]
        else:
            # Independent regressors is standard for XGB
            self.models = [xgb.XGBRegressor(**{**xgb_params, 'random_state': random_state + i}) 
                           for i in range(output_dim)]
    
    def to(self, device):
        """Move XGBoost models to specified device (CPU or CUDA)"""
        self.device = device
        
        # Update device for all models
        for model in self.models:
            model.set_params(device=device)
        
        return self
    
    def train(self):
        """Compatibility method for .train() calls"""
        pass
    
    def eval(self):
        """Compatibility method for .eval() calls"""
        pass
    
    def parameters(self):
        """Compatibility method for optimizer - returns empty list"""
        return []
    
    def fit(self, X, y):
        """
        Fit XGBoost models.
        
        Args:
            X: numpy array of shape (N, D)
            y: numpy array of shape (N, output_dim)
        """
        if self.output_dim == 1:
            self.models[0].fit(X, y.ravel())
        else:
            for i, model in enumerate(self.models):
                model.fit(X, y[:, i])
    
    def predict(self, X):
        """
        Predict with XGBoost models.
        
        Args:
            X: numpy array of shape (N, D)
        
        Returns:
            numpy array of shape (N, output_dim)
        """
        if self.output_dim == 1:
            preds = self.models[0].predict(X).reshape(-1, 1)
        else:
            preds = np.column_stack([model.predict(X) for model in self.models])
        return preds


class MultiLayerPerceiverSurrogate(nn.Module):
    """
    Surrogate model that uses a Perceiver Resampler to aggregate 
    embeddings from multiple LLM layers before regression.
    
    Input: (Batch_Size, Num_Selected_Layers, Hidden_Dim)
    Output: (Batch_Size, Output_Dim)
    """
    def __init__(self, input_dim, num_layers_to_pool, hidden_dims=[512, 256, 128], output_dim=2, dropout=0.1):
        super().__init__()
        
        # 1. The Perceiver Head (aggregates layer depth info)
        # We assume latent_dim matches input_dim for simplicity; it can be tuned.
        # 'num_latents_value' usually small for compression, but here we output 1 vector per batch item eventually.
        self.perceiver = PerceiverResampler(
            dim=input_dim,
            hidden_dim=input_dim,
            latent_dim=input_dim, 
            layers=num_layers_to_pool, # Number of positions for the position embedding
            num_latents_value=1,       # We want to compress to 1 specific latent vector ideally, or pool later
            cross_dim_head=64,
            num_cross_heads=8
        )
        
        # 2. The MLP Regressor (Same as MLPSurrogate)
        layers = []
        prev_dim = input_dim 
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (Batch, Num_Layers, Hidden_Dim)
        
        # Pass through Perceiver
        # Output of perceiver in your snippet is normalized pooled vector: (Batch, Hidden_Dim)
        pooled_features = self.perceiver(x) 
        
        # Pass through MLP
        return self.mlp(pooled_features)