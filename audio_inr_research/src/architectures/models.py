import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Type, Any

# ==========================================
# 1. THE BASE CONTRACT
# ==========================================
class BaseINR(nn.Module):
    """
    The Base Class ensures every model we build accepts 
    coordinates and returns a signal value.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward pass.")

# ==========================================
# 2. MLP BASELINE
# ==========================================
class MLP(BaseINR):
    def __init__(self, in_features=1, out_features=1, hidden_features=256, num_layers=4):
        super().__init__()
        layers = []
        last_dim = in_features
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_features))
            layers.append(nn.ReLU())
            last_dim = hidden_features
        layers.append(nn.Linear(last_dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. SIREN (Sine Representation Network)
# ==========================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                           np.sqrt(6 / self.linear.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN(BaseINR):
    def __init__(self, in_features=1, out_features=1, hidden_features=256, num_layers=4, first_omega_0=30.0, hidden_omega_0=30.0):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))
        for _ in range(num_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))
        self.net.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)

# ==========================================
# 4. MOE (Mixture of Experts)
# ==========================================
class MoE(BaseINR):
    """
    Modular MoE: It can take any BaseINR class as an expert.
    """
    def __init__(self, expert_type: str, num_experts=4, expert_params=None):
        super().__init__()
        # Get the expert class from our registry
        expert_cls = MODEL_REGISTRY[expert_type]
        
        self.experts = nn.ModuleList([expert_cls(**expert_params) for _ in range(num_experts)])
        self.gate = nn.Linear(expert_params['in_features'], num_experts)

    def forward(self, x):
        # x shape: (batch, seq_len, in_features)
        weights = torch.softmax(self.gate(x), dim=-1) # (batch, seq_len, num_experts)
        
        # Collect outputs from all experts
        # expert_outputs shape: (batch, seq_len, out_features, num_experts)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        
        # Weighted sum: (batch, seq_len, out_features)
        combined = torch.sum(weights.unsqueeze(-2) * expert_outputs, dim=-1)
        return combined

# ==========================================
# 5. LISA (Local Implicit representation for Super resolution of Arbitrary scale)
# ==========================================
class PositionEmbedding(nn.Module):
    """Positional encoding using sine and cosine functions."""
    def __init__(self, in_channels=1, N_freqs=6):
        super().__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)

    def forward(self, x):
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, -1)


class LISA(BaseINR):
    """
    LISA: Local Implicit representation for Super resolution of Arbitrary scale.
    Uses local features and ensemble for better audio reconstruction.
    """
    def __init__(
        self,
        in_features=1,
        out_features=1,
        hidden_features=256,
        num_layers=4,
        latent_dim=64,
        feat_unfold=True,
        use_positional_encoding=True,
    ):
        super().__init__()
        self.feat_unfold = feat_unfold
        self.latent_dim = latent_dim
        
        # Positional encoding
        self.embed = None
        if use_positional_encoding:
            self.embed = PositionEmbedding(in_features, N_freqs=6)
            imnet_in_dim = self.embed.out_channels + (latent_dim * 3 if feat_unfold else latent_dim)
        else:
            imnet_in_dim = in_features + (latent_dim * 3 if feat_unfold else latent_dim)
        
        # IMNET: Implicit network that processes coordinates + features
        layers = []
        last_dim = imnet_in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(last_dim, hidden_features))
            layers.append(nn.ReLU())
            last_dim = hidden_features
        layers.append(nn.Linear(last_dim, out_features))
        self.imnet = nn.Sequential(*layers)

    def query_features(self, coord, latent):
        """
        Query features for given coordinates using local ensemble.
        
        Args:
            coord: [B, chunk_len, 1] coordinates
            latent: [B, num_latents, latent_dim] latent features
            
        Returns:
            pred: [B, chunk_len, out_features]
        """
        batch_size = latent.shape[0]
        latent_dim = latent.shape[2]
        num_latents = latent.shape[1]
        chunk_len = coord.shape[1]
        
        # Prepare latent features: [B, latent_dim, num_latents]
        feat = latent.permute(0, 2, 1)
        
        # Local ensemble: use previous, current, and next features
        if self.feat_unfold:
            feat_prev = torch.cat([feat[:, :, 0:1], feat[:, :, :-1]], dim=2)
            feat_next = torch.cat([feat[:, :, 1:], feat[:, :, -1:]], dim=2)
            feat = torch.cat([feat_prev, feat, feat_next], dim=1)  # [B, latent_dim*3, num_latents]
        
        # Interpolate features to match coordinate positions
        # Normalize coordinates to [-1, 1] for grid_sample
        coord_norm = coord * 2 - 1  # Assuming coord is in [0, 1]
        
        # Use interpolation to get features at query coordinates
        # For 1D audio, we use linear interpolation along the sequence
        # feat: [B, latent_dim*k, num_latents]
        # coord_norm: [B, chunk_len, 1] - values in [-1, 1]
        
        # Prepare for grid_sample: need [B, C, H, W] input and [B, H_out, W_out, 2] grid
        feat = feat.unsqueeze(2)  # [B, latent_dim*k, 1, num_latents]
        
        # Create grid: [B, chunk_len, 1, 2]
        coord_x = coord_norm.squeeze(-1)  # [B, chunk_len]
        coord_y = torch.zeros_like(coord_x)  # Dummy y coordinate
        coord_grid = torch.stack([coord_x, coord_y], dim=-1)  # [B, chunk_len, 2]
        coord_grid = coord_grid.unsqueeze(2)  # [B, chunk_len, 1, 2]
        
        # Sample features
        feat_sampled = torch.nn.functional.grid_sample(
            feat, coord_grid, mode='bilinear', padding_mode='border', align_corners=False
        )  # [B, latent_dim*k, chunk_len, 1]
        feat_sampled = feat_sampled.squeeze(3).permute(0, 2, 1)  # [B, chunk_len, latent_dim*k]
        
        # Process coordinates
        rel_coord = coord
        if self.embed is not None:
            rel_coord = self.embed(rel_coord)
        
        # Concatenate features and coordinates
        inp = torch.cat([rel_coord, feat_sampled], dim=-1)
        
        # Pass through implicit network
        pred = self.imnet(inp)
        return pred

    def forward(self, x):
        """
        Forward pass assuming x contains both coordinates and latent features.
        For compatibility, if x is just coordinates, use zero latent.
        """
        if x.shape[-1] == 1:
            # Just coordinates, create dummy latent
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            num_latents = max(seq_len // 100, 1)
            latent = torch.zeros(batch_size, num_latents, self.latent_dim, device=x.device)
            return self.query_features(x, latent)
        else:
            # Assume x contains [coord, latent_features]
            # This is a simplified version; actual usage requires proper latent generation
            return self.imnet(x)


# ==========================================
# 6. THE PLUG-AND-PLAY REGISTRY
# ==========================================

# Import new architectures
from .siren_square import SIREN_square
from .siren2_encodec import SIREN2_EnCodec

MODEL_REGISTRY: Dict[str, Type[BaseINR]] = {
    "mlp": MLP,
    "siren": SIREN,
    "siren2": SIREN_square,          # Standalone SIREN²
    "siren2_encodec": SIREN2_EnCodec,  # Combined SIREN² + EnCodec
    "moe": MoE,
    "lisa": LISA,
}

def build_model(model_name: str, config: Dict[str, Any]) -> BaseINR:
    """
    The orchestrator function to initialize any model by name.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Options: {list(MODEL_REGISTRY.keys())}")
    
    return MODEL_REGISTRY[model_name](**config)

# ==========================================
# TEST SCRIPT (Optional)
# ==========================================
if __name__ == "__main__":
    # Test SIREN
    siren_cfg = {"in_features": 1, "out_features": 1, "hidden_features": 128}
    model = build_model("siren", siren_cfg)
    test_input = torch.randn(1, 16000, 1) # 1 second of audio coords
    output = model(test_input)
    print(f"SIREN Output Shape: {output.shape}")