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
# 5. THE PLUG-AND-PLAY REGISTRY
# ==========================================
MODEL_REGISTRY: Dict[str, Type[BaseINR]] = {
    "mlp": MLP,
    "siren": SIREN,
    "moe": MoE
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