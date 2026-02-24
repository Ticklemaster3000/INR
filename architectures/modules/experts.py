import torch
import torch.nn as nn
from .base import BaseINR


class MoE(BaseINR):
    """
    Fully flexible Mixture of Experts.

    Accepts:
        - List of model instances
        - Any nn.Module compatible with input/output
        - Nested MoEs
    """

    def __init__(self, experts, gate=None):
        """
        Args:
            experts: List[nn.Module]
                Already-initialized expert models.

            gate: Optional nn.Module
                Custom gating network.
                If None → default linear gate inferred.
        """
        super().__init__()

        if not experts or not isinstance(experts, (list, tuple)):
            raise ValueError("experts must be a non-empty list of models")

        # Ensure all are nn.Modules
        for e in experts:
            if not isinstance(e, nn.Module):
                raise TypeError("All experts must be nn.Module instances")

        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)

        # Infer input dimension dynamically
        sample_expert = experts[0]
        if hasattr(sample_expert, "net"):
            # Try to infer from first layer
            first_layer = list(sample_expert.modules())[1]
            if isinstance(first_layer, nn.Linear):
                in_features = first_layer.in_features
            else:
                raise ValueError("Cannot infer input dimension automatically.")
        else:
            raise ValueError("Cannot infer input dimension automatically.")

        # Default gate if none provided
        if gate is None:
            self.gate = nn.Linear(in_features, self.num_experts)
        else:
            self.gate = gate

    def forward(self, x):
        """
        x shape: (B, T, in_features)
        """

        # Compute gating weights
        weights = torch.softmax(self.gate(x), dim=-1)
        # (B, T, num_experts)

        # Collect expert outputs
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts],
            dim=-1
        )
        # (B, T, out_features, num_experts)

        # Weighted combination
        output = torch.sum(
            weights.unsqueeze(-2) * expert_outputs,
            dim=-1
        )

        return output