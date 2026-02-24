from typing import Dict, Type
from .base import BaseINR
from .decoders import MLP, SIREN, LISA
from .experts import MoE
from .encoders import LISAEncoder

MODEL_REGISTRY: Dict[str, Type[BaseINR]] = {
    "mlp": MLP,
    "siren": SIREN,
    "lisa": LISA,
    "lisa_enc": LISAEncoder,
    "moe": MoE,
}