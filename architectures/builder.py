from modules.registry import MODEL_REGISTRY

def build_model(config: dict):
    """
    ------------------------------------------------------------
    Supported model types and their required parameters
    ------------------------------------------------------------

    1) "mlp"  → MLP(in_features, out_features, hidden_features, num_layers)

        params = {
            "in_features": int,        # Input coordinate dimension
            "out_features": int,       # Output dimension
            "hidden_features": int,    # Units per hidden layer
            "num_layers": int          # Number of hidden layers
        }

    2) "siren" → SIREN(in_features, out_features, hidden_features, num_layers)

        params = {
            "in_features": int,
            "out_features": int,
            "hidden_features": int,
            "num_layers": int
        }

    3) "lisa" → LISAEncoder(latent_dim, hidden_features, num_layers)

        params = {
            "latent_dim": int,         # Latent feature dimension from ConvEncoder
            "hidden_features": int,    # Hidden units in LISA decoder
            "num_layers": int          # Number of decoder layers
        }

        Internally:
            - Uses ConvEncoder(latent_dim)
            - Uses LISA decoder with:
                in_features=1
                out_features=1
                latent_dim=latent_dim

    """
    model_type = config["type"]
    params = config.get("params", {})

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}")

    return MODEL_REGISTRY[model_type](**params)