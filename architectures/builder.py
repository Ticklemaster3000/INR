from modules.registry import MODEL_REGISTRY

def build_model(config: dict):
    """
    Generic model builder.
    
    config example:
    {
        "type": "siren",
        "params": {...}
    }
    """
    model_type = config["type"]
    params = config.get("params", {})

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}")

    return MODEL_REGISTRY[model_type](**params)