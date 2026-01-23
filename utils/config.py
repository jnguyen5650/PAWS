import yaml

def load_config(path):
    """
    Load a YAML config file and return as a dict.

    Args:
        path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration.
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_lambda_dict(config):
    """
    Extract loss weighting coefficients from config.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        dict: Dictionary mapping loss names to weights.
    """
    return {
        "charbonnier": float(config["loss_weights"].get("charbonnier", 1.0)),
        "tv": float(config["loss_weights"].get("tv", 1.0)),
        "dists": float(config["loss_weights"].get("dists", 1.0)),
        "lpips": float(config["loss_weights"].get("lpips", 1.0)),
        "adv": float(config["loss_weights"].get("adv", 1.0)),
        "ldl": float(config["loss_weights"].get("ldl", 1.0)),
        "cleaning_charbonnier": float(config["loss_weights"].get("cleaning_charbonnier", 1.0)),
        "cleaning_dists": float(config["loss_weights"].get("cleaning_dists", 1.0))
    }