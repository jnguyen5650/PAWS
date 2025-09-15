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
        "charbonnier": config["loss_weights"].get("charbonnier", 1.0),
        "tv": config["loss_weights"].get("tv", 1.0),
        "dists": config["loss_weights"].get("dists", 1.0),
        "lpips": config["loss_weights"].get("lpips", 1.0),
        "adv": config["loss_weights"].get("adv", 1.0),
        "ldl": config["loss_weights"].get("ldl", 1.0),
    }