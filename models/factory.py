from .basicvsrpp import BasicVSRPlusPlus
from .discriminator import SpatioTemporalUNetDiscriminator
from utils.checkpoint import CheckpointWrapper

MODEL_REGISTRY = {
    "BasicVSRPlusPlus": BasicVSRPlusPlus
}

DISCRIMINATOR_REGISTRY = {
    "SpatioTemporalUNetDiscriminator": SpatioTemporalUNetDiscriminator
}

def build_generator(config, device):
    model_name = config["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown generator model name: {model_name}")
    model_cls = MODEL_REGISTRY[model_name]

    model_kwargs = config["model"].copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("flow_param_keywords", None)

    model = model_cls(**model_kwargs).to(device)
    return CheckpointWrapper(model)


def build_ema_model(config, src_model, device):
    if not config["training"].get("use_ema", False):
        return None
    model_name = config["model"]["name"]
    model_cls = MODEL_REGISTRY[model_name]
    ema = model_cls().to(device)
    src = src_model.module if hasattr(src_model, "module") else src_model
    ema.load_state_dict(src.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def build_discriminator(config, device):
    if not config.get("gan", {}).get("enabled", False):
        return None
    disc_name = config.get("gan", {}).get("discriminator", "SpatioTemporalUNetDiscriminator")
    if disc_name not in DISCRIMINATOR_REGISTRY:
        raise ValueError(f"Unknown discriminator model name: {disc_name}")
    disc_cls = DISCRIMINATOR_REGISTRY[disc_name]
    disc = disc_cls(num_in_ch=3).to(device)
    return CheckpointWrapper(disc)
