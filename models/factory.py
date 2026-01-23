from utils.checkpoint import CheckpointWrapper
from .registry import get_model_cls, get_discriminator_cls


def build_generator(config, device):
    model_name = config["model"]["name"]
    model_cls = get_model_cls(model_name)

    model_kwargs = config["model"].copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("flow_param_keywords", None)

    model = model_cls(**model_kwargs).to(device)
    return CheckpointWrapper(model)


def build_ema_model(config, src_model, device):
    if not config["training"].get("use_ema", False):
        return None
    model_name = config["model"]["name"]
    model_cls = get_model_cls(model_name)
    model_kwargs = config["model"].copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("flow_param_keywords", None)

    ema = model_cls(**model_kwargs).to(device)

    def _unwrap_all_modules(m):
        # Repeatedly unwrap common wrappers that expose .module
        while hasattr(m, "module"):
            nxt = m.module
            if nxt is m:
                break
            m = nxt
        return m

    src = _unwrap_all_modules(src_model)
    ema.load_state_dict(src.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def build_discriminator(config, device):
    if not config.get("gan", {}).get("enabled", False):
        return None

    disc_name = config.get("gan", {}).get("discriminator", "SpatioTemporalUNetDiscriminator")
    disc_cls = get_discriminator_cls(disc_name)

    disc = disc_cls(num_in_ch=3).to(device)
    return CheckpointWrapper(disc)
