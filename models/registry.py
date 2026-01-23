import torch.nn as nn

MODEL_REGISTRY = {}
DISCRIMINATOR_REGISTRY = {}

def _register(registry, name, cls, allow_override=False):
    if not isinstance(name, str) or not name:
        raise ValueError("Registry name must be a non-empty string.")

    # cls should be a class that subclasses nn.Module
    if not isinstance(cls, type):
        raise TypeError(f"Registered object for '{name}' must be a class, got: {type(cls)}")

    if not issubclass(cls, nn.Module):
        raise TypeError(f"Registered class '{cls.__name__}' must subclass nn.Module.")

    if name in registry and not allow_override and registry[name] is not cls:
        prev = registry[name]
        raise ValueError(
            f"Duplicate registry key '{name}': "
            f"{prev.__module__}.{prev.__name__} vs "
            f"{cls.__module__}.{cls.__name__}"
        )

    registry[name] = cls
    return cls

def register_model(name=None, allow_override=False):
    def decorator(cls):
        key = name if name is not None else cls.__name__
        return _register(MODEL_REGISTRY, key, cls, allow_override=allow_override)
    return decorator

def register_discriminator(name=None, allow_override=False):
    def decorator(cls):
        key = name if name is not None else cls.__name__
        return _register(DISCRIMINATOR_REGISTRY, key, cls, allow_override=allow_override)
    return decorator

def get_model_cls(name):
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown generator model name: {name}. Available: {available}")
    return MODEL_REGISTRY[name]

def get_discriminator_cls(name):
    if name not in DISCRIMINATOR_REGISTRY:
        available = ", ".join(sorted(DISCRIMINATOR_REGISTRY.keys()))
        raise ValueError(f"Unknown discriminator model name: {name}. Available: {available}")
    return DISCRIMINATOR_REGISTRY[name]

