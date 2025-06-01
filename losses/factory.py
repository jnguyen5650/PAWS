from losses import CharbonnierLoss, TVLoss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def build_losses(config, device):
    losses = {}
    if config["losses"].get("charbonnier", False):
        losses["charbonnier"] = CharbonnierLoss().to(device)
    if config["losses"].get("tv", False):
        losses["tv"] = TVLoss().to(device)
    if config["losses"].get("lpips", False):
        losses["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean').to(device)
    if config["losses"].get("ldl", False):
        losses["ldl"] = CharbonnierLoss().to(device)
    
    assert not config["losses"].get("ldl", False) or config["training"].get("use_ema", False), \
        "When using LDL (ldl: true), you must also enable EMA (use_ema: true)"
    
    return losses
