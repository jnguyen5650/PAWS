import math
from torch.optim import lr_scheduler


def linear_lr_decay(current_step, total_steps):
    return 1 - (current_step / total_steps)


def cosine_lr_decay(current_step, total_steps):
    return 0.5 * (1 + math.cos(math.pi * current_step / total_steps))


def build_schedulers(config, train_loader_len, optimizer_G, optimizer_D=None):
    total_epochs = config["training"]["epochs"]
    scheduler_type = config["lr_scheduler"]["type"]

    scheduler_G = None
    scheduler_D = None

    if scheduler_type == "fixed":
        scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda step: 1.0)
        if optimizer_D is not None:
            scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda step: 1.0)
    elif scheduler_type == "cosine_decay":
        total_steps = total_epochs * train_loader_len
        scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda step: cosine_lr_decay(step, total_steps))
        if optimizer_D is not None:
            scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda step: cosine_lr_decay(step, total_steps))
    elif scheduler_type == "cosine":
        T_max = config["lr_scheduler"]["cosine"]["T_max"] * train_loader_len
        eta_min = float(config["lr_scheduler"]["cosine"]["eta_min"])
        scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=T_max, eta_min=eta_min)
        if optimizer_D is not None:
            scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=T_max, eta_min=eta_min)
    else:
        # Linear decay
        total_steps = total_epochs * train_loader_len
        scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda step: linear_lr_decay(step, total_steps))
        if optimizer_D is not None:
            scheduler_D = lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda step: linear_lr_decay(step, total_steps))
    return scheduler_G, scheduler_D
