from utils.dist import is_main_process
import torch.optim as optim


def build_optimizers(config, model, discriminator=None):
    g_lr = float(config["training"]["learning_rate"])
    flow_lr = float(config["training"]["flow_learning_rate"])
    flow_lr_mult = float(config["training"].get("flow_lr_multiplier", 1.0))
    g_betas = tuple(config["training"].get("betas", (0.9, 0.99)))
    g_wd = float(config["training"].get("weight_decay", 0.0))
    gan_enabled = config.get("gan", {}).get("enabled", False)
    
    # Set up parameter groups: main vs flow parameters
    flow_params, main_params = [], []
    match_terms = config.get("model", {}).get("flow_param_keywords", ["flow_net"])
    
    for name, param in model.named_parameters():
        if any(term in name for term in match_terms):
            if is_main_process():
                print("FOUND FLOW PARAMS", name)
            flow_params.append(param)
        else:
            main_params.append(param)
    
    optimizer_G = optim.Adam(
        [
            {'params': main_params, 'lr': g_lr},
            {'params': flow_params, 'lr': flow_lr * flow_lr_mult}
        ],
        betas=g_betas,
        weight_decay=g_wd,
    )

    optimizer_D = None
    if gan_enabled and discriminator is not None:
        d_lr = float(config["gan"].get("d_learning_rate", g_lr))
        d_betas = tuple(config["gan"].get("d_betas", g_betas))
        d_wd = float(config["gan"].get("d_weight_decay", g_wd))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=d_lr, betas=d_betas, weight_decay=d_wd)
    
    return optimizer_G, optimizer_D
