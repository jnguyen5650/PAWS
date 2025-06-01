import torch.optim as optim


def build_optimizers(config, model, discriminator=None):
    lr = float(config["training"]["learning_rate"])
    flow_lr = float(config["training"]["flow_learning_rate"])
    flow_lr_mult = float(config["training"].get("flow_lr_multiplier", 1.0))
    gan_enabled = config.get("gan", {}).get("enabled", False)
    
    # Set up parameter groups: main vs flow parameters
    flow_params, main_params = [], []
    match_terms = config.get("model", {}).get("flow_param_keywords", ["flow_net"])
    
    for name, param in model.named_parameters():
        if any(term in name for term in match_terms):
            print("FOUND FLOW PARAMS", name)
            flow_params.append(param)
        else:
            main_params.append(param)
    
    optimizer_G = optim.Adam([
        {'params': main_params, 'lr': lr},
        {'params': flow_params, 'lr': flow_lr * flow_lr_mult}
    ])

    optimizer_D = None
    if gan_enabled and discriminator is not None:
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    
    return optimizer_G, optimizer_D
