from utils.dist import is_main_process
import torch.optim as optim


def build_optimizers(config, model, discriminator=None):
    g_lr = float(config["training"]["learning_rate"])
    flow_lr = float(config["training"]["flow_learning_rate"])
    flow_lr_mult = float(config["training"].get("flow_lr_multiplier", 1.0))
    g_betas = tuple(config["training"].get("betas", (0.9, 0.99)))
    g_wd = float(config["training"].get("weight_decay", 0.0))
    optimizer_name = str(config["training"].get("optimizer", "adam")).lower()
    if optimizer_name not in ("adam", "adamw"):
        raise ValueError(
            "training.optimizer must be 'adam' or 'adamw', got "
            f"'{optimizer_name}'."
        )
    optimizer_cls = optim.Adam if optimizer_name == "adam" else optim.AdamW
    gan_enabled = config.get("gan", {}).get("enabled", False)

    # Set up parameter groups: main vs flow parameters
    flow_params, flow_param_names, main_params = [], [], []
    match_terms = config.get("model", {}).get(
        "flow_param_keywords",
        ["flow_net"],
    )

    for name, param in model.named_parameters():
        if any(term in name for term in match_terms):
            flow_params.append(param)
            flow_param_names.append(name)
        else:
            main_params.append(param)

    optimizer_G = optimizer_cls(
        [
            {"params": main_params, "lr": g_lr},
            {"params": flow_params, "lr": flow_lr * flow_lr_mult},
        ],
        betas=g_betas,
        weight_decay=g_wd,
    )

    optimizer_D = None
    if gan_enabled and discriminator is not None:
        d_lr = float(config["gan"].get("d_learning_rate", g_lr))
        d_betas = tuple(config["gan"].get("d_betas", g_betas))
        d_wd = float(config["gan"].get("d_weight_decay", g_wd))
        optimizer_D = optimizer_cls(
            discriminator.parameters(),
            lr=d_lr,
            betas=d_betas,
            weight_decay=d_wd,
        )

    if is_main_process():
        print("\n" + "=" * 72)
        print(" Optimizer Configuration Summary")
        print("=" * 72)

        print(f"  Optimizer                  : {optimizer_name}")
        print(f"  Generator LR (main)        : {g_lr}")
        print(
            "  Generator LR (flow)        : "
            f"{flow_lr} (x{flow_lr_mult} -> {flow_lr * flow_lr_mult})"
        )
        print(f"  Betas                      : {g_betas}")
        print(f"  Weight Decay               : {g_wd}")
        print()

        print("  Parameter Groups")
        print(f"    Main Params              : {len(main_params)} tensors")
        print(f"    Flow Params              : {len(flow_params)} tensors")
        print(f"    Flow Match Keywords      : {match_terms}")

        if len(flow_params) == 0:
            print("    WARNING                  : No flow parameters matched!")
        else:
            print("    Flow Param Examples      :")
            for n in flow_param_names[:5]:
                print(f"      - {n}")
            if len(flow_param_names) > 5:
                print(f"      ... ({len(flow_param_names) - 5} more)")

        if gan_enabled and discriminator is not None:
            print()
            print("  Discriminator Optimizer")
            print("    Enabled                  : True")
            print(f"    LR                       : {d_lr}")
            print(f"    Betas                    : {d_betas}")
            print(f"    Weight Decay             : {d_wd}")
        else:
            print()
            print("  Discriminator Optimizer")
            print("    Enabled                  : False")

        print("=" * 72 + "\n")

    return optimizer_G, optimizer_D
