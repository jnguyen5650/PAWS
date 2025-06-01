import os
import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm


class CheckpointWrapper(torch.nn.Module):
    """
    Wraps an `nn.Module` to enable activation checkpointing, reducing memory usage during training
    by discarding and recomputing intermediate activations in the backward pass.

    This mechanism permits training of larger models or batch sizes under fixed memory constraints,
    at the cost of increased computation during backpropagation.

    Args:
        module (nn.Module): The module to wrap. Must support standard forward and backward passes.

    Usage:
        model = CheckpointWrapper(model)
        output = model(input)

    Notes:
        - At least one input tensor to the wrapped module must have `requires_grad=True`. If not,
          this wrapper will call `requires_grad_()` in-place on all tensor inputs.
        - By default, `use_reentrant=False` is set for compatibility with PyTorch 2.x+.
        - The wrapper is transparent to the module's interface but not to module inspection or state_dict behavior.

    References:
        - See: https://pytorch.org/docs/stable/checkpoint.html
    """
    def __init__(self, module):
        super(CheckpointWrapper, self).__init__()
        self.module = module

    def forward(self, *inputs):
        # Ensure that at least one tensor input has requires_grad=True.
        new_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and not inp.requires_grad:
                inp = inp.requires_grad_()
            new_inputs.append(inp)
            
        def custom_forward(*inputs):
            return self.module(*inputs)
        
        return checkpoint(custom_forward, *new_inputs, use_reentrant=False)


def save_checkpoint(
    epoch,
    model,
    optimizer_G,
    scheduler_G,
    scaler,
    ema_model=None,
    discriminator=None,
    optimizer_D=None,
    scheduler_D=None,
    path="checkpoint_latest.pth",
):
    """Save training state so we can resume later."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "scheduler_G_state_dict": scheduler_G.state_dict() if scheduler_G is not None else None,
        "amp_scaler_state_dict": scaler.state_dict() if scaler is not None else None
    }

    if discriminator is not None and optimizer_D is not None:
        checkpoint["discriminator_state_dict"] = discriminator.state_dict()
        checkpoint["optimizer_D_state_dict"] = optimizer_D.state_dict()
        if scheduler_D is not None:
            checkpoint["scheduler_D_state_dict"] = scheduler_D.state_dict()
    
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()

    torch.save(checkpoint, path)
    tqdm.write(f"Checkpoint saved to {path}")


def maybe_save_checkpoint(
    epoch,
    config,
    model,
    optimizer_G,
    scheduler_G,
    scaler=None,
    ema_model=None,
    discriminator=None,
    optimizer_D=None,
    scheduler_D=None,
    use_amp=False,
    use_ema=False,
    gan_enabled=False,
):
    save_freq = config["training"]["save_checkpoint_freq"]
    save_dir = config["training"]["save_checkpoint_dir"]
    
    should_save = ((epoch + 1) % save_freq == 0)
        
    if should_save:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer_G=optimizer_G,
            scheduler_G=scheduler_G,
            scaler=scaler if use_amp else None,
            ema_model=ema_model if use_ema else None,
            discriminator=discriminator if gan_enabled else None,
            optimizer_D=optimizer_D if gan_enabled else None,
            scheduler_D=scheduler_D if gan_enabled else None,
            path=ckpt_path
        )


def save_final_models(
    model,
    ema_model=None,
    discriminator=None,
    config=None,
    output_dir=None,
    experiment_name=None,
):
    if config is not None:
        output_dir = output_dir or config["training"].get("final_model_dir", "final_models")
        experiment_name = experiment_name or config.get("logging", {}).get("name", "default_experiment")
    else:
        output_dir = output_dir or "final_models"
        experiment_name = experiment_name or "default_experiment"
    
    os.makedirs(output_dir, exist_ok=True)

    # Generator
    model_path = os.path.join(output_dir, f"{experiment_name}_G.pth")
    torch.save(model.state_dict(), model_path)

    # EMA model
    if ema_model is not None:
        ema_path = os.path.join(output_dir, f"{experiment_name}_G_EMA.pth")
        torch.save(ema_model.state_dict(), ema_path)

    # Discriminator
    if discriminator is not None:
        disc_path = os.path.join(output_dir, f"{experiment_name}_D.pth")
        torch.save(discriminator.state_dict(), disc_path)

    print(f"Saved final models to {output_dir}")


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer_G,
    scheduler_G,
    scaler,
    ema_model=None,
    discriminator=None,
    optimizer_D=None,
    scheduler_D=None,
    map_location="cpu",
):
    tqdm.write(f"Loading checkpoint from {checkpoint_path} ...")
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    
    model.load_state_dict(ckpt["model_state_dict"])
    if ema_model is not None:
        ema_model.load_state_dict(ckpt["ema_state_dict"])

    optimizer_G.load_state_dict(ckpt["optimizer_G_state_dict"])
    if ckpt.get("scheduler_G_state_dict", None) is not None and scheduler_G is not None:
        scheduler_G.load_state_dict(ckpt["scheduler_G_state_dict"])
    if ckpt.get("amp_scaler_state_dict", None) is not None and scaler is not None:
        scaler.load_state_dict(ckpt["amp_scaler_state_dict"])

    if discriminator is not None and optimizer_D is not None:
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        optimizer_D.load_state_dict(ckpt["optimizer_D_state_dict"])
        if ckpt.get("scheduler_D_state_dict", None) is not None and scheduler_D is not None:
            scheduler_D.load_state_dict(ckpt["scheduler_D_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    tqdm.write(f"Resuming training from epoch {start_epoch}")
    return start_epoch


def load_checkpoint_if_exists(
    config, model, optimizer_G, scheduler_G, scaler,
    ema_model=None, discriminator=None, optimizer_D=None, scheduler_D=None, device="cpu"
):
    resume_mode = config["training"].get("resume_mode", "resume")
    resume_checkpoint_path = config["training"].get("resume_checkpoint_path", None)
    use_ema = config["training"].get("use_ema", False)
    start_epoch = 0

    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        if resume_mode == "resume":
            start_epoch = load_checkpoint(
                checkpoint_path=resume_checkpoint_path,
                model=model,
                optimizer_G=optimizer_G,
                scheduler_G=scheduler_G,
                scaler=scaler,
                ema_model=ema_model,
                discriminator=discriminator,
                optimizer_D=optimizer_D,
                scheduler_D=scheduler_D,
                map_location=device
            )
            print(f"[Resume Mode] Resumed from epoch {start_epoch}.")

        elif resume_mode == "finetune":
            print(f"[Finetune Mode] Loading only generator weights from {resume_checkpoint_path}")
            ckpt = torch.load(resume_checkpoint_path, map_location=device)
            
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            
            if use_ema:
                src = model.module if hasattr(model, 'module') else model
                ema_model.load_state_dict(src.state_dict())

            start_epoch = 0
            print("[Finetune Mode] Generator weights loaded. Starting new training run at epoch 0.")
        else:
            print(f"Unknown resume_mode: {resume_mode}. Not loading checkpoint.")
    else:
        print("No checkpoint path provided or file does not exist. Starting from scratch.")
        start_epoch = 0

    return start_epoch
