from utils.traceback import *  # noqa: F401, F403
from utils.config import load_config, get_lambda_dict
from utils.logging import setup_tensorboard, log_training_scalars, log_validation_images
from models.factory import build_generator, build_ema_model, build_discriminator
from losses.factory import build_losses
from datasets.factory import build_dataloaders, apply_resume_skip_to_train_loader
from utils.optimizer import build_optimizers
from utils.scheduler import build_schedulers
from utils.checkpoint import load_checkpoint_if_exists, maybe_save_checkpoint, save_final_models
from engine.trainer import train_one_epoch, validate_one_epoch
import torch
import argparse
from torchinfo import summary


def main():
    parser = argparse.ArgumentParser(description="Train a VSR model.")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tensorboard_writer = setup_tensorboard(config)
    
    losses = build_losses(config, DEVICE)
    
    model = build_generator(config, DEVICE)
    ema_model = build_ema_model(config, model, DEVICE)
    discriminator = build_discriminator(config, DEVICE)

    train_loader, val_loader, train_loader_len, _, _, _ = build_dataloaders(config)
    
    optimizer_G, optimizer_D = build_optimizers(config, model, discriminator)
    scheduler_G, scheduler_D = build_schedulers(config, train_loader_len, optimizer_G, optimizer_D)
    
    scaler = torch.amp.GradScaler(AMP_DEVICE)
    
    start_epoch, global_step, step_in_epoch = load_checkpoint_if_exists(
        config, model, optimizer_G, scheduler_G, scaler, ema_model, discriminator, optimizer_D, scheduler_D, DEVICE
    )
    end_epoch = config["training"]["epochs"]

    # If checkpoint is at/after the end of the epoch, start next epoch instead
    if step_in_epoch >= train_loader_len:
        start_epoch += 1
        step_in_epoch = 0

    train_loader = apply_resume_skip_to_train_loader(train_loader, step_in_epoch)

    print("\n" + "=" * 72)
    print("Model Summary | Generator (G)")
    print("=" * 72)
    summary(model, input_size=(config["training"]["batch_size"], config["dataset"]["train"]["num_frames"], 3, config["dataset"]["train"]["patch_size"], config["dataset"]["train"]["patch_size"]))
    print()
    if config.get("gan", {}).get("enabled", False):
        print("\n" + "=" * 72)
        print("Model Summary | Discriminator (D)")
        print("=" * 72)
        summary(discriminator, input_size=(config["training"]["batch_size"], config["dataset"]["train"]["num_frames"], 3, config["dataset"]["train"]["patch_size"], config["dataset"]["train"]["patch_size"]))
        print()

    # Training loop
    for epoch in range(start_epoch, end_epoch):
        avg_train_loss, train_loss_details, global_step = train_one_epoch(
            model,
            ema_model,
            discriminator,
            optimizer_G,
            optimizer_D,
            scheduler_G,
            scheduler_D,
            scaler,
            train_loader,
            losses,
            DEVICE,
            config,
            epoch,
            global_step,
            step_in_epoch,
            lambda_dict=get_lambda_dict(config)
        )

        step_in_epoch = 0

        avg_val_loss, avg_psnr, avg_ssim, sample_input, sample_fake, sample_real, sample_ema, sample_clean = validate_one_epoch(
            model, ema_model, val_loader, losses, DEVICE, config, epoch, get_lambda_dict(config))
        
        if tensorboard_writer:
            log_training_scalars(
                tensorboard_writer, avg_train_loss, avg_val_loss, optimizer_G,
                avg_psnr, avg_ssim, train_loss_details, epoch
            )
            log_validation_images(
                tensorboard_writer, sample_input, sample_fake, sample_real, sample_ema, sample_clean, epoch
            )
    
        maybe_save_checkpoint(
            epoch=epoch,
            config=config,
            model=model,
            optimizer_G=optimizer_G,
            scheduler_G=scheduler_G,
            scaler=scaler,
            ema_model=ema_model,
            discriminator=discriminator,
            optimizer_D=optimizer_D,
            scheduler_D=scheduler_D,
            global_step=global_step,
            step_in_epoch=step_in_epoch
        )
    
    if tensorboard_writer:
        tensorboard_writer.close()
    
    save_final_models(
        model,
        ema_model=ema_model,
        discriminator=discriminator,
        config=config
    )


if __name__ == "__main__":
    main()
